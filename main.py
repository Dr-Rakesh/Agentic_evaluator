import asyncio
import io
import logging
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# Add this after the existing imports
import traceback
from fastapi import Request

# Add this import at the top
import time
# Import agents from the agents folder
import sys
sys.path.append('agents')

try:
    from agents.input_agent import InputAgent
    logger = logging.getLogger(__name__)
    logger.info("InputAgent imported successfully")
except ImportError as e:
    logger.error(f"Failed to import InputAgent: {e}")
    InputAgent = None

try:
    from agents.evaluation_agent import EvaluationAgent
    logger.info("EvaluationAgent imported successfully")
except ImportError as e:
    logger.error(f"Failed to import EvaluationAgent: {e}")
    EvaluationAgent = None

try:
    from agents.mathematical_evaluation_agent import MathematicalEvaluationAgent
    logger.info("MathematicalEvaluationAgent imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MathematicalEvaluationAgent: {e}")
    MathematicalEvaluationAgent = None

try:
    from agents.report_generation_agent import FlexibleReportGenerationAgent, generate_evaluation_report
    logger.info("ReportGenerationAgent imported successfully")
except ImportError as e:
    logger.error(f"Failed to import ReportGenerationAgent: {e}")
    FlexibleReportGenerationAgent = None
    generate_evaluation_report = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="LLM Evaluation Platform",
    description="A comprehensive platform for evaluating Large Language Model responses using multiple evaluation frameworks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted successfully")
else:
    logger.warning("Static folder not found")

# Global variables to store evaluation results
evaluation_results_store = {}
task_status_store = {}

# Pydantic models for API requests/responses
class EvaluationRequest(BaseModel):
    file_id: str
    enable_deepeval: bool = True
    enable_mathematical: bool = True
    max_workers: int = 3

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0
    message: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result_id: Optional[str] = None

class EvaluationResults(BaseModel):
    deepeval_results: Optional[Dict[str, Any]] = None
    mathematical_results: Optional[Dict[str, Any]] = None
    input_data_summary: Dict[str, Any]
    evaluation_summary: Dict[str, Any]

# Utility functions
def generate_task_id() -> str:
    """Generate a unique task ID."""
    return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

def generate_result_id() -> str:
    """Generate a unique result ID."""
    return f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

def update_task_status(task_id: str, status: str, progress: float = 0.0, message: str = "", result_id: str = None):
    """Update task status in the store."""
    if task_id in task_status_store:
        task_status_store[task_id].update({
            "status": status,
            "progress": progress,
            "message": message,
            "result_id": result_id
        })
        if status in ["completed", "failed"]:
            task_status_store[task_id]["end_time"] = datetime.now().isoformat()
        logger.info(f"Task {task_id}: {status} - {message}")

async def run_evaluation_pipeline(
    dataframe: pd.DataFrame,
    task_id: str,
    enable_deepeval: bool = True,
    enable_mathematical: bool = True,
    max_workers: int = 3
):
    """Run the complete evaluation pipeline asynchronously."""
    try:
        update_task_status(task_id, "running", 10, "Starting evaluation pipeline...")
        
        results = {
            "deepeval_results": None,
            "mathematical_results": None,
            "input_data_summary": {
                "total_rows": len(dataframe),
                "columns": list(dataframe.columns),
                "sample_data": dataframe.head(3).to_dict('records') if len(dataframe) > 0 else []
            }
        }
        
        # Run DeepEval evaluation if enabled and available
        if enable_deepeval and EvaluationAgent is not None:
            try:
                update_task_status(task_id, "running", 20, "Running DeepEval metrics...")
                eval_agent = EvaluationAgent()
                results["deepeval_results"] = eval_agent.evaluate_dataframe(dataframe, max_workers)
                update_task_status(task_id, "running", 50, "DeepEval evaluation completed")
                logger.info("DeepEval evaluation completed successfully")
            except Exception as e:
                logger.error(f"DeepEval evaluation failed: {str(e)}")
                update_task_status(task_id, "running", 50, f"DeepEval evaluation failed: {str(e)}")
        else:
            update_task_status(task_id, "running", 50, "DeepEval evaluation skipped")
        
        # Run Mathematical evaluation if enabled and available
        if enable_mathematical and MathematicalEvaluationAgent is not None:
            try:
                update_task_status(task_id, "running", 60, "Running Mathematical metrics...")
                math_agent = MathematicalEvaluationAgent()
                results["mathematical_results"] = math_agent.evaluate_dataframe(dataframe)
                update_task_status(task_id, "running", 90, "Mathematical evaluation completed")
                logger.info("Mathematical evaluation completed successfully")
            except Exception as e:
                logger.error(f"Mathematical evaluation failed: {str(e)}")
                update_task_status(task_id, "running", 90, f"Mathematical evaluation failed: {str(e)}")
        else:
            update_task_status(task_id, "running", 90, "Mathematical evaluation skipped")
        
        # Create evaluation summary
        evaluation_summary = {
            "total_rows_processed": len(dataframe),
            "deepeval_enabled": enable_deepeval,
            "mathematical_enabled": enable_mathematical,
            "deepeval_success": results["deepeval_results"] is not None,
            "mathematical_success": results["mathematical_results"] is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add performance summaries if available
        if results["deepeval_results"]:
            evaluation_summary["deepeval_average_score"] = results["deepeval_results"].get("summary", {}).get("overall_average_score", 0.0)
        
        if results["mathematical_results"]:
            evaluation_summary["mathematical_average_score"] = results["mathematical_results"].get("summary", {}).get("overall_scores", {}).get("average_across_metrics", 0.0)
        
        results["evaluation_summary"] = evaluation_summary
        
        # Store results
        result_id = generate_result_id()
        evaluation_results_store[result_id] = results
        
        update_task_status(task_id, "completed", 100, "Evaluation pipeline completed successfully", result_id)
        logger.info(f"Evaluation pipeline completed successfully for task {task_id}")
        
    except Exception as e:
        error_msg = f"Evaluation pipeline failed: {str(e)}"
        logger.error(f"Task {task_id}: {error_msg}")
        logger.error(traceback.format_exc())
        update_task_status(task_id, "failed", 0, error_msg)

# API Routes
# Add this middleware for better debugging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    start_time = time.time()
    
    # Log the request
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log the response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} for {request.method} {request.url} (took {process_time:.3f}s)")
    
    return response

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working."""
    return {
        "status": "API is working",
        "timestamp": datetime.now().isoformat(),
        "message": "This is a test endpoint"
    }

@app.get("/")
async def root():
    """Serve the main frontend page."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return JSONResponse(
            content={"message": "LLM Evaluation Platform API", "status": "running"},
            status_code=200
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    agent_status = {
        "input_agent": InputAgent is not None,
        "evaluation_agent": EvaluationAgent is not None,
        "mathematical_agent": MathematicalEvaluationAgent is not None,
        "report_agent": FlexibleReportGenerationAgent is not None
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_available": agent_status,
        "total_agents": sum(agent_status.values())
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and validate CSV/XLSX file."""
    try:
        if not InputAgent:
            raise HTTPException(status_code=500, detail="InputAgent not available")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        
        # Process file using InputAgent
        input_agent = InputAgent()
        result = input_agent.process_file(content, file.filename)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Store the dataframe temporarily
        file_id = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        # Convert dataframe to JSON for storage (in a real app, use a proper database)
        dataframe_json = result['dataframe'].to_json(orient='records')
        
        # Store in a temporary way (in production, use proper storage)
        temp_file_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file_path.write(dataframe_json)
        temp_file_path.close()
        
        # Store file mapping
        if not hasattr(app.state, 'file_store'):
            app.state.file_store = {}
        
        app.state.file_store[file_id] = {
            'filename': file.filename,
            'temp_path': temp_file_path.name,
            'validation_details': result['validation_details'],
            'upload_time': datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "validation_details": result['validation_details'],
            "message": result['message']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/api/debug/endpoints")
async def list_endpoints():
    """List all available API endpoints for debugging."""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'Unknown')
            })
    
    return {
        "available_endpoints": routes,
        "total_endpoints": len(routes)
    }

@app.post("/api/evaluate")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start the evaluation process."""
    try:
        # Check if file exists
        if not hasattr(app.state, 'file_store') or request.file_id not in app.state.file_store:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = app.state.file_store[request.file_id]
        
        # Load dataframe from temporary storage
        with open(file_info['temp_path'], 'r') as f:
            dataframe_json = f.read()
        
        dataframe = pd.read_json(dataframe_json, orient='records')
        
        # Generate task ID
        task_id = generate_task_id()
        
        # Initialize task status
        task_status_store[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Evaluation queued",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result_id": None
        }
        
        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation_pipeline,
            dataframe,
            task_id,
            request.enable_deepeval,
            request.enable_mathematical,
            request.max_workers
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Evaluation started",
            "estimated_time": f"{len(dataframe) * 30} seconds"  # Rough estimate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation start error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of an evaluation task."""
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status_store[task_id]

@app.get("/api/results/{result_id}")
async def get_evaluation_results(result_id: str):
    """Get evaluation results."""
    if result_id not in evaluation_results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    results = evaluation_results_store[result_id]
    
    # Create simplified summary for frontend
    summary = {
        "input_data_summary": results["input_data_summary"],
        "evaluation_summary": results["evaluation_summary"]
    }
    
    # Add metric summaries if available
    if results["deepeval_results"]:
        summary["deepeval_summary"] = {
            "overall_score": results["deepeval_results"].get("summary", {}).get("overall_average_score", 0.0),
            "total_metrics": len(results["deepeval_results"].get("summary", {}).get("metric_averages", {})),
            "successful_rows": results["deepeval_results"].get("summary", {}).get("successful_rows", 0)
        }
    
    if results["mathematical_results"]:
        summary["mathematical_summary"] = {
            "overall_score": results["mathematical_results"].get("summary", {}).get("overall_scores", {}).get("average_across_metrics", 0.0),
            "total_metrics": len(results["mathematical_results"].get("summary", {}).get("metric_averages", {})),
            "successful_rows": results["mathematical_results"].get("summary", {}).get("successful_rows", 0)
        }
    
    return {
        "success": True,
        "results": summary,
        "has_deepeval": results["deepeval_results"] is not None,
        "has_mathematical": results["mathematical_results"] is not None
    }

@app.get("/api/results/{result_id}/detailed")
async def get_detailed_results(result_id: str, agent_type: str = "all"):
    """Get detailed evaluation results."""
    if result_id not in evaluation_results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    results = evaluation_results_store[result_id]
    
    response_data = {}
    
    if agent_type in ["all", "deepeval"] and results["deepeval_results"]:
        response_data["deepeval_results"] = results["deepeval_results"]
    
    if agent_type in ["all", "mathematical"] and results["mathematical_results"]:
        response_data["mathematical_results"] = results["mathematical_results"]
    
    if agent_type == "all":
        response_data["input_data_summary"] = results["input_data_summary"]
        response_data["evaluation_summary"] = results["evaluation_summary"]
    
    return response_data

@app.post("/api/generate-report/{result_id}")
async def generate_report(result_id: str, background_tasks: BackgroundTasks):
    """Generate a PDF report for the evaluation results."""
    try:
        if result_id not in evaluation_results_store:
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Import the enhanced function
        from agents.report_generation_agent import generate_enhanced_evaluation_report
        
        results = evaluation_results_store[result_id]
        
        # Create input dataframe from stored data
        input_data = None
        if results["input_data_summary"]["sample_data"]:
            input_data = pd.DataFrame(results["input_data_summary"]["sample_data"])
        
        # Generate enhanced report with individual row analysis
        pdf_buffer = generate_enhanced_evaluation_report(
            deepeval_results=results["deepeval_results"],
            math_results=results["mathematical_results"],
            input_data=input_data
        )
        
        # Return PDF as streaming response
        filename = f"enhanced_llm_evaluation_report_{result_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer.getvalue()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate enhanced report: {str(e)}")
        

@app.get("/api/tasks")
async def list_tasks():
    """List all evaluation tasks."""
    return {
        "tasks": list(task_status_store.values()),
        "total_tasks": len(task_status_store)
    }

@app.delete("/api/cleanup")
async def cleanup_old_data():
    """Clean up old temporary files and data."""
    try:
        cleaned_files = 0
        cleaned_tasks = 0
        cleaned_results = 0
        
        # Clean up temporary files
        if hasattr(app.state, 'file_store'):
            for file_id, file_info in list(app.state.file_store.items()):
                try:
                    if os.path.exists(file_info['temp_path']):
                        os.unlink(file_info['temp_path'])
                        cleaned_files += 1
                except Exception as e:
                    logger.warning(f"Could not delete temp file: {e}")
            
            app.state.file_store.clear()
        
        # Clean up old tasks (keep only last 100)
        if len(task_status_store) > 100:
            old_tasks = list(task_status_store.keys())[:-100]
            for task_id in old_tasks:
                del task_status_store[task_id]
                cleaned_tasks += 1
        
        # Clean up old results (keep only last 50)
        if len(evaluation_results_store) > 50:
            old_results = list(evaluation_results_store.keys())[:-50]
            for result_id in old_results:
                del evaluation_results_store[result_id]
                cleaned_results += 1
        
        return {
            "success": True,
            "cleaned_files": cleaned_files,
            "cleaned_tasks": cleaned_tasks,
            "cleaned_results": cleaned_results,
            "message": "Cleanup completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Global exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize app state on startup."""
    logger.info("LLM Evaluation Platform starting up...")
    
    # Initialize file store
    app.state.file_store = {}
    
    # Log available agents
    available_agents = []
    if InputAgent:
        available_agents.append("InputAgent")
    if EvaluationAgent:
        available_agents.append("EvaluationAgent")
    if MathematicalEvaluationAgent:
        available_agents.append("MathematicalEvaluationAgent")
    if FlexibleReportGenerationAgent:
        available_agents.append("ReportGenerationAgent")
    
    logger.info(f"Available agents: {', '.join(available_agents)}")
    logger.info("LLM Evaluation Platform started successfully!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("LLM Evaluation Platform shutting down...")
    
    # Clean up temporary files
    if hasattr(app.state, 'file_store'):
        for file_info in app.state.file_store.values():
            try:
                if os.path.exists(file_info['temp_path']):
                    os.unlink(file_info['temp_path'])
            except Exception as e:
                logger.warning(f"Could not delete temp file during shutdown: {e}")
    
    logger.info("LLM Evaluation Platform shutdown complete")

if __name__ == "__main__":
    # Run the FastAPI application
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    logger.info(f"Starting LLM Evaluation Platform on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )