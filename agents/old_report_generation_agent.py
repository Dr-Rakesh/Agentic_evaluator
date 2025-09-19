import pandas as pd
import os
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
import io
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import json
from config import (
    AZURE_OPENAI_DEPLOYMENT_GPT4O_NAME,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    COGNITIVE_URL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("input_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("InputAgent")

class InputFileAgenticNode:
    """
    An agentic node that processes input files (CSV/XLSX) for the RAG system.
    This node validates and prepares the data for further processing.
    """
    
    def __init__(
        self,
        node_id: str = "file_input_node",
        llm_model: Optional[Any] = None,
    ):
        """
        Initialize the input file agentic node.
        
        Args:
            node_id: Unique identifier for this node
            llm_model: Language model to use for agent decisions (optional)
        """
        self.node_id = node_id
        self.state = {}
        self.required_columns = ["questions", "answers", "contexts", "llm_answer"]
        
        # Initialize LLM if provided
        self.llm_model = llm_model
        if llm_model:
            self._initialize_agent()
        
        logger.info(f"InputFileAgenticNode '{node_id}' initialized")
    
    def _initialize_agent(self) -> None:
        """Initialize the agent with appropriate tools and prompt if LLM is available"""
        logger.info("Initializing agent with LLM capabilities")
        
        # Define tools for the agent
        tools = [
            Tool(
                name="validate_file_format",
                func=self.validate_file_format,
                description="Validate if the file format (CSV/XLSX) is correct"
            ),
            Tool(
                name="validate_required_columns",
                func=self.validate_required_columns,
                description="Validate if the file contains all required columns"
            ),
            Tool(
                name="process_file_data",
                func=self.process_file_data,
                description="Process the validated file data"
            )
        ]
        
        # Define prompt template
        prompt_template = """
        You are an intelligent agent responsible for validating and processing input files.
        Your node ID is {node_id}.
        
        Your task is to validate that uploaded files meet the requirements for the RAG system.
        
        Use the following tools to accomplish your task:
        {tools}
        
        Current state of knowledge:
        {state}
        
        Input: {input}
        
        Think through this step by step:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "state"],
            partial_variables={
                "node_id": self.node_id,
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            }
        )
        
        # Create the agent
        agent = create_react_agent(self.llm_model, tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        logger.info("Agent executor initialized")
    
    def process_file(self, file_content: bytes, filename: str) -> Dict:
        """
        Process the uploaded file content and validate it meets requirements.
        
        Args:
            file_content: The raw file content in bytes
            filename: The name of the uploaded file
            
        Returns:
            Dict containing validation results and processed data if valid
        """
        logger.info(f"Processing file: {filename}")
        
        try:
            # Check file extension
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension not in ['.csv', '.xlsx']:
                error_msg = f"Unsupported file format: {file_extension}. Only CSV and XLSX files are supported."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Read the file based on its extension
            df = self._read_file(file_content, file_extension)
            if isinstance(df, dict) and not df.get("success", True):
                return df  # Return error if file reading failed
            
            # Validate required columns
            validation_result = self.validate_required_columns(df)
            if not validation_result["success"]:
                return validation_result
            
            # Process the data
            processed_data = self.process_file_data(df)
            
            # Update state
            self.state.update({
                "last_processed_file": filename,
                "row_count": len(df),
                "processing_time": processed_data.get("processing_time", 0)
            })
            
            return {
                "success": True,
                "message": "File processed successfully",
                "data": processed_data["data"],
                "metadata": {
                    "filename": filename,
                    "row_count": len(df),
                    "columns": list(df.columns)
                }
            }
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.exception(error_msg)
            return {"success": False, "error": error_msg}
    
    def _read_file(self, file_content: bytes, file_extension: str) -> Union[pd.DataFrame, Dict]:
        """
        Read the file content into a pandas DataFrame.
        
        Args:
            file_content: The raw file content in bytes
            file_extension: The file extension to determine the reading method
            
        Returns:
            pandas DataFrame or Dict with error information
        """
        try:
            file_obj = io.BytesIO(file_content)
            
            if file_extension == '.csv':
                logger.info("Reading CSV file")
                df = pd.read_csv(file_obj)
            else:  # .xlsx
                logger.info("Reading Excel file")
                df = pd.read_excel(file_obj)
            
            # Check if the DataFrame is empty
            if df.empty:
                error_msg = "The uploaded file is empty."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            return df
            
        except pd.errors.EmptyDataError:
            error_msg = "The uploaded file is empty."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except pd.errors.ParserError:
            error_msg = "The file format is incorrect or corrupted."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            logger.exception(error_msg)
            return {"success": False, "error": error_msg}
    
    def validate_file_format(self, file_info: Union[str, Dict]) -> Dict:
        """
        Validate the file format.
        
        Args:
            file_info: File information as string or dict
            
        Returns:
            Validation result
        """
        try:
            if isinstance(file_info, str):
                file_info = json.loads(file_info)
            
            filename = file_info.get("filename", "")
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension not in ['.csv', '.xlsx']:
                error_msg = f"Unsupported file format: {file_extension}. Only CSV and XLSX files are supported."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            return {"success": True, "message": "File format is valid"}
            
        except Exception as e:
            error_msg = f"Error validating file format: {str(e)}"
            logger.exception(error_msg)
            return {"success": False, "error": error_msg}
    
    def validate_required_columns(self, df: Union[pd.DataFrame, str, Dict]) -> Dict:
        """
        Validate if the DataFrame contains all required columns (case-insensitive).
        
        Args:
            df: pandas DataFrame or serialized representation
            
        Returns:
            Validation result
        """
        try:
            # Handle different input types
            if isinstance(df, str):
                try:
                    df = pd.read_json(df)
                except:
                    return {"success": False, "error": "Invalid DataFrame JSON representation"}
            elif isinstance(df, dict) and "data" in df:
                try:
                    df = pd.DataFrame(df["data"])
                except:
                    return {"success": False, "error": "Invalid DataFrame dictionary representation"}
            
            # Convert all column names to lowercase for case-insensitive comparison
            df_columns_lower = [col.lower() for col in df.columns]
            required_columns_lower = [col.lower() for col in self.required_columns]
            
            # Check if all required columns are present
            missing_columns = [col for col in required_columns_lower if col not in df_columns_lower]
            
            if missing_columns:
                error_msg = f"Missing required columns: {', '.join(missing_columns)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Map actual column names to expected column names for later processing
            self.column_mapping = {}
            for req_col in self.required_columns:
                for actual_col in df.columns:
                    if actual_col.lower() == req_col.lower():
                        self.column_mapping[req_col] = actual_col
                        break
            
            logger.info(f"Column mapping: {self.column_mapping}")
            return {"success": True, "message": "All required columns are present"}
            
        except Exception as e:
            error_msg = f"Error validating required columns: {str(e)}"
            logger.exception(error_msg)
            return {"success": False, "error": error_msg}
    
    def process_file_data(self, df: Union[pd.DataFrame, str, Dict]) -> Dict:
        """
        Process the validated file data.
        
        Args:
            df: pandas DataFrame or serialized representation
            
        Returns:
            Processed data result
        """
        try:
            # Handle different input types
            if isinstance(df, str):
                try:
                    df = pd.read_json(df)
                except:
                    return {"success": False, "error": "Invalid DataFrame JSON representation"}
            elif isinstance(df, dict) and "data" in df:
                try:
                    df = pd.DataFrame(df["data"])
                except:
                    return {"success": False, "error": "Invalid DataFrame dictionary representation"}
            
            # Validate required columns if not already done
            if not hasattr(self, 'column_mapping'):
                validation_result = self.validate_required_columns(df)
                if not validation_result["success"]:
                    return validation_result
            
            # Normalize the data using the column mapping
            normalized_data = []
            for _, row in df.iterrows():
                normalized_row = {}
                for expected_col, actual_col in self.column_mapping.items():
                    normalized_row[expected_col] = row[actual_col]
                normalized_data.append(normalized_row)
            
            # Check if the "contexts" column contains list-like objects
            for i, row in enumerate(normalized_data):
                if isinstance(row["contexts"], str):
                    try:
                        # Try to parse JSON string as list
                        contexts_list = json.loads(row["contexts"])
                        if isinstance(contexts_list, list):
                            row["contexts"] = contexts_list
                    except json.JSONDecodeError:
                        # If not JSON, treat as a single context
                        row["contexts"] = [row["contexts"]]
                elif not isinstance(row["contexts"], list):
                    # If not a list or string, convert to a list with one item
                    row["contexts"] = [str(row["contexts"])]
            
            logger.info(f"Successfully processed {len(normalized_data)} rows of data")
            return {
                "success": True,
                "message": "Data processed successfully",
                "data": normalized_data,
                "processing_time": 0.0  # Placeholder for actual processing time
            }
            
        except Exception as e:
            error_msg = f"Error processing file data: {str(e)}"
            logger.exception(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_state(self) -> Dict:
        """Get the current state of the agent node"""
        return self.state


def create_input_agent() -> InputFileAgenticNode:
    """
    Create an instance of the InputFileAgenticNode with Azure OpenAI LLM integration.
    
    Returns:
        Configured InputFileAgenticNode
    """
    try:
        # Configure Azure OpenAI
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_URL)
        
        llm = AzureChatOpenAI(
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_GPT4O_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=token_provider
        )
        
        # Create and return the agent node
        return InputFileAgenticNode(llm_model=llm)
    except Exception as e:
        logger.exception(f"Failed to create input agent with LLM: {str(e)}")
        # Fallback to agent without LLM capabilities
        return InputFileAgenticNode()


# Example usage function
def process_uploaded_file(file_content: bytes, filename: str) -> Dict:
    """
    Process an uploaded file using the input agent.
    
    Args:
        file_content: The raw file content in bytes
        filename: The name of the uploaded file
        
    Returns:
        Processing result
    """
    agent = create_input_agent()
    return agent.process_file(file_content, filename)


# If the module is run directly, perform a test
if __name__ == "__main__":
    try:
        logger.info("Testing InputFileAgenticNode")
        
        # Create test data (simulating a CSV in memory)
        test_data = """questions,answers,contexts,llm_answer
How can I customize Active Workspace?,You can configure nearly every aspect of the commands for the Active Workspace interface.,"User interface configuration for Active Workspace",This is a sample LLM answer
What is the purpose of this tool?,This tool helps evaluate LLM responses.,"Context about evaluation",Another sample LLM answer"""
        
        # Convert test data to bytes
        test_file_content = test_data.encode('utf-8')
        
        # Process the test file
        result = process_uploaded_file(test_file_content, "test_file.csv")
        
        # Log the result
        logger.info(f"Test result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        logger.exception(f"Test failed: {str(e)}")