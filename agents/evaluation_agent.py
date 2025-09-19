import pandas as pd
import logging
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

# DeepEval imports
from deepeval.metrics import (
    AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, 
    ToxicityMetric, BiasMetric, SummarizationMetric, 
    ContextualPrecisionMetric, ContextualRecallMetric, 
    ContextualRelevancyMetric, GEval
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM

# Azure OpenAI imports
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded Azure credentials (as requested)
AZURE_OPENAI_ENDPOINT = "https://openai-aiattack-msa-001758-swedencentral-adi.openai.azure.com"
AZURE_OPENAI_DEPLOYMENT_GPT4O_NAME = "gpt-4o-2"
AZURE_OPENAI_API_VERSION = "2024-06-01"
COGNITIVE_URL = "https://cognitiveservices.azure.com/.default"

class AzureOpenAI(DeepEvalBaseLLM):
    """Custom Azure OpenAI wrapper for DeepEval framework."""
    
    def __init__(self, model):
        self.model = model
        logger.info("AzureOpenAI wrapper initialized")

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        try:
            chat_model = self.load_model()
            response = chat_model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def a_generate(self, prompt: str) -> str:
        try:
            chat_model = self.load_model()
            res = await chat_model.ainvoke(prompt)
            return res.content
        except Exception as e:
            logger.error(f"Error generating async response: {str(e)}")
            raise

    def get_model_name(self):
        return "Custom Azure OpenAI GPT-4o Model"

class EvaluationAgent:
    """
    Agent responsible for evaluating LLM responses using DeepEval metrics
    with Azure OpenAI as the judge model.
    """
    
    def __init__(self):
        """Initialize the EvaluationAgent with Azure OpenAI configuration."""
        logger.info("Initializing EvaluationAgent...")
        
        try:
            # Initialize Azure OpenAI with DefaultAzureCredential
            self.azure_openai = self._setup_azure_openai()
            self.metrics = self._setup_metrics()
            logger.info("EvaluationAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EvaluationAgent: {str(e)}")
            raise
    
    def _setup_azure_openai(self) -> AzureOpenAI:
        """Setup Azure OpenAI client with authentication."""
        try:
            logger.info("Setting up Azure OpenAI client...")
            
            # Get Azure credential and token
            credential = DefaultAzureCredential()
            token = credential.get_token(COGNITIVE_URL).token
            
            # Create Azure ChatOpenAI instance
            custom_model = AzureChatOpenAI(
                openai_api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT_GPT4O_NAME,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_ad_token=token
            )
            
            # Wrap in DeepEval compatible class
            azure_openai = AzureOpenAI(model=custom_model)
            
            logger.info("Azure OpenAI client setup completed")
            return azure_openai
            
        except Exception as e:
            logger.error(f"Failed to setup Azure OpenAI: {str(e)}")
            raise
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup all evaluation metrics."""
        try:
            logger.info("Setting up evaluation metrics...")
            
            metrics = {
                "contextual_precision": ContextualPrecisionMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "contextual_recall": ContextualRecallMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "contextual_relevancy": ContextualRelevancyMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "answer_relevancy": AnswerRelevancyMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "faithfulness": FaithfulnessMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "hallucination": HallucinationMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "bias": BiasMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "toxicity": ToxicityMetric(
                    model=self.azure_openai, 
                    async_mode=False
                ),
                "g_eval": GEval(
                    model=self.azure_openai,
                    name="Coherence",
                    criteria="Coherence - determine if the actual output is coherent with the input.",
                    evaluation_params=[
                        LLMTestCaseParams.INPUT, 
                        LLMTestCaseParams.EXPECTED_OUTPUT, 
                        LLMTestCaseParams.ACTUAL_OUTPUT
                    ],
                    async_mode=False
                ),
                "summarization": SummarizationMetric(
                    threshold=0.5,
                    model=self.azure_openai,
                    strict_mode=False,
                    async_mode=False
                )
            }
            
            logger.info(f"Successfully setup {len(metrics)} evaluation metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {str(e)}")
            raise
    
    def _create_test_cases(self, df: pd.DataFrame) -> List[Dict[str, LLMTestCase]]:
        """Create test cases for each row in the DataFrame."""
        try:
            logger.info(f"Creating test cases for {len(df)} rows...")
            
            test_cases_list = []
            
            for index, row in df.iterrows():
                # Extract data from row
                question = str(row['questions'])
                answer = str(row['answers'])
                llm_answer = str(row['llm_answer'])
                
                # Handle contexts - convert to list if it's a string
                contexts = row['contexts']
                if isinstance(contexts, str):
                    # Split by common delimiters or treat as single context
                    context_list = [contexts.strip()]
                elif isinstance(contexts, list):
                    context_list = [str(ctx).strip() for ctx in contexts]
                else:
                    context_list = [str(contexts).strip()]
                
                # Create standard test case template
                test_case_template = LLMTestCase(
                    input=question,
                    actual_output=llm_answer,
                    expected_output=answer,
                    retrieval_context=context_list,
                    context=context_list
                )
                
                # Create specific test cases for different metrics
                test_cases = {
                    "summarization": LLMTestCase(
                        input=context_list[0] if context_list else "",
                        actual_output=llm_answer
                    ),
                    "g_eval": test_case_template,
                    "contextual_precision": test_case_template,
                    "contextual_recall": test_case_template,
                    "contextual_relevancy": test_case_template,
                    "answer_relevancy": test_case_template,
                    "faithfulness": test_case_template,
                    "hallucination": test_case_template,
                    "bias": test_case_template,
                    "toxicity": test_case_template
                }
                
                test_cases_list.append({
                    'row_index': index,
                    'test_cases': test_cases,
                    'original_data': {
                        'question': question,
                        'answer': answer,
                        'llm_answer': llm_answer,
                        'contexts': context_list
                    }
                })
            
            logger.info(f"Successfully created test cases for {len(test_cases_list)} rows")
            return test_cases_list
            
        except Exception as e:
            logger.error(f"Error creating test cases: {str(e)}")
            raise
    
    def _evaluate_single_metric(self, metric_name: str, metric: Any, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate a single metric for a test case."""
        try:
            start_time = time.time()
            logger.debug(f"Evaluating metric: {metric_name}")
            
            # Measure the metric
            metric.measure(test_case)
            
            # Extract results
            result = {
                "metric_name": metric_name,
                "score": round(float(metric.score), 4) if hasattr(metric, 'score') and metric.score is not None else 0.0,
                "reason": getattr(metric, 'reason', 'No reason provided'),
                "success": True,
                "evaluation_time": round(time.time() - start_time, 2)
            }
            
            logger.debug(f"Successfully evaluated {metric_name} - Score: {result['score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
            return {
                "metric_name": metric_name,
                "score": 0.0,
                "reason": f"Evaluation failed: {str(e)}",
                "success": False,
                "error": str(e),
                "evaluation_time": 0.0
            }
    
    def _evaluate_row(self, row_data: Dict) -> Dict[str, Any]:
        """Evaluate all metrics for a single row."""
        try:
            row_index = row_data['row_index']
            test_cases = row_data['test_cases']
            original_data = row_data['original_data']
            
            logger.info(f"Evaluating row {row_index}...")
            
            row_results = {
                'row_index': row_index,
                'original_data': original_data,
                'metrics': {},
                'summary': {
                    'total_metrics': len(self.metrics),
                    'successful_evaluations': 0,
                    'failed_evaluations': 0,
                    'average_score': 0.0,
                    'evaluation_time': 0.0
                }
            }
            
            start_time = time.time()
            total_score = 0.0
            successful_count = 0
            
            # Evaluate each metric
            for metric_name, metric in self.metrics.items():
                try:
                    # Get appropriate test case for this metric
                    test_case = test_cases.get(metric_name, test_cases['g_eval'])
                    
                    # Evaluate the metric
                    result = self._evaluate_single_metric(metric_name, metric, test_case)
                    row_results['metrics'][metric_name] = result
                    
                    if result['success']:
                        successful_count += 1
                        total_score += result['score']
                    else:
                        row_results['summary']['failed_evaluations'] += 1
                        
                except Exception as e:
                    logger.error(f"Error evaluating {metric_name} for row {row_index}: {str(e)}")
                    row_results['metrics'][metric_name] = {
                        "metric_name": metric_name,
                        "score": 0.0,
                        "reason": f"Evaluation failed: {str(e)}",
                        "success": False,
                        "error": str(e),
                        "evaluation_time": 0.0
                    }
                    row_results['summary']['failed_evaluations'] += 1
            
            # Calculate summary statistics
            row_results['summary']['successful_evaluations'] = successful_count
            row_results['summary']['average_score'] = round(
                total_score / successful_count if successful_count > 0 else 0.0, 4
            )
            row_results['summary']['evaluation_time'] = round(time.time() - start_time, 2)
            
            logger.info(f"Completed evaluation for row {row_index} - "
                       f"Success rate: {successful_count}/{len(self.metrics)} - "
                       f"Average score: {row_results['summary']['average_score']}")
            
            return row_results
            
        except Exception as e:
            logger.error(f"Error evaluating row {row_data['row_index']}: {str(e)}")
            return {
                'row_index': row_data['row_index'],
                'error': str(e),
                'success': False
            }
    
    def evaluate_dataframe(self, df: pd.DataFrame, max_workers: int = 3) -> Dict[str, Any]:
        """
        Evaluate all rows in the DataFrame using specified metrics.
        
        Args:
            df (pd.DataFrame): DataFrame with required columns
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        try:
            logger.info(f"Starting evaluation of DataFrame with {len(df)} rows...")
            start_time = time.time()
            
            # Validate DataFrame
            required_columns = ['questions', 'answers', 'contexts', 'llm_answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Create test cases
            test_cases_list = self._create_test_cases(df)
            
            # Initialize results structure
            evaluation_results = {
                'metadata': {
                    'total_rows': len(df),
                    'total_metrics': len(self.metrics),
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_name': self.azure_openai.get_model_name(),
                    'max_workers': max_workers
                },
                'results': [],
                'summary': {
                    'successful_rows': 0,
                    'failed_rows': 0,
                    'total_evaluations': 0,
                    'successful_evaluations': 0,
                    'failed_evaluations': 0,
                    'overall_average_score': 0.0,
                    'metric_averages': {},
                    'total_evaluation_time': 0.0
                }
            }
            
            # Execute evaluations with parallel processing
            logger.info(f"Starting parallel evaluation with {max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_row = {
                    executor.submit(self._evaluate_row, row_data): row_data['row_index']
                    for row_data in test_cases_list
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_row):
                    row_index = future_to_row[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per row
                        evaluation_results['results'].append(result)
                        
                        if result.get('success', True):
                            evaluation_results['summary']['successful_rows'] += 1
                        else:
                            evaluation_results['summary']['failed_rows'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing row {row_index}: {str(e)}")
                        evaluation_results['results'].append({
                            'row_index': row_index,
                            'error': str(e),
                            'success': False
                        })
                        evaluation_results['summary']['failed_rows'] += 1
            
            # Calculate summary statistics
            self._calculate_summary_statistics(evaluation_results)
            
            total_time = time.time() - start_time
            evaluation_results['summary']['total_evaluation_time'] = round(total_time, 2)
            
            logger.info(f"Evaluation completed in {total_time:.2f} seconds")
            logger.info(f"Successfully evaluated {evaluation_results['summary']['successful_rows']}/{len(df)} rows")
            logger.info(f"Overall average score: {evaluation_results['summary']['overall_average_score']}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during DataFrame evaluation: {str(e)}")
            raise
    
    def _calculate_summary_statistics(self, evaluation_results: Dict[str, Any]) -> None:
        """Calculate summary statistics for the evaluation results."""
        try:
            metric_scores = {metric_name: [] for metric_name in self.metrics.keys()}
            total_successful_evaluations = 0
            total_failed_evaluations = 0
            all_scores = []
            
            for result in evaluation_results['results']:
                if result.get('success', True) and 'metrics' in result:
                    for metric_name, metric_result in result['metrics'].items():
                        if metric_result.get('success', False):
                            score = metric_result.get('score', 0.0)
                            metric_scores[metric_name].append(score)
                            all_scores.append(score)
                            total_successful_evaluations += 1
                        else:
                            total_failed_evaluations += 1
                else:
                    # Count all metrics as failed for this row
                    total_failed_evaluations += len(self.metrics)
            
            # Calculate metric averages
            for metric_name, scores in metric_scores.items():
                if scores:
                    evaluation_results['summary']['metric_averages'][metric_name] = {
                        'average_score': round(sum(scores) / len(scores), 4),
                        'min_score': round(min(scores), 4),
                        'max_score': round(max(scores), 4),
                        'total_evaluations': len(scores)
                    }
                else:
                    evaluation_results['summary']['metric_averages'][metric_name] = {
                        'average_score': 0.0,
                        'min_score': 0.0,
                        'max_score': 0.0,
                        'total_evaluations': 0
                    }
            
            # Calculate overall statistics
            evaluation_results['summary']['total_evaluations'] = total_successful_evaluations + total_failed_evaluations
            evaluation_results['summary']['successful_evaluations'] = total_successful_evaluations
            evaluation_results['summary']['failed_evaluations'] = total_failed_evaluations
            evaluation_results['summary']['overall_average_score'] = round(
                sum(all_scores) / len(all_scores) if all_scores else 0.0, 4
            )
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
    
    def get_metric_scores_only(self, evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract only the metric scores from evaluation results.
        
        Args:
            evaluation_results (Dict[str, Any]): Complete evaluation results
            
        Returns:
            Dict[str, Dict[str, float]]: Simplified metric scores by row
        """
        scores_only = {}
        
        for result in evaluation_results['results']:
            if result.get('success', True) and 'metrics' in result:
                row_index = result['row_index']
                scores_only[f"row_{row_index}"] = {}
                
                for metric_name, metric_result in result['metrics'].items():
                    scores_only[f"row_{row_index}"][metric_name] = metric_result.get('score', 0.0)
        
        return scores_only

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    test_data = {
        'questions': [
            "How can I customize Active Workspace?",
            "How does it work?"
        ],
        'answers': [
            "You can configure nearly every aspect of the commands for the Active Workspace interface.",
            "When the user selects an object, the universal viewer builds a list of available files..."
        ],
        'contexts': [
            "User interface configuration for Active Workspace. How do I change the user interface...",
            "How do I? The following is a set of best practices for commonly asked questions..."
        ],
        'llm_answer': [
            "Hello! Customizing Active Workspace in Teamcenter can significantly enhance your user experience...",
            "Hello! Thank you for your question about Siemens PLM software. Siemens PLM products..."
        ]
    }
    
    # Create test DataFrame
    test_df = pd.DataFrame(test_data)
    
    try:
        # Initialize evaluation agent
        agent = EvaluationAgent()
        
        # Run evaluation
        results = agent.evaluate_dataframe(test_df, max_workers=2)
        
        # Print results
        print("✅ Evaluation completed successfully!")
        print(f"📊 Overall average score: {results['summary']['overall_average_score']}")
        print(f"⏱️ Total evaluation time: {results['summary']['total_evaluation_time']} seconds")
        
        print("\n📈 Metric Averages:")
        for metric_name, stats in results['summary']['metric_averages'].items():
            print(f"  {metric_name}: {stats['average_score']}")
        
        # Get simplified scores
        scores_only = agent.get_metric_scores_only(results)
        print(f"\n🎯 Simplified scores structure created for {len(scores_only)} rows")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        logger.error(f"Test execution failed: {str(e)}")