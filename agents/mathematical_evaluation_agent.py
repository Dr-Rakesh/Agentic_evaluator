import pandas as pd
import logging
import math
import time
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime
import numpy as np

# NLTK imports for BLEU and METEOR
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# ROUGE score import
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathematicalEvaluationAgent:
    """
    Agent responsible for calculating mathematical evaluation metrics
    for LLM responses including perplexity, BLEU, ROUGE, and METEOR scores.
    """
    
    def __init__(self):
        """Initialize the Mathematical Evaluation Agent."""
        logger.info("Initializing Mathematical Evaluation Agent...")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # BLEU smoothing function
        self.smoothing_function = SmoothingFunction().method4
        
        logger.info("Mathematical Evaluation Agent initialized successfully")
    
    def calculate_perplexity_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate perplexity score for a given text.
        Perplexity measures how well a probability model predicts a sample.
        
        Args:
            text (str): Input text to calculate perplexity for
            
        Returns:
            Dict[str, Any]: Perplexity calculation results
        """
        try:
            start_time = time.time()
            logger.debug(f"Calculating perplexity for text of length {len(text)}")
            
            # Tokenize the text
            tokens = self._tokenize_text(text)
            
            if not tokens:
                return {
                    'score': float('inf'),
                    'raw_score': float('inf'),
                    'normalized_score': 0.0,
                    'token_count': 0,
                    'calculation_time': time.time() - start_time,
                    'success': False,
                    'error': 'Empty text or no valid tokens'
                }
            
            N = len(tokens)
            
            # Simple perplexity calculation based on token frequency
            # In a real scenario, you would use a trained language model
            token_freq = {}
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            # Calculate probability for each token (simple frequency-based)
            total_tokens = len(tokens)
            log_prob_sum = 0.0
            
            for token in tokens:
                # Add smoothing to avoid zero probabilities
                prob = (token_freq[token] + 1) / (total_tokens + len(token_freq))
                log_prob_sum += math.log(prob)
            
            # Calculate perplexity: 2^(-1/N * sum(log2(P(xi))))
            avg_log_prob = log_prob_sum / N
            perplexity = math.exp(-avg_log_prob)
            
            # Normalize perplexity to a 0-1 scale (lower perplexity is better)
            # Using a sigmoid-like transformation
            normalized_score = 1 / (1 + math.log(perplexity + 1))
            
            result = {
                'score': round(normalized_score, 4),
                'raw_score': round(perplexity, 4),
                'normalized_score': round(normalized_score, 4),
                'token_count': N,
                'unique_tokens': len(token_freq),
                'calculation_time': round(time.time() - start_time, 4),
                'success': True,
                'interpretation': 'Lower perplexity indicates better predictability'
            }
            
            logger.debug(f"Perplexity calculated: {result['raw_score']} (normalized: {result['score']})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {str(e)}")
            return {
                'score': 0.0,
                'raw_score': float('inf'),
                'normalized_score': 0.0,
                'token_count': 0,
                'calculation_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'success': False,
                'error': str(e)
            }
    
    def calculate_bleu_score(self, candidate: str, reference: str, weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> Dict[str, Any]:
        """
        Calculate BLEU score between candidate and reference text.
        BLEU measures n-gram precision between machine translation and reference.
        
        Args:
            candidate (str): Generated text (LLM output)
            reference (str): Reference text (expected answer)
            weights (Tuple[float, ...]): Weights for n-gram precisions (1-gram to 4-gram)
            
        Returns:
            Dict[str, Any]: BLEU calculation results
        """
        try:
            start_time = time.time()
            logger.debug(f"Calculating BLEU score for candidate length {len(candidate)} vs reference length {len(reference)}")
            
            # Tokenize texts
            candidate_tokens = self._tokenize_text(candidate)
            reference_tokens = self._tokenize_text(reference)
            
            if not candidate_tokens or not reference_tokens:
                return {
                    'score': 0.0,
                    'individual_scores': [0.0, 0.0, 0.0, 0.0],
                    'brevity_penalty': 0.0,
                    'candidate_length': len(candidate_tokens),
                    'reference_length': len(reference_tokens),
                    'calculation_time': time.time() - start_time,
                    'success': False,
                    'error': 'Empty candidate or reference text'
                }
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens, 
                weights=weights,
                smoothing_function=self.smoothing_function
            )
            
            # Calculate individual n-gram scores
            individual_scores = []
            for i in range(1, 5):  # 1-gram to 4-gram
                try:
                    weight = [0.0] * 4
                    weight[i-1] = 1.0
                    score = sentence_bleu(
                        [reference_tokens], 
                        candidate_tokens, 
                        weights=weight,
                        smoothing_function=self.smoothing_function
                    )
                    individual_scores.append(round(score, 4))
                except:
                    individual_scores.append(0.0)
            
            # Calculate brevity penalty
            candidate_length = len(candidate_tokens)
            reference_length = len(reference_tokens)
            
            if candidate_length > reference_length:
                brevity_penalty = 1.0
            else:
                brevity_penalty = math.exp(1 - reference_length / candidate_length) if candidate_length > 0 else 0.0
            
            result = {
                'score': round(bleu_score, 4),
                'individual_scores': {
                    '1-gram': individual_scores[0],
                    '2-gram': individual_scores[1],
                    '3-gram': individual_scores[2],
                    '4-gram': individual_scores[3]
                },
                'brevity_penalty': round(brevity_penalty, 4),
                'candidate_length': candidate_length,
                'reference_length': reference_length,
                'weights_used': weights,
                'calculation_time': round(time.time() - start_time, 4),
                'success': True,
                'interpretation': 'Higher BLEU score indicates better similarity to reference'
            }
            
            logger.debug(f"BLEU score calculated: {result['score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {str(e)}")
            return {
                'score': 0.0,
                'individual_scores': {'1-gram': 0.0, '2-gram': 0.0, '3-gram': 0.0, '4-gram': 0.0},
                'brevity_penalty': 0.0,
                'candidate_length': 0,
                'reference_length': 0,
                'calculation_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'success': False,
                'error': str(e)
            }
    
    def calculate_rouge_scores(self, candidate: str, reference: str) -> Dict[str, Any]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between candidate and reference.
        ROUGE measures recall-oriented n-gram overlap.
        
        Args:
            candidate (str): Generated text (LLM output)
            reference (str): Reference text (expected answer)
            
        Returns:
            Dict[str, Any]: ROUGE calculation results
        """
        try:
            start_time = time.time()
            logger.debug(f"Calculating ROUGE scores for candidate length {len(candidate)} vs reference length {len(reference)}")
            
            if not candidate.strip() or not reference.strip():
                return {
                    'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                    'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                    'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                    'average_fmeasure': 0.0,
                    'calculation_time': time.time() - start_time,
                    'success': False,
                    'error': 'Empty candidate or reference text'
                }
            
            # Calculate ROUGE scores
            scores = self.rouge_scorer.score(reference, candidate)
            
            # Extract and format results
            rouge1_scores = {
                'precision': round(scores['rouge1'].precision, 4),
                'recall': round(scores['rouge1'].recall, 4),
                'fmeasure': round(scores['rouge1'].fmeasure, 4)
            }
            
            rouge2_scores = {
                'precision': round(scores['rouge2'].precision, 4),
                'recall': round(scores['rouge2'].recall, 4),
                'fmeasure': round(scores['rouge2'].fmeasure, 4)
            }
            
            rougeL_scores = {
                'precision': round(scores['rougeL'].precision, 4),
                'recall': round(scores['rougeL'].recall, 4),
                'fmeasure': round(scores['rougeL'].fmeasure, 4)
            }
            
            # Calculate average F-measure
            avg_fmeasure = round(
                (rouge1_scores['fmeasure'] + rouge2_scores['fmeasure'] + rougeL_scores['fmeasure']) / 3, 4
            )
            
            result = {
                'rouge1': rouge1_scores,
                'rouge2': rouge2_scores,
                'rougeL': rougeL_scores,
                'average_fmeasure': avg_fmeasure,
                'best_fmeasure': max(rouge1_scores['fmeasure'], rouge2_scores['fmeasure'], rougeL_scores['fmeasure']),
                'calculation_time': round(time.time() - start_time, 4),
                'success': True,
                'interpretation': 'Higher ROUGE scores indicate better overlap with reference'
            }
            
            logger.debug(f"ROUGE scores calculated - ROUGE-1: {rouge1_scores['fmeasure']}, ROUGE-2: {rouge2_scores['fmeasure']}, ROUGE-L: {rougeL_scores['fmeasure']}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return {
                'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'average_fmeasure': 0.0,
                'calculation_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'success': False,
                'error': str(e)
            }
    
    def calculate_meteor_score(self, candidate: str, reference: str) -> Dict[str, Any]:
        """
        Calculate METEOR score between candidate and reference.
        METEOR considers precision, recall, synonyms, and word order.
        
        Args:
            candidate (str): Generated text (LLM output)
            reference (str): Reference text (expected answer)
            
        Returns:
            Dict[str, Any]: METEOR calculation results
        """
        try:
            start_time = time.time()
            logger.debug(f"Calculating METEOR score for candidate length {len(candidate)} vs reference length {len(reference)}")
            
            # Tokenize texts
            candidate_tokens = self._tokenize_text(candidate)
            reference_tokens = self._tokenize_text(reference)
            
            if not candidate_tokens or not reference_tokens:
                return {
                    'score': 0.0,
                    'candidate_length': len(candidate_tokens),
                    'reference_length': len(reference_tokens),
                    'calculation_time': time.time() - start_time,
                    'success': False,
                    'error': 'Empty candidate or reference text'
                }
            
            # Calculate METEOR score
            meteor_score_value = meteor_score([reference_tokens], candidate_tokens)
            
            result = {
                'score': round(meteor_score_value, 4),
                'candidate_length': len(candidate_tokens),
                'reference_length': len(reference_tokens),
                'calculation_time': round(time.time() - start_time, 4),
                'success': True,
                'interpretation': 'Higher METEOR score indicates better similarity considering synonyms and word order'
            }
            
            logger.debug(f"METEOR score calculated: {result['score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating METEOR score: {str(e)}")
            return {
                'score': 0.0,
                'candidate_length': 0,
                'reference_length': 0,
                'calculation_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK word tokenizer.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        try:
            if not text or not text.strip():
                return []
            
            # Use NLTK word tokenizer
            tokens = word_tokenize(text.lower())
            
            # Filter out empty tokens and punctuation-only tokens
            filtered_tokens = [token for token in tokens if token.isalnum()]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            # Fallback to simple splitting
            return text.lower().split()
    
    def evaluate_single_row(self, row_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate a single row of data with all mathematical metrics.
        
        Args:
            row_data (Dict[str, str]): Row data containing questions, answers, contexts, llm_answer
            
        Returns:
            Dict[str, Any]: Complete evaluation results for the row
        """
        try:
            start_time = time.time()
            
            # Extract data
            llm_answer = str(row_data.get('llm_answer', ''))
            expected_answer = str(row_data.get('answers', ''))
            question = str(row_data.get('questions', ''))
            contexts = str(row_data.get('contexts', ''))
            
            logger.debug(f"Evaluating row with LLM answer length: {len(llm_answer)}")
            
            # Calculate all metrics
            results = {
                'input_data': {
                    'question': question,
                    'expected_answer': expected_answer,
                    'llm_answer': llm_answer,
                    'contexts': contexts
                },
                'metrics': {}
            }
            
            # 1. Perplexity Score
            logger.debug("Calculating perplexity score...")
            results['metrics']['perplexity'] = self.calculate_perplexity_score(llm_answer)
            
            # 2. BLEU Score
            logger.debug("Calculating BLEU score...")
            results['metrics']['bleu'] = self.calculate_bleu_score(llm_answer, expected_answer)
            
            # 3. ROUGE Scores
            logger.debug("Calculating ROUGE scores...")
            results['metrics']['rouge'] = self.calculate_rouge_scores(llm_answer, expected_answer)
            
            # 4. METEOR Score
            logger.debug("Calculating METEOR score...")
            results['metrics']['meteor'] = self.calculate_meteor_score(llm_answer, expected_answer)
            
            # Calculate summary
            successful_metrics = sum(1 for metric in results['metrics'].values() if metric.get('success', False))
            total_metrics = len(results['metrics'])
            
            # Extract main scores for summary
            main_scores = {
                'perplexity': results['metrics']['perplexity'].get('score', 0.0),
                'bleu': results['metrics']['bleu'].get('score', 0.0),
                'rouge_avg': results['metrics']['rouge'].get('average_fmeasure', 0.0),
                'meteor': results['metrics']['meteor'].get('score', 0.0)
            }
            
            # Calculate average score (note: perplexity is inverted - higher is worse)
            # For averaging, we'll use the normalized perplexity score
            scorable_metrics = ['bleu', 'rouge_avg', 'meteor', 'perplexity']
            valid_scores = [main_scores[metric] for metric in scorable_metrics if main_scores[metric] > 0]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            results['summary'] = {
                'total_metrics': total_metrics,
                'successful_metrics': successful_metrics,
                'failed_metrics': total_metrics - successful_metrics,
                'success_rate': round(successful_metrics / total_metrics, 4),
                'main_scores': main_scores,
                'average_score': round(average_score, 4),
                'evaluation_time': round(time.time() - start_time, 4)
            }
            
            logger.debug(f"Row evaluation completed in {results['summary']['evaluation_time']} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating single row: {str(e)}")
            return {
                'input_data': row_data,
                'metrics': {},
                'summary': {
                    'total_metrics': 4,
                    'successful_metrics': 0,
                    'failed_metrics': 4,
                    'success_rate': 0.0,
                    'average_score': 0.0,
                    'evaluation_time': 0.0
                },
                'error': str(e),
                'success': False
            }
    
    def evaluate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate all rows in a DataFrame with mathematical metrics.
        
        Args:
            df (pd.DataFrame): DataFrame with required columns
            
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        try:
            logger.info(f"Starting mathematical evaluation of DataFrame with {len(df)} rows...")
            start_time = time.time()
            
            # Validate DataFrame
            required_columns = ['questions', 'answers', 'contexts', 'llm_answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Initialize results structure
            evaluation_results = {
                'metadata': {
                    'total_rows': len(df),
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'metrics_calculated': ['perplexity', 'bleu', 'rouge', 'meteor']
                },
                'results': [],
                'summary': {
                    'successful_rows': 0,
                    'failed_rows': 0,
                    'metric_averages': {},
                    'overall_scores': {},
                    'total_evaluation_time': 0.0
                }
            }
            
            # Process each row
            for index, row in df.iterrows():
                logger.debug(f"Processing row {index + 1}/{len(df)}")
                
                row_data = {
                    'questions': row['questions'],
                    'answers': row['answers'],
                    'contexts': row['contexts'],
                    'llm_answer': row['llm_answer']
                }
                
                # Evaluate row
                row_result = self.evaluate_single_row(row_data)
                row_result['row_index'] = index
                
                evaluation_results['results'].append(row_result)
                
                # Update counters
                if row_result.get('success', True):
                    evaluation_results['summary']['successful_rows'] += 1
                else:
                    evaluation_results['summary']['failed_rows'] += 1
            
            # Calculate summary statistics
            self._calculate_summary_statistics(evaluation_results)
            
            total_time = time.time() - start_time
            evaluation_results['summary']['total_evaluation_time'] = round(total_time, 4)
            
            logger.info(f"Mathematical evaluation completed in {total_time:.2f} seconds")
            logger.info(f"Successfully evaluated {evaluation_results['summary']['successful_rows']}/{len(df)} rows")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during DataFrame evaluation: {str(e)}")
            raise
    
    def _calculate_summary_statistics(self, evaluation_results: Dict[str, Any]) -> None:
        """Calculate summary statistics across all evaluations."""
        try:
            # Initialize metric collections
            metric_scores = {
                'perplexity': [],
                'bleu': [],
                'rouge1_fmeasure': [],
                'rouge2_fmeasure': [],
                'rougeL_fmeasure': [],
                'rouge_average': [],
                'meteor': []
            }
            
            # Collect scores from all successful evaluations
            for result in evaluation_results['results']:
                if result.get('success', True) and 'metrics' in result:
                    metrics = result['metrics']
                    
                    # Perplexity
                    if metrics.get('perplexity', {}).get('success', False):
                        metric_scores['perplexity'].append(metrics['perplexity']['score'])
                    
                    # BLEU
                    if metrics.get('bleu', {}).get('success', False):
                        metric_scores['bleu'].append(metrics['bleu']['score'])
                    
                    # ROUGE
                    if metrics.get('rouge', {}).get('success', False):
                        rouge = metrics['rouge']
                        metric_scores['rouge1_fmeasure'].append(rouge['rouge1']['fmeasure'])
                        metric_scores['rouge2_fmeasure'].append(rouge['rouge2']['fmeasure'])
                        metric_scores['rougeL_fmeasure'].append(rouge['rougeL']['fmeasure'])
                        metric_scores['rouge_average'].append(rouge['average_fmeasure'])
                    
                    # METEOR
                    if metrics.get('meteor', {}).get('success', False):
                        metric_scores['meteor'].append(metrics['meteor']['score'])
            
            # Calculate averages
            evaluation_results['summary']['metric_averages'] = {}
            for metric_name, scores in metric_scores.items():
                if scores:
                    evaluation_results['summary']['metric_averages'][metric_name] = {
                        'average': round(sum(scores) / len(scores), 4),
                        'min': round(min(scores), 4),
                        'max': round(max(scores), 4),
                        'count': len(scores),
                        'std_dev': round(np.std(scores), 4) if len(scores) > 1 else 0.0
                    }
                else:
                    evaluation_results['summary']['metric_averages'][metric_name] = {
                        'average': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0,
                        'std_dev': 0.0
                    }
            
            # Calculate overall performance scores
            main_metrics = ['bleu', 'rouge_average', 'meteor', 'perplexity']
            overall_scores = []
            
            for metric in main_metrics:
                avg_score = evaluation_results['summary']['metric_averages'].get(metric, {}).get('average', 0.0)
                if avg_score > 0:
                    overall_scores.append(avg_score)
            
            evaluation_results['summary']['overall_scores'] = {
                'average_across_metrics': round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0,
                'total_valid_metrics': len(overall_scores),
                'performance_grade': self._get_performance_grade(sum(overall_scores) / len(overall_scores) if overall_scores else 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
    
    def _get_performance_grade(self, average_score: float) -> str:
        """Convert average score to performance grade."""
        if average_score >= 0.8:
            return "Excellent"
        elif average_score >= 0.6:
            return "Good"
        elif average_score >= 0.4:
            return "Fair"
        elif average_score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def get_simplified_scores(self, evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract simplified scores for easy consumption.
        
        Args:
            evaluation_results (Dict[str, Any]): Complete evaluation results
            
        Returns:
            Dict[str, Dict[str, float]]: Simplified scores by row
        """
        simplified = {}
        
        for result in evaluation_results['results']:
            if 'row_index' in result and 'summary' in result:
                row_key = f"row_{result['row_index']}"
                simplified[row_key] = result['summary']['main_scores']
        
        return simplified

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
            "When the user selects an object, the universal viewer builds a list of available files and their associated viewers."
        ],
        'contexts': [
            "User interface configuration for Active Workspace. You can emphasize information specific to your company's processes.",
            "The following is a set of best practices for commonly asked questions gathered into one place for easy access."
        ],
        'llm_answer': [
            "Hello! Customizing Active Workspace in Teamcenter can significantly enhance your user experience by tailoring the interface to meet your specific needs. You can use CSS to customize the appearance and modify existing tools.",
            "Hello! Thank you for your question about Siemens PLM software. The universal viewer works by building a list of available files when users select objects, using configured viewer preferences."
        ]
    }
    
    # Create test DataFrame
    test_df = pd.DataFrame(test_data)
    
    try:
        # Initialize mathematical evaluation agent
        agent = MathematicalEvaluationAgent()
        
        # Run evaluation
        results = agent.evaluate_dataframe(test_df)
        
        # Print results
        print("✅ Mathematical evaluation completed successfully!")
        print(f"📊 Processed {results['metadata']['total_rows']} rows")
        print(f"⏱️ Total evaluation time: {results['summary']['total_evaluation_time']} seconds")
        print(f"🎯 Overall performance: {results['summary']['overall_scores']['performance_grade']}")
        
        print("\n📈 Metric Averages:")
        for metric_name, stats in results['summary']['metric_averages'].items():
            if stats['count'] > 0:
                print(f"  {metric_name}: {stats['average']} (min: {stats['min']}, max: {stats['max']})")
        
        # Get simplified scores
        simplified_scores = agent.get_simplified_scores(results)
        print(f"\n🎯 Simplified scores created for {len(simplified_scores)} rows")
        
        # Show first row scores as example
        if simplified_scores:
            first_row = list(simplified_scores.keys())[0]
            print(f"\nExample scores for {first_row}:")
            for metric, score in simplified_scores[first_row].items():
                print(f"  {metric}: {score}")
        
    except Exception as e:
        print(f"❌ Mathematical evaluation failed: {str(e)}")
        logger.error(f"Test execution failed: {str(e)}")