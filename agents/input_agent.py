import pandas as pd
import logging
import io
import chardet
from typing import Union, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputAgent:
    """
    Agent responsible for processing and validating uploaded CSV/XLSX files
    for LLM evaluation pipeline.
    """
    
    REQUIRED_COLUMNS = ['questions', 'answers', 'contexts', 'llm_answer']
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
    
    def __init__(self):
        """Initialize the InputAgent with required configurations."""
        logger.info("InputAgent initialized")
    
    def validate_file_format(self, filename: str) -> bool:
        """
        Validate if the uploaded file format is supported.
        
        Args:
            filename (str): Name of the uploaded file
            
        Returns:
            bool: True if format is supported, False otherwise
        """
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format: {file_extension}. Supported formats: {self.SUPPORTED_FORMATS}")
            return False
            
        logger.info(f"File format validation passed: {file_extension}")
        return True
    
    def detect_encoding(self, file_content: bytes) -> str:
        """
        Detect the encoding of the file content.
        
        Args:
            file_content (bytes): Raw file content
            
        Returns:
            str: Detected encoding
        """
        try:
            # Use chardet to detect encoding
            result = chardet.detect(file_content)
            detected_encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.info(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
            
            # If confidence is low, try common encodings
            if confidence < 0.7:
                common_encodings = ['utf-8', 'windows-1252', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in common_encodings:
                    try:
                        file_content.decode(encoding)
                        logger.info(f"Successfully validated encoding: {encoding}")
                        return encoding
                    except UnicodeDecodeError:
                        continue
            
            return detected_encoding or 'utf-8'
            
        except Exception as e:
            logger.warning(f"Error detecting encoding: {str(e)}, defaulting to utf-8")
            return 'utf-8'
    
    def read_file(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Read the uploaded file content and convert to pandas DataFrame.
        
        Args:
            file_content (bytes): Raw file content
            filename (str): Name of the uploaded file
            
        Returns:
            pd.DataFrame: Parsed DataFrame from file
            
        Raises:
            ValueError: If file cannot be read or parsed
        """
        try:
            file_extension = Path(filename).suffix.lower()
            
            if file_extension == '.csv':
                logger.info("Reading CSV file")
                
                # Detect encoding for CSV files
                encoding = self.detect_encoding(file_content)
                
                # Try multiple encodings if the detected one fails
                encodings_to_try = [encoding, 'utf-8', 'windows-1252', 'latin-1', 'iso-8859-1', 'cp1252']
                
                df = None
                successful_encoding = None
                
                for enc in encodings_to_try:
                    try:
                        file_buffer = io.BytesIO(file_content)
                        df = pd.read_csv(file_buffer, encoding=enc)
                        successful_encoding = enc
                        logger.info(f"Successfully read CSV with encoding: {enc}")
                        break
                    except UnicodeDecodeError as e:
                        logger.warning(f"Failed to read with encoding {enc}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to read CSV with encoding {enc}: {str(e)}")
                        continue
                
                if df is None:
                    # Last resort: try with error handling
                    try:
                        file_buffer = io.BytesIO(file_content)
                        df = pd.read_csv(file_buffer, encoding='utf-8', errors='replace')
                        logger.warning("Read CSV with UTF-8 and error replacement")
                        successful_encoding = 'utf-8 (with errors replaced)'
                    except Exception as e:
                        raise ValueError(f"Could not read CSV file with any encoding. Last error: {str(e)}")
                
            elif file_extension in ['.xlsx', '.xls']:
                logger.info(f"Reading Excel file: {file_extension}")
                file_buffer = io.BytesIO(file_content)
                
                try:
                    if file_extension == '.xlsx':
                        df = pd.read_excel(file_buffer, engine='openpyxl')
                    else:  # .xls
                        df = pd.read_excel(file_buffer, engine='xlrd')
                    successful_encoding = 'Excel (no encoding needed)'
                except Exception as e:
                    # Try with different engines
                    file_buffer = io.BytesIO(file_content)
                    try:
                        df = pd.read_excel(file_buffer)  # Let pandas choose the engine
                        successful_encoding = 'Excel (auto-detected engine)'
                    except Exception as e2:
                        raise ValueError(f"Could not read Excel file. Error: {str(e2)}")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Clean up the DataFrame
            df = self.clean_dataframe(df)
            
            logger.info(f"Successfully read file with {len(df)} rows and {len(df.columns)} columns using {successful_encoding}")
            return df
            
        except pd.errors.EmptyDataError:
            logger.error("File is empty or contains no data")
            raise ValueError("The uploaded file is empty or contains no valid data")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing file: {str(e)}")
            raise ValueError(f"Error parsing file: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error reading file: {str(e)}")
            raise ValueError(f"Error reading file: {str(e)}")
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by handling encoding issues and invalid characters.
        
        Args:
            df (pd.DataFrame): Original DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            # Convert all columns to string first to handle mixed types
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Replace common problematic characters
                    df[col] = df[col].astype(str).replace({
                        ''': "'",  # Smart quote
                        ''': "'",  # Smart quote
                        '"': '"',  # Smart quote
                        '"': '"',  # Smart quote
                        '–': '-',  # En dash
                        '—': '-',  # Em dash
                        '…': '...',  # Ellipsis
                        '\x92': "'",  # Common Windows-1252 issue
                        '\x93': '"',  # Common Windows-1252 issue
                        '\x94': '"',  # Common Windows-1252 issue
                        '\x96': '-',  # Common Windows-1252 issue
                        '\x97': '-',  # Common Windows-1252 issue
                    }, regex=False)
                    
                    # Remove any remaining non-printable characters
                    df[col] = df[col].str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            logger.info(f"DataFrame cleaned. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Error cleaning DataFrame: {str(e)}, returning original")
            return df
    
    def validate_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the DataFrame contains all required columns (case-insensitive).
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Dict[str, Any]: Validation result with status and details
        """
        try:
            # Convert all column names to lowercase for case-insensitive comparison
            df_columns_lower = [col.lower().strip() for col in df.columns]
            required_columns_lower = [col.lower() for col in self.REQUIRED_COLUMNS]
            
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"Required columns: {self.REQUIRED_COLUMNS}")
            
            # Check for missing columns
            missing_columns = []
            column_mapping = {}
            
            for required_col in required_columns_lower:
                found = False
                for i, df_col in enumerate(df_columns_lower):
                    if df_col == required_col:
                        column_mapping[required_col] = df.columns[i]  # Store original column name
                        found = True
                        break
                
                if not found:
                    missing_columns.append(required_col)
            
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                logger.error(error_msg)
                return {
                    'valid': False,
                    'error': error_msg,
                    'missing_columns': missing_columns,
                    'found_columns': list(df.columns)
                }
            
            logger.info("Column validation passed")
            return {
                'valid': True,
                'column_mapping': column_mapping,
                'total_rows': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error during column validation: {str(e)}")
            return {
                'valid': False,
                'error': f"Column validation error: {str(e)}"
            }
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data integrity of the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Dict[str, Any]: Validation result with status and details
        """
        try:
            issues = []
            
            # Check for empty DataFrame
            if df.empty:
                logger.error("DataFrame is empty")
                return {
                    'valid': False,
                    'error': "The file contains no data rows"
                }
            
            # Check for missing values in critical columns
            for col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"Column '{col}' has {null_count} missing values")
                    logger.warning(f"Column '{col}' has {null_count} missing values")
            
            # Check for empty strings
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    empty_strings = (df[col].astype(str).str.strip() == '').sum()
                    if empty_strings > 0:
                        issues.append(f"Column '{col}' has {empty_strings} empty strings")
                        logger.warning(f"Column '{col}' has {empty_strings} empty strings")
            
            # Check minimum content length for text columns
            text_columns = ['questions', 'answers', 'contexts', 'llm_answer']
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [tc.lower() for tc in text_columns]:
                    if df[col].dtype == 'object':
                        short_content = df[df[col].astype(str).str.len() < 10]
                        if not short_content.empty:
                            issues.append(f"Column '{col}' has {len(short_content)} entries with less than 10 characters")
                            logger.warning(f"Column '{col}' has {len(short_content)} entries with less than 10 characters")
            
            validation_result = {
                'valid': True,
                'total_rows': len(df),
                'issues': issues
            }
            
            if issues:
                logger.info(f"Data validation completed with {len(issues)} issues identified")
            else:
                logger.info("Data validation passed without issues")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during data integrity validation: {str(e)}")
            return {
                'valid': False,
                'error': f"Data integrity validation error: {str(e)}"
            }
    
    def normalize_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Normalize the DataFrame to have standard column names.
        
        Args:
            df (pd.DataFrame): Original DataFrame
            column_mapping (Dict[str, str]): Mapping from required columns to actual columns
            
        Returns:
            pd.DataFrame: Normalized DataFrame with standard column names
        """
        try:
            # Create a new DataFrame with normalized column names
            normalized_df = pd.DataFrame()
            
            for required_col_lower, actual_col in column_mapping.items():
                # Map back to the original required column name
                required_col = next(col for col in self.REQUIRED_COLUMNS if col.lower() == required_col_lower)
                normalized_df[required_col] = df[actual_col]
            
            logger.info(f"DataFrame normalized with columns: {list(normalized_df.columns)}")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Error normalizing DataFrame: {str(e)}")
            raise ValueError(f"Error normalizing DataFrame: {str(e)}")
    
    def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Main method to process the uploaded file and return validation results.
        
        Args:
            file_content (bytes): Raw file content
            filename (str): Name of the uploaded file
            
        Returns:
            Dict[str, Any]: Processing result with DataFrame and validation status
        """
        try:
            logger.info(f"Starting file processing for: {filename}")
            
            # Step 1: Validate file format
            if not self.validate_file_format(filename):
                return {
                    'success': False,
                    'error': f"Unsupported file format. Supported formats: {self.SUPPORTED_FORMATS}"
                }
            
            # Step 2: Read the file
            df = self.read_file(file_content, filename)
            
            # Step 3: Validate columns
            column_validation = self.validate_columns(df)
            if not column_validation['valid']:
                return {
                    'success': False,
                    'error': column_validation['error'],
                    'details': column_validation
                }
            
            # Step 4: Normalize DataFrame
            normalized_df = self.normalize_dataframe(df, column_validation['column_mapping'])
            
            # Step 5: Validate data integrity
            data_validation = self.validate_data_integrity(normalized_df)
            if not data_validation['valid']:
                return {
                    'success': False,
                    'error': data_validation['error'],
                    'details': data_validation
                }
            
            logger.info(f"File processing completed successfully for: {filename}")
            
            return {
                'success': True,
                'dataframe': normalized_df,
                'validation_details': {
                    'total_rows': data_validation['total_rows'],
                    'issues': data_validation.get('issues', []),
                    'original_columns': list(df.columns),
                    'normalized_columns': list(normalized_df.columns),
                    'encoding_info': 'File processed successfully with encoding detection'
                },
                'message': f"File processed successfully with {len(normalized_df)} rows"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error processing file {filename}: {str(e)}")
            return {
                'success': False,
                'error': f"Unexpected error processing file: {str(e)}"
            }

# Example usage and testing
if __name__ == "__main__":
    # Test with the provided CSV content
    test_csv_content = '''questions,answers,contexts,llm_answer
How can I customize Active Workspace?,You can configure nearly every aspect of the commands for the Active Workspace interface.,"User interface configuration for Active Workspace","Hello! Customizing Active Workspace in Teamcenter..."
How does it work?,"When the user selects an object, the universal viewer builds a list...","How do I? The following is a set of best practices...","Hello! Thank you for your question about Siemens PLM software..."'''
    
    agent = InputAgent()
    
    # Convert string to bytes for testing
    test_content_bytes = test_csv_content.encode('utf-8')
    
    # Process the test file
    result = agent.process_file(test_content_bytes, "test_file.csv")
    
    if result['success']:
        print("✅ File processing successful!")
        print(f"Processed {len(result['dataframe'])} rows")
        print(f"Columns: {list(result['dataframe'].columns)}")
        
        if result['validation_details']['issues']:
            print("\n⚠️ Issues found:")
            for issue in result['validation_details']['issues']:
                print(f"  - {issue}")
    else:
        print("❌ File processing failed!")
        print(f"Error: {result['error']}")