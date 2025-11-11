import pandas as pd
import chardet
import io
import tempfile
import os
import csv
from typing import Dict, Tuple, Optional, List, Any

class DataProcessor:
    """Handles file upload and DataFrame operations"""
    
    @staticmethod
    def read_uploaded_file(file_content: bytes, filename: str) -> pd.DataFrame:
        """Read uploaded file content into DataFrame with encoding detection"""
        try:
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(file_content))
            else:  # Assume CSV
                encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
                df = pd.read_csv(io.StringIO(file_content.decode(encoding, errors='ignore')))
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not parse the file. Error: {e}")
    
    @staticmethod
    def get_dataframe_context(df: pd.DataFrame) -> Dict[str, str]:
        """Generate comprehensive DataFrame context for AI analysis"""
        df_head_str = df.head(20).to_string()
        df_shape_str = str(df.shape)
        df_columns_str = str(df.columns.tolist())
        df_description_str = df.describe(include='all').to_string()
        
        with io.StringIO() as buf:
            df.info(buf=buf)
            df_info_str = buf.getvalue()

        return {
            "df_head": df_head_str,
            "df_shape": df_shape_str,
            "df_columns": df_columns_str,
            "df_description": df_description_str,
            "df_info": df_info_str,
        }
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
        """Validate that required columns exist in DataFrame"""
        required_cols = set()
        for key, value in column_mapping.items():
            if value and isinstance(value, str) and value.strip():
                clean_value = value.strip()
                if clean_value in df.columns:
                    required_cols.add(clean_value)
        
        missing_cols = required_cols - set(df.columns)
        return len(missing_cols) == 0, list(missing_cols)
    
    @staticmethod
    def extract_raw_csv_data(file_content: bytes, filename: str) -> str:
        """Extract raw CSV data as string for AI formatting"""
        try:
            if filename.endswith((".xlsx", ".xls")):
                # For Excel files, convert to CSV string
                df = pd.read_excel(io.BytesIO(file_content))
                return df.to_csv(index=False)
            else:
                # For CSV files, return raw content as string
                encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
                return file_content.decode(encoding, errors='ignore')
        except Exception as e:
            raise ValueError(f"Could not extract raw data from file. Error: {e}")
    
    @staticmethod
    def save_formatted_csv(formatted_csv_data: str, original_filename: str) -> Tuple[str, pd.DataFrame]:
        """
        Save formatted CSV data to a temporary file and return path and DataFrame
        
        Args:
            formatted_csv_data: The formatted CSV data as string
            original_filename: Original filename for reference
            
        Returns:
            Tuple of (temp_file_path, DataFrame)
        """
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            base_name = os.path.splitext(original_filename)[0]
            temp_filename = f"{base_name}_formatted.csv"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            # Save formatted CSV
            with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                f.write(formatted_csv_data)
            
            # Load as DataFrame to validate
            df = pd.read_csv(temp_path)
            
            return temp_path, df
            
        except Exception as e:
            raise ValueError(f"Could not save formatted CSV data. Error: {e}")
    
    @staticmethod
    def parse_formatted_csv_string(formatted_csv_data: str) -> pd.DataFrame:
        """Parse formatted CSV string directly into DataFrame"""
        try:
            return pd.read_csv(io.StringIO(formatted_csv_data))
        except Exception as e:
            raise ValueError(f"Could not parse formatted CSV data. Error: {e}")
    
    @staticmethod
    def parse_multiple_formatted_csv_strings(tables_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Parse multiple formatted CSV strings into DataFrames with robust error handling"""
        dataframes = {}
        
        for table_name, csv_data in tables_dict.items():
            try:
                # First attempt: standard parsing
                df = pd.read_csv(io.StringIO(csv_data))
                if len(df) > 0 and len(df.columns) > 0:  # Valid DataFrame
                    dataframes[table_name] = df
                    continue
                    
            except Exception as e:
                print(f"Warning: Could not parse table '{table_name}' with standard method: {e}")
                
                # Second attempt: More robust parsing with error handling
                try:
                    # Try with error_bad_lines=False to skip problematic rows
                    df = pd.read_csv(
                        io.StringIO(csv_data), 
                        on_bad_lines='skip',  # Skip bad lines instead of failing
                        engine='python',      # Use Python engine for better error handling
                        quoting=1,           # Handle quotes properly
                        skipinitialspace=True # Skip whitespace after delimiter
                    )
                    
                    if len(df) > 0 and len(df.columns) > 0:
                        print(f"Successfully parsed table '{table_name}' with robust method (some rows may have been skipped)")
                        dataframes[table_name] = df
                        continue
                        
                except Exception as e2:
                    print(f"Warning: Could not parse table '{table_name}' even with robust method: {e2}")
                    
                    # Third attempt: Clean the CSV data manually
                    try:
                        cleaned_csv = DataProcessor._clean_csv_data(csv_data)
                        df = pd.read_csv(io.StringIO(cleaned_csv))
                        
                        if len(df) > 0 and len(df.columns) > 0:
                            print(f"Successfully parsed table '{table_name}' after manual cleaning")
                            dataframes[table_name] = df
                        else:
                            print(f"Warning: Table '{table_name}' resulted in empty DataFrame after cleaning")
                            
                    except Exception as e3:
                        print(f"Error: Failed to parse table '{table_name}' after all attempts: {e3}")
                        continue
        
        return dataframes
    
    @staticmethod
    def _clean_csv_data(csv_data: str) -> str:
        """Clean CSV data to handle field count inconsistencies"""
        lines = csv_data.strip().split('\n')
        if not lines:
            return csv_data
            
        # Get the header line (first non-empty line)
        header_line = None
        header_fields_count = 0
        
        for line in lines:
            if line.strip():
                header_line = line
                # Count fields in header (accounting for quoted fields)
                import csv
                reader = csv.reader([line])
                header_fields_count = len(next(reader))
                break
        
        if not header_line:
            return csv_data
            
        # Clean each line to match header field count
        cleaned_lines = []
        
        for line in lines:
            if not line.strip():
                cleaned_lines.append(line)
                continue
                
            try:
                # Parse the line to count actual fields
                import csv
                reader = csv.reader([line])
                fields = next(reader)
                
                # If field count matches, keep as is
                if len(fields) == header_fields_count:
                    cleaned_lines.append(line)
                elif len(fields) > header_fields_count:
                    # Too many fields - truncate to header count
                    truncated_fields = fields[:header_fields_count]
                    # Re-encode as CSV line
                    import io
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(truncated_fields)
                    cleaned_lines.append(output.getvalue().strip())
                else:
                    # Too few fields - pad with empty strings
                    padded_fields = fields + [''] * (header_fields_count - len(fields))
                    # Re-encode as CSV line
                    import io
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(padded_fields)
                    cleaned_lines.append(output.getvalue().strip())
                    
            except Exception:
                # If line parsing fails, try to salvage by basic field splitting
                fields = line.split(',')
                if len(fields) > header_fields_count:
                    fields = fields[:header_fields_count]
                elif len(fields) < header_fields_count:
                    fields.extend([''] * (header_fields_count - len(fields)))
                cleaned_lines.append(','.join(fields))
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def get_table_summary(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Get summary information for a table"""
        return {
            "name": table_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "preview": df.head(3).to_dict('records'),
            "missing_values": df.isnull().sum().sum(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        }
    
    @staticmethod
    def detect_if_needs_formatting(df: pd.DataFrame) -> bool:
        """
        Detect if a DataFrame likely needs CSV formatting
        Returns True if formatting is recommended
        """
        # Check for common indicators of unformatted data
        indicators = []
        
        # 1. Check for multi-level headers (columns with unnamed patterns)
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if len(unnamed_cols) > 0:
            indicators.append("unnamed_columns")
        
        # 2. Check for sparse data in first few rows (likely header rows)
        if len(df) > 3:
            first_rows_empty_ratio = df.head(3).isnull().sum().sum() / (3 * len(df.columns))
            if first_rows_empty_ratio > 0.3:  # More than 30% empty in first 3 rows
                indicators.append("sparse_header_rows")
        
        # 3. Check for numeric data in column names (year indicators)
        numeric_pattern_in_cols = any(any(char.isdigit() for char in str(col)) for col in df.columns)
        if numeric_pattern_in_cols:
            indicators.append("numeric_in_headers")
        
        # 4. Check for very long column names (might be concatenated)
        long_columns = [col for col in df.columns if len(str(col)) > 50]
        if len(long_columns) > 0:
            indicators.append("long_column_names")
        
        # 5. Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            indicators.append("duplicate_columns")
        
        # Return True if 2 or more indicators are present
        return len(indicators) >= 2
