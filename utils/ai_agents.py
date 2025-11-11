from typing import List, Dict, Optional
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from .models import ChartSuggestion, Suggestions, PDFAnalysisReport
from .prompts import ANALYSIS_TEMPLATE, CODE_GEN_TEMPLATE, EXTRACTION_TEMPLATE, QUERY_PROCESSING_TEMPLATE, CSV_FORMATTING_TEMPLATE

class DataAnalyst:
    """Analyzes datasets and generates chart suggestions"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def analyze_dataset(self, df_context: Dict[str, str], num_suggestions: int = 5) -> str:
        """Generate textual chart suggestions from dataset context"""
        analysis_prompt = PromptTemplate.from_template(ANALYSIS_TEMPLATE)
        analysis_chain = analysis_prompt | self.llm | StrOutputParser()
        
        # Add num_suggestions to the context
        context_with_suggestions = {**df_context, "num_suggestions": num_suggestions}
        return analysis_chain.invoke(context_with_suggestions)

class SuggestionExtractor:
    """Extracts structured data from textual analysis"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def extract_suggestions(self, text_report: str, df_columns: List[str], num_suggestions: int = 5) -> List[ChartSuggestion]:
        """Extract structured chart suggestions from text analysis"""
        json_parser = JsonOutputParser(pydantic_object=Suggestions)
        extraction_prompt = PromptTemplate.from_template(EXTRACTION_TEMPLATE)
        extraction_chain = extraction_prompt | self.llm | json_parser
        
        result = extraction_chain.invoke({
            "text_report": text_report,
            "df_columns": str(df_columns),
            "num_suggestions": num_suggestions,
            "format_instructions": json_parser.get_format_instructions()
        })
        
        # Convert dictionaries to ChartSuggestion objects if needed
        suggestions = [ ]
        if isinstance(result, dict) and 'charts' in result:
            charts_data = result['charts']
        else:
            charts_data = result
            
        for chart_data in charts_data:
            if isinstance(chart_data, dict):
                # Convert dict to ChartSuggestion object
                suggestions.append(ChartSuggestion(**chart_data))
            elif isinstance(chart_data, ChartSuggestion):
                suggestions.append(chart_data)
            else:
                # Handle unexpected format
                print(f"Warning: Unexpected chart data format: {type(chart_data)}")
                
        return suggestions

class ChartQueryProcessor:
    """Processes natural language chart queries and converts them to ChartSuggestion format"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
    
    def process_query(self, user_query: str, df_context: Dict[str, str]) -> ChartSuggestion:
        """Convert natural language query to structured chart suggestion"""
        query_prompt = PromptTemplate.from_template(QUERY_PROCESSING_TEMPLATE)
        json_parser = JsonOutputParser(pydantic_object=ChartSuggestion)
        query_chain = query_prompt | self.llm | json_parser
        
        result = query_chain.invoke({
            "user_query": user_query,
            "df_columns": df_context["df_columns"],
            "df_info": df_context["df_info"],
            "df_description": df_context["df_description"],
            "format_instructions": json_parser.get_format_instructions()
        })
        
        # Convert to ChartSuggestion object if it's a dict
        if isinstance(result, dict):
            return ChartSuggestion(**result)
        return result

class ChartCodeGenerator:
    """Generates executable Python code for charts"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def generate_chart_code(self, suggestion: ChartSuggestion, df_context: Dict[str, str]) -> str:
        """Generate Python code for a specific chart suggestion"""
        code_gen_prompt = PromptTemplate.from_template(CODE_GEN_TEMPLATE)
        code_gen_chain = code_gen_prompt | self.llm | StrOutputParser()
        
        return code_gen_chain.invoke({
            "df_info": df_context["df_info"],
            "df_description": df_context["df_description"],
            "df_columns": df_context["df_columns"],
            "chart_type": suggestion.chart_type,
            "title": suggestion.title,
            "question": suggestion.question,
            "column_mapping": suggestion.column_mapping,
            "pre_processing_steps": suggestion.pre_processing_steps
        })

    def clean_generated_code(self, code_string: str) -> str:
        """Strips markdown and import statements from the LLM's generated code."""
        if "```python" in code_string:
            code_string = code_string.split("```python")[1].strip()
        if "```" in code_string:
            code_string = code_string.split("```")[0].strip()
        
        lines = [line for line in code_string.split('\n') 
                if not (line.strip().startswith('import ') or line.strip().startswith('from '))]
        return '\n'.join(lines).strip()

class CSVFormatter:
    """Formats and cleans unstructured CSV data from government sources and similar complex datasets"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def format_csv_data(self, raw_csv_data: str) -> Dict[str, str]:
        """
        Process raw CSV data and return formatted data
        
        Args:
            raw_csv_data: Raw CSV content as string
            
        Returns:
            Dictionary with table names as keys and formatted CSV data as values
            - Single table: {"main_table": "formatted_csv_data"}
            - Multiple tables: {"table_1_title": "csv_data_1", "table_2_title": "csv_data_2", ...}
        """
        formatting_prompt = PromptTemplate.from_template(CSV_FORMATTING_TEMPLATE)
        formatting_chain = formatting_prompt | self.llm | StrOutputParser()
        
        formatted_response = formatting_chain.invoke({
            "raw_csv_data": raw_csv_data
        })
        
        # Validate against hallucination before parsing
        if self._contains_hallucination(formatted_response, raw_csv_data):
            print("⚠️ Warning: AI response contains hallucinated content. Attempting re-processing...")
            # Try again with stricter prompt
            stricter_prompt = self._create_stricter_prompt()
            stricter_chain = stricter_prompt | self.llm | StrOutputParser()
            formatted_response = stricter_chain.invoke({
                "raw_csv_data": raw_csv_data
            })
            
            # If still hallucinating, return error
            if self._contains_hallucination(formatted_response, raw_csv_data):
                raise ValueError("AI is generating hallucinated content. Please try with a different file or contact support.")
        
        return self._parse_formatting_response(formatted_response)
    
    def _parse_formatting_response(self, response: str) -> Dict[str, str]:
        """Parse the AI response to extract single or multiple tables"""
        response = response.strip()
        
        # Check if response contains multiple tables (indicated by TABLE_X_TITLE: pattern)
        if "TABLE_" in response and "_TITLE:" in response:
            return self._parse_multiple_tables(response)
        else:
            # Single table - clean and return
            cleaned_csv = self._clean_csv_output(response)
            return {"main_table": cleaned_csv}
    
    def _parse_multiple_tables(self, response: str) -> Dict[str, str]:
        """Parse multiple tables from AI response"""
        tables = {}
        
        # Split by TABLE_X_TITLE: pattern
        import re
        
        # Find all table sections
        table_pattern = r'TABLE_\d+_TITLE:\s*([^\n]+)\n(.*?)(?=TABLE_\d+_TITLE:|$)'
        matches = re.findall(table_pattern, response, re.DOTALL)
        
        for i, (title, csv_data) in enumerate(matches, 1):
            title = title.strip()
            csv_data = csv_data.strip()
            
            # Clean the CSV data
            cleaned_csv = self._clean_csv_output(csv_data)
            
            # Use title as key, fallback to generic name if title is unclear
            table_key = title if title and len(title) > 0 else f"Table_{i}"
            tables[table_key] = cleaned_csv
        
        # If no matches found but response suggests multiple tables, try alternative parsing
        if not tables and ("TABLE" in response.upper() or len(response.split('\n\n')) > 2):
            # Fallback: split by double newlines and try to identify tables
            sections = [s.strip() for s in response.split('\n\n') if s.strip()]
            
            for i, section in enumerate(sections):
                if ',' in section and '\n' in section:  # Looks like CSV data
                    lines = section.split('\n')
                    # Try to find a title line before CSV data
                    title = f"Table_{i+1}"
                    if i > 0 and len(lines) > 2:
                        # Previous section might contain title
                        prev_section = sections[i-1] if i > 0 else ""
                        if prev_section and not ',' in prev_section:
                            title = prev_section[:50]  # Use first 50 chars as title
                    
                    cleaned_csv = self._clean_csv_output(section)
                    if cleaned_csv:  # Only add if we have valid CSV data
                        tables[title] = cleaned_csv
        
        return tables if tables else {"main_table": self._clean_csv_output(response)}
    
    def _clean_csv_output(self, csv_output: str) -> str:
        """Clean the LLM output to ensure it's valid CSV"""
        # Remove any markdown formatting
        if "```csv" in csv_output:
            csv_output = csv_output.split("```csv")[1].strip()
        if "```" in csv_output:
            csv_output = csv_output.split("```")[0].strip()
        
        # Remove any explanatory text that might be before or after CSV
        lines = csv_output.strip().split('\n')
        
        # Find the start of actual CSV data (first line with commas)
        start_idx = 0
        for i, line in enumerate(lines):
            if ',' in line and not line.strip().startswith('#'):
                start_idx = i
                break
        
        # Find the end of CSV data (last line with commas)
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            if ',' in lines[i] and not lines[i].strip().startswith('#'):
                end_idx = i
                break
        
        # Extract only the CSV portion
        csv_lines = lines[start_idx:end_idx + 1]
        
        return '\n'.join(csv_lines).strip()
    
    def _contains_hallucination(self, ai_response: str, original_data: str) -> bool:
        """Detect if AI response contains hallucinated content"""
        
        # Common hallucination patterns
        hallucination_patterns = [
            r'\.strip\(\)',                    # Python string methods
            r'\.replace\(',                    # Python string methods
            r're\.sub\(',                      # Regex patterns
            r'text\s*=\s*',                    # Variable assignments
            r'base_name\s*=',                  # Variable assignments
            r'#\s*Remove',                     # Processing comments
            r'#\s*Convert',                    # Processing comments
            r'r[\'"][^\'"]*[\'"]',            # Regex patterns
            r'import\s+\w+',                   # Import statements
            r'def\s+\w+\(',                    # Function definitions
            r'\.lower\(\)',                    # String methods
            r'\.upper\(\)',                    # String methods
            r'\.split\(',                      # String methods
        ]
        
        import re
        
        # Check for hallucination patterns
        for pattern in hallucination_patterns:
            if re.search(pattern, ai_response, re.IGNORECASE):
                print(f"🚨 Hallucination detected: Found pattern '{pattern}' in AI response")
                return True
        
        # Check for completely made-up data that doesn't exist in original
        # Extract potential data values from both original and response
        original_words = set(re.findall(r'\b\w+\b', original_data.lower()))
        
        # Look for suspicious content in response that suggests code/instructions
        response_lines = ai_response.split('\n')
        for line in response_lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Skip obvious CSV header/data lines
            if line_clean.startswith('TABLE_') or ',' in line_clean:
                continue
                
            # Check if line contains programming-like content
            if any(char in line_clean for char in ['(', ')', '=', '.strip', '.replace', 'import']):
                print(f"🚨 Hallucination detected: Suspicious line '{line_clean[:50]}...'")
                return True
        
        return False
    
    def _create_stricter_prompt(self) -> PromptTemplate:
        """Create a stricter prompt template to prevent hallucination"""
        
        stricter_template = """
You are a data processing expert. Your ONLY job is to reformat existing CSV data.

**ABSOLUTE RULES:**
1. ONLY extract and reformat data that EXISTS in the input
2. DO NOT generate any Python code, regex patterns, or processing instructions
3. DO NOT include comments, explanations, or processing notes in the output
4. DO NOT create fake data or examples

**INPUT DATA:**
{raw_csv_data}

**TASK:**
Extract the actual tabular data from above and format it as clean CSV with proper headers.
Return ONLY the CSV data, nothing else.

**OUTPUT FORMAT:**
If single table: Just return the clean CSV
If multiple tables: Use this format:
TABLE_1_TITLE: [actual title from data]
[clean csv data]

TABLE_2_TITLE: [actual title from data]  
[clean csv data]

Return the formatted data now:
"""
        
        return PromptTemplate.from_template(stricter_template)
