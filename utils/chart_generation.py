import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import pandas as pd
from typing import List, Dict, Optional, Tuple
from .models import ChartSuggestion

class ChartGenerator:
    """Manages chart creation and execution"""
    
    def __init__(self):
        self.successful_charts = 0
        self.failed_charts = 0
        self.skipped_charts = 0
        
    def generate_chart_from_suggestion(
        self, 
        suggestion: ChartSuggestion, 
        df: pd.DataFrame, 
        generated_code: str
    ) -> Optional[io.BytesIO]:
        """Execute generated code and create chart"""
        try:
            # Set up execution environment
            img_buffer = io.BytesIO()
            exec_scope = {
                'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 
                'np': np, 'io': io, 'img_buffer': img_buffer
            }
            
            # Execute the generated code
            exec(generated_code, exec_scope)
            
            # Validate chart was created
            img_buffer.seek(0)
            if img_buffer.getbuffer().nbytes > 1000:
                self.successful_charts += 1
                return img_buffer
            else:
                self.failed_charts += 1
                return None
                
        except Exception as e:
            self.failed_charts += 1
            print(f"Chart generation failed: {e}")
            return None
    
    def _validate_suggestion_columns(self, suggestion: ChartSuggestion, df: pd.DataFrame) -> bool:
        """Validate that suggestion columns exist in DataFrame"""
        for key, value in suggestion.column_mapping.items():
            if value and isinstance(value, str) and value.strip():
                clean_value = value.strip()
                if clean_value not in df.columns:
                    print(f"Column '{clean_value}' not found in DataFrame")
                    return False
        return True
    
    def generate_dashboard_charts(
        self, 
        suggestions: List[ChartSuggestion], 
        df: pd.DataFrame,
        code_generator,  # ChartCodeGenerator
        df_context: Dict[str, str]
    ) -> List[Tuple[io.BytesIO, str]]:
        """Generate all charts for dashboard"""
        chart_results = []
        
        for suggestion in suggestions:
            # Validate columns
            if not self._validate_suggestion_columns(suggestion, df):
                self.skipped_charts += 1
                continue
            
            # Generate code
            try:
                generated_code = code_generator.generate_chart_code(suggestion, df_context)
                cleaned_code = code_generator.clean_generated_code(generated_code)
                
                # Generate chart
                chart_buffer = self.generate_chart_from_suggestion(suggestion, df, cleaned_code)
                
                if chart_buffer:
                    chart_results.append((chart_buffer, suggestion.title))
                    
            except Exception as e:
                print(f"Failed to generate chart '{suggestion.title}': {e}")
                self.failed_charts += 1
                
        return chart_results
    
    def get_generation_summary(self) -> Dict[str, int]:
        """Return summary of chart generation results"""
        total = self.successful_charts + self.failed_charts + self.skipped_charts
        return {
            "successful": self.successful_charts,
            "failed": self.failed_charts,
            "skipped": self.skipped_charts,
            "total": total
        }
