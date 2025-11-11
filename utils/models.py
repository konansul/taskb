from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ChartSuggestion(BaseModel):
    title: str = Field(description="The concise, descriptive title from 'Chart Suggestion Title'.")
    question: str = Field(description="The analytical question from 'The Question it Answers'.")
    chart_type: str = Field(description="The single chart type from 'Chart Type', preferably from Seaborn.")
    pre_processing_steps: str = Field(description="The data manipulation steps from 'Data Pre-processing/Aggregation (if any)'. Should be 'None' if no steps are required.")
    column_mapping: Dict[str, Optional[str]] = Field(description="A dictionary with keys like 'X-Axis', 'Y-Axis', 'Color/Hue (Optional)', 'Facet (Optional)' and values that are exact column names from the dataset or None.")
    description: str = Field(description="The rationale and insight from 'Rationale and Insight'.")

class Suggestions(BaseModel):
    charts: List[ChartSuggestion]

class ChartAnalysis(BaseModel):
    chart_number: int = Field(description="The sequential number of the chart in the PDF.")
    chart_title: str = Field(description="The title of the chart as displayed in the PDF.")
    chart_type: str = Field(description="The type of chart (e.g., histogram, scatter plot, box plot, etc.).")
    detailed_description: str = Field(description="A comprehensive description of what the chart shows, including specific data points, patterns, and trends visible in the chart.")
    key_insights: List[str] = Field(description="A list of 3-5 key insights or findings that can be derived from this chart.")
    data_trends: str = Field(description="Description of the main trends, patterns, or relationships shown in the data.")
    statistical_observations: str = Field(description="Any notable statistical observations like outliers, distributions, correlations, etc.")

class PDFAnalysisReport(BaseModel):
    total_charts: int = Field(description="Total number of charts analyzed in the PDF.")
    charts: List[ChartAnalysis] = Field(description="List of detailed analysis for each chart.")
    overall_trend_summary: str = Field(description="A single comprehensive sentence describing the overall trend across all charts in the dataset.")
