import os
import pathlib
import json
import traceback
import io
from typing import List, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from google import genai
from google.genai import types
from .models import PDFAnalysisReport, ChartAnalysis

def setup_unicode_fonts():
    """Finds and registers a Unicode-compatible font for PDF generation."""
    font_paths = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for font_path in font_paths:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            return 'DejaVuSans'
    print("Warning: DejaVuSans font not found. PDF might have issues with special characters.")
    return 'Helvetica'

class StreamlitPDFGenerator:
    """Streamlit-optimized PDF generation"""
    
    def __init__(self):
        self.unicode_font = setup_unicode_fonts()
        
    def generate_enhanced_pdf(
        self, 
        chart_images: List[io.BytesIO], 
        pdf_analysis: PDFAnalysisReport, 
        output_filename: str
    ) -> str:
        """Generates an enhanced PDF with detailed chart descriptions from the analysis."""
        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontName=self.unicode_font, alignment=1)
        chart_title_style = ParagraphStyle('ChartTitle', parent=styles['h2'], fontName=self.unicode_font, spaceAfter=6)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=self.unicode_font, spaceAfter=12, leading=14)
        insight_style = ParagraphStyle('InsightStyle', parent=styles['Normal'], fontName=self.unicode_font, spaceAfter=8, leading=12, leftIndent=20)
        
        story = [
            Paragraph("AI-Generated Enhanced Data Dashboard", title_style), 
            Spacer(1, 0.25 * inch),
            Paragraph(f"<b>Overall Analysis Summary:</b> {pdf_analysis.overall_trend_summary}", normal_style),
            Spacer(1, 0.3 * inch)
        ]
        
        # Add each chart with its enhanced description
        charts_to_process = min(len(chart_images), len(pdf_analysis.charts))
        
        for i in range(charts_to_process):
            img_buffer = chart_images[i]
            chart_analysis = pdf_analysis.charts[i]
            
            # Chart image
            img_buffer.seek(0)
            img = Image(img_buffer, width=6*inch, height=3.75*inch, kind='proportional')
            
            # Chart title
            story.append(Paragraph(f"Chart {chart_analysis.chart_number}: {chart_analysis.chart_title}", chart_title_style))
            story.append(img)
            story.append(Spacer(1, 0.1 * inch))
            
            # Chart type
            story.append(Paragraph(f"<b>Chart Type:</b> {chart_analysis.chart_type}", normal_style))
            
            # Detailed description
            story.append(Paragraph(f"<b>Detailed Analysis:</b> {chart_analysis.detailed_description}", normal_style))
            
            # Data trends
            story.append(Paragraph(f"<b>Data Trends:</b> {chart_analysis.data_trends}", normal_style))
            
            # Statistical observations
            if chart_analysis.statistical_observations:
                story.append(Paragraph(f"<b>Statistical Observations:</b> {chart_analysis.statistical_observations}", normal_style))
            
            # Key insights
            story.append(Paragraph("<b>Key Insights:</b>", normal_style))
            for insight in chart_analysis.key_insights:
                story.append(Paragraph(f"• {insight}", insight_style))
            
            story.append(Spacer(1, 0.3 * inch))
        
        # Handle remaining chart images if analysis has fewer charts
        for i in range(charts_to_process, len(chart_images)):
            img_buffer = chart_images[i]
            img_buffer.seek(0)
            img = Image(img_buffer, width=6*inch, height=3.75*inch, kind='proportional')
            
            story.append(Paragraph(f"Chart {i+1}: Additional Chart", chart_title_style))
            story.append(img)
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph("<b>Analysis:</b> Detailed analysis not available for this chart.", normal_style))
            story.append(Spacer(1, 0.3 * inch))
        
        # Add final summary
        story.append(Spacer(1, 0.25 * inch))
        story.append(Paragraph("<b>Final Summary:</b>", chart_title_style))
        story.append(Paragraph(pdf_analysis.overall_trend_summary, normal_style))
        
        doc.build(story)
        return output_filename

class PDFAnalyzer:
    """Analyzes PDF files using Google GenAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def analyze_pdf_with_genai(self, pdf_file_path: str) -> PDFAnalysisReport:
        """Analyzes a PDF file using Google GenAI and returns structured analysis."""
        try:
            # Initialize the Google GenAI client
            client = genai.Client(api_key=self.api_key)
            print("Google GenAI client initialized successfully.")
            
            # Define the analysis prompt
            analysis_prompt = """
            Please analyze each chart in this PDF file in detail. For each chart you find:

            1. Identify the chart number (sequential order in the PDF)
            2. Extract the chart title
            3. Determine the chart type (histogram, scatter plot, box plot, etc.)
            4. Provide a detailed description of what the chart shows, including specific data points, patterns, and trends
            5. List 3-5 key insights or findings from the chart
            6. Describe the main data trends or patterns visible
            7. Note any statistical observations like outliers, distributions, correlations

            At the end, provide a 3-sentence comprehensive summary describing the overall trend across all charts in the dataset.

            Please be thorough and specific in your analysis, mentioning actual values and patterns you can see in the charts.
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                        types.Part.from_bytes(
                            data=pathlib.Path(pdf_file_path).read_bytes(),
                            mime_type='application/pdf',
                        ),
                        analysis_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PDFAnalysisReport.model_json_schema(),
                }
            )
            
            print("Response received from Google GenAI.")
            analysis_data = json.loads(response.text)
            parsed_result = PDFAnalysisReport(**analysis_data)
            print(f"Successfully parsed JSON with {len(parsed_result.charts)} charts")
            return parsed_result
                
        except Exception as e:
            print(f"Error analyzing PDF with GenAI: {e}")
            traceback.print_exc()
            # Return a fallback analysis
            return PDFAnalysisReport(
                total_charts=0,
                charts=[],
                overall_trend_summary="Unable to analyze the PDF due to an error."
            )
