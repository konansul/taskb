import os
import io
import time
import chardet
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import pathlib
import json
# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# Google GenAI for PDF Analysis
from google import genai
from google.genai import types

# Matplotlib/Seaborn for Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# PDF Generation Imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- ENVIRONMENT SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
model_name = "gemini-2.5-flash" 

app = FastAPI(
    title="Robust AI Data Visualization Agent v3",
    description="Upload a CSV or Excel file to automatically generate a multi-page PDF dashboard with AI-powered insights and visualizations."
)


# Models for Structured Output 

class ChartSuggestion(BaseModel):
    title: str = Field(description="The concise, descriptive title from 'Chart Suggestion Title'.")
    question: str = Field(description="The analytical question from 'The Question it Answers'.")
    chart_type: str = Field(description="The single chart type from 'Chart Type', preferably from Seaborn.")
    pre_processing_steps: str = Field(description="The data manipulation steps from 'Data Pre-processing/Aggregation (if any)'. Should be 'None' if no steps are required.")
    column_mapping: Dict[str, Optional[str]] = Field(description="A dictionary with keys like 'X-Axis', 'Y-Axis', 'Color/Hue (Optional)', 'Facet (Optional)' and values that are exact column names from the dataset or None.")
    description: str = Field(description="The rationale and insight from 'Rationale and Insight'.")

class Suggestions(BaseModel):
    charts: List[ChartSuggestion]


# Models for PDF Analysis Output 

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


# Helper Functions 

def setup_unicode_fonts():
    """Finds and registers a Unicode-compatible font for PDF generation."""
    font_paths = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for font_path in font_paths:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            return 'DejaVuSans'
    print("Warning: DejaVuSans font not found. PDF might have issues with special characters.")
    return 'Helvetica'

def read_file_with_encoding_detection(file_content: bytes, filename: str) -> pd.DataFrame:
    """Reads CSV or Excel file content into a pandas DataFrame."""
    try:
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_content))
        else: # Assume CSV
            encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
            return pd.read_csv(io.StringIO(file_content.decode(encoding, errors='ignore')))
    except Exception as e:
        raise ValueError(f"Could not parse the file. Error: {e}")

def get_dataframe_context(df: pd.DataFrame) -> dict:
    """Generates a dictionary of context strings from a DataFrame."""
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

def clean_generated_code(code_string: str) -> str:
    """Strips markdown and import statements from the LLM's generated code."""
    if "```python" in code_string:
        code_string = code_string.split("```python")[1].strip()
    if "```" in code_string:
        code_string = code_string.split("```")[0].strip()
    
    lines = [line for line in code_string.split('\n') if not (line.strip().startswith('import ') or line.strip().startswith('from '))]
    return '\n'.join(lines).strip()


def analyze_pdf_with_genai(pdf_file_path: str, google_api_key: str) -> PDFAnalysisReport:
    """Analyzes a PDF file using Google GenAI and returns structured analysis."""
    try:
        
        
        # Initialize the Google GenAI client
        client = genai.Client(api_key=google_api_key)
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
        import traceback
        traceback.print_exc()
        # Return a fallback analysis
        return PDFAnalysisReport(
            total_charts=0,
            charts=[],
            overall_trend_summary="Unable to analyze the PDF due to an error."
        )


def generate_enhanced_pdf(chart_images: List[io.BytesIO], pdf_analysis: PDFAnalysisReport, output_filename: str) -> str:
    """Generates an enhanced PDF with detailed chart descriptions from the analysis."""
    unicode_font = setup_unicode_fonts()
    
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontName=unicode_font, alignment=1)
    chart_title_style = ParagraphStyle('ChartTitle', parent=styles['h2'], fontName=unicode_font, spaceAfter=6)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=unicode_font, spaceAfter=12, leading=14)
    insight_style = ParagraphStyle('InsightStyle', parent=styles['Normal'], fontName=unicode_font, spaceAfter=8, leading=12, leftIndent=20)
    
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


# --- PROMPT TEMPLATES ---

analysis_template = """
You are an expert Data Scientist and a master of data storytelling. 
Your primary skill is to look at any dataset and instantly identify the most compelling stories that can be told through visualizations. 
You think critically about the data, considering potential relationships, distributions, comparisons, compositions, and trends over time. 
Your suggestions must be modern, clear, and insightful, leveraging the capabilities of libraries like Seaborn.

Your mission is to analyze the provided dataset context and propose diverse 5 visualizations that tell a coherent story about the data. Your suggestions must be a complete blueprint that an automated tool can use to generate the charts directly.

**CRITICAL: You MUST only use the exact column names that exist in the dataset. The available columns are: {df_columns}**

**Full Dataset Context:**
- **Shape (Rows, Columns):** {df_shape}
- **Column Names:** {df_columns}
- **Schema (dtypes and non-null counts):**
{df_info}
- **Statistical Summary:**
{df_description}
- **Data Sample (first 20 rows):**
{df_head}

**Your Analysis Process & Structure:**
Your response must follow a logical narrative. Start with broad overviews and then drill down into more specific, complex relationships. Structure your suggestions into these categories:
1.  **Foundational Distributions & Overviews.**
2.  **Core Relationships & Comparisons.**
3.  **Multivariate & Deep-Dive Insights.**

**CRITICAL INSTRUCTIONS for Output:**
For each of the chart suggestions, provide the following details in a clear, structured format. Follow this template precisely for each suggestion:

---
**1. Chart Suggestion Title:** (A concise, descriptive title, e.g., "Age Distribution of Medal-Winning vs. Non-Winning Athletes")
*   **The Question it Answers:** (A clear, one-sentence analytical question, e.g., "Is there a significant difference in the age distribution between athletes who won a medal and those who did not?")
*   **Chart Type:** (The single most appropriate chart type from Seaborn, e.g., "Box Plot" or "Violin Plot".)
*   **Data Pre-processing/Aggregation (if any):** (A brief, clear description of any data manipulation required *before* plotting. If none, state "None".)
*   **Column Mapping (Seaborn style):**
    *   **X-Axis:** [MUST be an exact column name from {df_columns}]
    *   **Y-Axis:** [MUST be an exact column name from {df_columns} or None if not applicable]
    *   **Color/Hue (Optional):** [MUST be an exact column name from {df_columns} or None]
    *   **Facet (Optional):** [MUST be an exact column name from {df_columns} or None]
*   **Rationale and Insight:** (A brief explanation of why this chart is valuable and what specific patterns it might reveal.)
---
**(Repeat the above structure for all suggestions)**

**REMEMBER: Every column name you reference MUST exist exactly in this list: {df_columns}. Do not create new column names or use descriptive text as column names.**
"""

code_gen_template = """You are an elite Python data scientist and a Seaborn specialist. Your sole focus is on writing clean, direct, and executable code for data visualization using the Seaborn library. Your code must run without any modifications and produce a publication-quality chart.

**CONTEXT**:
- A pandas DataFrame named `df` is already loaded and available.
- The following libraries and variables are pre-imported and available: `pd`, `plt`, `sns`, `io`, `img_buffer`.
- **DataFrame Schema Context**:
    - Columns Available: {df_columns}
    - Data Types & Nulls: {df_info}
    - Statistical Summary: {df_description}

**TASK**:
- **Goal**: Write Python code using Seaborn to generate a '{chart_type}'.
- **Title**: The chart should be titled '{title}'.
- **Insight**: This chart is intended to answer: "{question}"
- **Data Pre-processing Steps**: {pre_processing_steps}
- **Column Mapping**: {column_mapping}

**SEABORN-CENTRIC CODE GENERATION LOGIC**:
- Use `sns.histplot`, `sns.countplot`, `sns.barplot`, `sns.scatterplot`, `sns.boxplot`, `sns.violinplot`, `sns.lineplot`, `sns.heatmap` as appropriate.
- Always use the `data` parameter in Seaborn plots (e.g., `sns.histplot(data=df_copy, ...)`).
- For heatmaps, the pre-processing step must create a suitable correlation matrix or pivot table.

**MANDATORY CODE STRUCTURE**:
1.  `sns.set_theme(style="whitegrid", palette="muted")`
2.  `df_copy = df.copy()`
3.  # **Pre-processing Block**: Implement the exact data pre-processing steps described. If 'None', this block can be empty.
4.  `fig, ax = plt.subplots(figsize=(12, 8))`
5.  # **Plotting Block**: The single Seaborn plotting command.
6.  `ax.set_title('{title}', fontsize=16, weight='bold')`
7.  Set appropriate xlabel and ylabel.
8.  If the x-axis has many categorical labels, use `plt.xticks(rotation=45, ha='right')`.
9.  `fig.tight_layout()`
10. `fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')`
11. `plt.close(fig)`

**CRITICAL RULES**:
- **NO FUNCTIONS, IMPORTS, COMMENTS (except in the designated blocks), or MARKDOWN**.
- **DIRECT CODE ONLY**: Your entire output must be executable Python.
- **LEVERAGE SEABORN**: Use Seaborn functions and their parameters (`x`, `y`, `hue`, `data`) as the primary method for plotting.

Produce the Python code now.
"""


# --- FASTAPI ENDPOINTS ---

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    """Provides a high-level summary and narrative of the uploaded dataset."""
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)
        context = get_dataframe_context(df)

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1, api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate.from_template(
            "You are a helpful data analyst. Given the following dataset context, provide a one-paragraph summary highlighting key characteristics, potential insights, and areas for further analysis.\n\nContext:\n{context_str}"
        )
        chain = prompt | llm | StrOutputParser()
        narrative = chain.invoke({"context_str": str(context)})

        return {
            "summary_stats": df.describe(include='all').to_dict(),
            "narrative": narrative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/generate-dashboard/", response_class=FileResponse)
async def generate_dashboard(file: UploadFile = File(...)):
    """Generates a full PDF dashboard from the uploaded data."""
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    context = get_dataframe_context(df)
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, api_key=GOOGLE_API_KEY)

    # --- Agent Step 1: Analyst generates textual suggestions ---
    print("Step 1: Invoking Analyst Agent to generate text report...")
    analysis_prompt = PromptTemplate.from_template(analysis_template)
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    try:
        text_report = analysis_chain.invoke(context)
        print(f"Generated text report preview: {text_report[:500]}...")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyst Agent failed. Error: {e}")

    # --- Agent Step 2: Extractor parses text report into structured JSON ---
    print("Step 2: Invoking Extractor Agent to parse the report into JSON...")
    json_parser = JsonOutputParser(pydantic_object=Suggestions)
    extraction_prompt = PromptTemplate(
        template="""You are a data extraction expert. Parse the user's text report and convert it into a structured JSON object that conforms to the provided format instructions.

CRITICAL: When extracting column mappings, ensure that every column name is exactly one of these available columns: {df_columns}

If the text report mentions column names that don't exist in the available columns, you must map them to the closest matching actual column name from the list above, or use null/None if no appropriate mapping exists.

Available columns: {df_columns}

Text Report to Parse:
{text_report}

{format_instructions}""",
        input_variables=["text_report", "df_columns"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )
    extraction_chain = extraction_prompt | llm | json_parser
    try:
        suggestions_result = extraction_chain.invoke({
            "text_report": text_report, 
            "df_columns": context["df_columns"]
        })
        chart_suggestions = suggestions_result['charts']
        print(f"Extracted {len(chart_suggestions)} chart suggestions:")
        for idx, suggestion in enumerate(chart_suggestions):
            print(f"  Chart {idx+1}: {suggestion.get('title', 'No title')} - Mapping: {suggestion.get('column_mapping', {})}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extractor Agent failed to parse report into JSON. Error: {e}")

    chart_images = []
    annotations = []
    print(f"\n=== CHART GENERATION SUMMARY ===\nTotal chart suggestions received: {len(chart_suggestions)}")
    
    # --- Agent Step 3: Coder generates Python code for each chart ---
    code_gen_prompt = PromptTemplate.from_template(code_gen_template)
    code_gen_chain = code_gen_prompt | llm | StrOutputParser()
    
    successful_charts, failed_charts, skipped_charts = 0, 0, 0

    for i, suggestion in enumerate(chart_suggestions, 1):
        title = suggestion.get('title', 'Untitled')
        print(f"\n--- Processing Chart {i}/{len(chart_suggestions)}: '{title}' ---")
        generated_code = ""
        try:
            # Validate required columns exist before calling the coder
            mapping = suggestion.get('column_mapping', {})
            print(f"Raw column mapping: {mapping}")
            
            # Extract actual column names, filtering out None, empty strings, and descriptive text
            required_cols = set()
            clean_mapping = {}
            
            for key, value in mapping.items():
                if value and isinstance(value, str) and value.strip():
                    # Clean the value and check if it's an actual column
                    clean_value = value.strip()
                    if clean_value in df.columns:
                        required_cols.add(clean_value)
                        clean_mapping[key] = clean_value
                    else:
                        # Try to find a close match in actual columns
                        clean_value_lower = clean_value.lower()
                        for col in df.columns:
                            if col.lower() == clean_value_lower or clean_value_lower in col.lower():
                                required_cols.add(col)
                                clean_mapping[key] = col
                                print(f"Mapped '{clean_value}' to actual column '{col}'")
                                break
                        else:
                            print(f"Warning: Could not map '{clean_value}' to any actual column. Skipping this mapping.")
                            clean_mapping[key] = None
                else:
                    clean_mapping[key] = None
            
            print(f"Clean column mapping: {clean_mapping}")
            print(f"Required columns: {required_cols}")
            
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                print(f"❌ Missing columns {missing_cols} for chart '{title}', skipping.")
                skipped_charts += 1
                continue
            
            if not required_cols:
                print(f"❌ No valid columns found for chart '{title}', skipping.")
                skipped_charts += 1
                continue

            # Update the suggestion with clean mapping
            suggestion['column_mapping'] = clean_mapping

            # Invoke the coder agent
            raw_code = code_gen_chain.invoke({
                "df_info": context["df_info"],
                "df_description": context["df_description"],
                "df_columns": context["df_columns"],
                "chart_type": suggestion['chart_type'],
                "title": title,
                "question": suggestion['question'],
                "column_mapping": suggestion['column_mapping'],
                "pre_processing_steps": suggestion['pre_processing_steps']
            })
            generated_code = clean_generated_code(raw_code)

            # Execute the generated code
            img_buffer = io.BytesIO()
            exec_scope = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 'np': np, 'io': io, 'img_buffer': img_buffer}
            exec(generated_code, exec_scope)
            
            img_buffer.seek(0)
            if img_buffer.getbuffer().nbytes > 1000: # Check if the image is non-empty
                chart_images.append(img_buffer)
                annotations.append(suggestion['description'])
                successful_charts += 1
                print(f"✅ Successfully generated chart '{title}'")
            else:
                failed_charts += 1
                print(f"❌ Code for chart '{title}' executed but produced an empty image. Skipping.")
        
        except Exception as e:
            failed_charts += 1
            print(f"❌ FAILED to execute code for chart '{title}'. Error: {e}")
            print(f"--- Failing Code ---\n{generated_code}\n--------------------")
            continue

    print(f"\n=== FINAL CHART GENERATION REPORT ===\n✅ Successful: {successful_charts} | ❌ Failed: {failed_charts} | ⏭️ Skipped: {skipped_charts}\n")
    if not chart_images:
        raise HTTPException(status_code=500, detail="No charts could be generated successfully. Check server logs for details.")

    # --- Step 4: Generate initial PDF for analysis ---
    initial_pdf_filename = f"{file.filename.split('.')[0]}_initial_dashboard.pdf"
    unicode_font = setup_unicode_fonts()
    
    doc = SimpleDocTemplate(initial_pdf_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontName=unicode_font, alignment=1)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=unicode_font, spaceAfter=12, leading=14)
    
    story = [Paragraph("AI-Generated Data Dashboard", title_style), Spacer(1, 0.25 * inch)]
    for img_buffer, text in zip(chart_images, annotations):
        img_buffer.seek(0)  # Reset buffer position
        img = Image(img_buffer, width=6*inch, height=3.75*inch, kind='proportional')
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"<b>Insight:</b> {text}", normal_style))
        story.append(Spacer(1, 0.25 * inch))

    doc.build(story)
    print(f"Initial PDF generated: {initial_pdf_filename}")

    # --- Step 5: Analyze the generated PDF with Google GenAI ---
    print("Step 5: Analyzing the generated PDF with Google GenAI...")
    try:
        pdf_analysis = analyze_pdf_with_genai(initial_pdf_filename, GOOGLE_API_KEY)
        print(f"PDF analysis completed. Found {pdf_analysis.total_charts} charts.")
        
        # --- Step 6: Generate enhanced PDF with detailed descriptions ---
        print("Step 6: Generating enhanced PDF with detailed descriptions...")
        timestamp = int(time.time())
        enhanced_pdf_filename = f"{file.filename.split('.')[0]}_enhanced_dashboard_{timestamp}.pdf"

        # Reset buffer positions for chart images
        for img_buffer in chart_images:
            img_buffer.seek(0)
        
        generate_enhanced_pdf(chart_images, pdf_analysis, enhanced_pdf_filename)
        print(f"Enhanced PDF generated: {enhanced_pdf_filename}")
        
        # Clean up the initial PDF
        try:
            os.remove(initial_pdf_filename)
            print("Initial PDF cleaned up.")
        except:
            pass
        
        return FileResponse(enhanced_pdf_filename, media_type="application/pdf", filename=enhanced_pdf_filename)
        
    except Exception as e:
        print(f"Error during PDF analysis: {e}")
        print("Falling back to the initial PDF...")
        return FileResponse(initial_pdf_filename, media_type="application/pdf", filename=initial_pdf_filename)



from config.database import init_db

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully.")