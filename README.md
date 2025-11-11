# 4Sim AI Dashboard Generator - Streamlit Implementation

## Overview

This is a comprehensive Streamlit implementation of the AI-powered dashboard generator that transforms CSV/Excel data into intelligent visualizations using Google Gemini AI.

## Features

- 🤖 **AI-Powered Analysis**: Automatic data analysis using Google Gemini
- � **Smart CSV Formatting**: AI-powered cleaning of complex/unstructured CSV files
- �📊 **Smart Chart Suggestions**: Generate 5 intelligent chart suggestions
- 🎨 **Interactive Preview**: Real-time chart generation and editing
- 📄 **Multiple Export Formats**: PDF, ZIP, individual images
- 🔧 **Configurable Settings**: Customizable AI parameters
- 🚀 **Easy Deployment**: Docker support included

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd 4sim-gov-ai-mvp/taskb

# Install dependencies
pip install -r ../shared/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Google API key
```

### 2. Configuration

Add your Google API key to `.env`:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the Application

#### Option A: Direct Python
```bash
python run_streamlit.py
```

#### Option B: Streamlit CLI
```bash
streamlit run streamlit_app.py
```

#### Option C: Docker
```bash
# From project root
docker-compose up streamlit-app
```

The application will be available at `http://localhost:8501`

## User Guide

### Step 1: Upload Data
1. Go to the "📁 Upload Data" tab
2. Upload your CSV or Excel file (max 200MB)
3. Preview your data in the interactive tables
4. **NEW**: Use AI-powered CSV formatting for complex files (e.g., from stat.gov.az)
   - System automatically detects if formatting is recommended
   - Preview changes before applying
   - Handles multi-level headers and unstructured data

### Step 2: AI Analysis
1. Navigate to "🔍 AI Analysis" tab
2. Configure your Google API key in the sidebar
3. Adjust analysis settings (number of suggestions, focus area)
4. Click "🚀 Start AI Analysis"

### Step 3: Chart Preview
1. Switch to "📊 Dashboard Preview" tab
2. Review AI-generated chart suggestions
3. Generate individual charts with the preview buttons
4. Edit chart parameters if needed

### Step 4: Export & Download
1. Open "📄 Export & Download" tab
2. Choose your preferred export format:
   - Individual Images (ZIP)
   - Simple PDF
   - Enhanced PDF (with AI analysis)
3. Download your completed dashboard

## Architecture

```
taskb/
├── streamlit_app.py           # Main Streamlit application
├── config/
│   └── settings.py           # Configuration classes
├── utils/
│   ├── ai_agents.py          # AI analysis agents
│   ├── data_processing.py    # Data handling utilities
│   ├── chart_generation.py   # Chart creation engine
│   ├── pdf_utils.py          # PDF generation and analysis
│   ├── models.py             # Pydantic models
│   └── prompts.py            # AI prompt templates
├── components/
│   ├── file_upload.py        # File upload widget + CSV formatting
│   ├── chart_preview.py      # Chart preview component
│   └── dashboard_export.py   # Export functionality
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## Key Components

### AI Agents
- **DataAnalyst**: Analyzes datasets and generates chart suggestions
- **SuggestionExtractor**: Converts text analysis to structured data
- **ChartCodeGenerator**: Creates executable Python visualization code
- **CSVFormatter**: AI-powered cleaning and structuring of complex CSV files
- **ChartQueryProcessor**: Handles natural language chart queries

### Data Processing
- **DataProcessor**: Handles file uploads and DataFrame operations
- **ChartGenerator**: Manages chart creation and execution

### User Interface
- **Enhanced File Upload**: Drag-drop with validation and preview
- **Interactive Charts**: Real-time generation with editing capabilities
- **Multi-format Export**: PDF, images, and enhanced AI analysis

## Configuration Options

### Sidebar Settings
- **Google API Key**: Required for AI analysis
- **Max Suggestions**: 3-10 chart suggestions (default: 5)
- **AI Model**: Choose between Gemini models
- **Debug Mode**: Enable detailed error reporting

### Environment Variables
```bash
GOOGLE_API_KEY=your_api_key
DEBUG=False
OUTPUT_DIRECTORY=output
STREAMLIT_SERVER_PORT=8501
```

## API Integration

The Streamlit app preserves all core functionality from the original FastAPI implementation:

- Multi-agent AI system (Analyst → Extractor → Coder)
- Advanced prompt engineering for chart suggestions
- Intelligent column mapping and validation
- Seaborn-based chart generation
- PDF analysis with Google GenAI

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure Google API key is correctly set
   - Verify API key has Gemini access enabled

2. **File Upload Issues**
   - Check file format (CSV, Excel only)
   - Ensure file size < 200MB
   - Try UTF-8 encoding for CSV files

3. **Chart Generation Failures**
   - Verify column mappings match dataset
   - Check for missing or invalid data
   - Enable debug mode for detailed errors

4. **Memory Issues**
   - Use smaller datasets (< 100K rows recommended)
   - Close other resource-intensive applications
   - Consider using Docker with increased memory

### Performance Tips
- Enable caching for repeated analyses
- Use appropriate chart types for data size
- Clear generated charts periodically
- Monitor memory usage with large datasets

## Development

### Adding New Chart Types
1. Update prompts in `utils/prompts.py`
2. Modify chart type options in `components/chart_preview.py`
3. Test with various datasets

### Extending Export Formats
1. Add new format to `components/dashboard_export.py`
2. Implement export logic
3. Update UI options

### Custom AI Models
1. Modify `config/settings.py`
2. Update agent initialization in `utils/ai_agents.py`
3. Test compatibility with new models

## Deployment

### Production Deployment
```bash
# Build and run with Docker
docker-compose up -d streamlit-app

# Or deploy to cloud platforms
# Streamlit Cloud, Heroku, AWS, etc.
```

### Environment Setup
- Python 3.11+
- 4GB+ RAM recommended
- Google API access
- Internet connection for AI services

## Support

For issues or questions:
- Check the troubleshooting section
- Review debug output in debug mode
- Contact the development team

## License

This project is part of the 4Sim Government AI MVP suite.
