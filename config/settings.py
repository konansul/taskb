import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class StreamlitConfig:
    """Streamlit-specific configuration"""
    page_title: str = "4Sim AI Dashboard Generator"
    page_icon: str = "📊"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    max_upload_size_mb: int = 200
    
@dataclass
class AIConfig:
    """AI service configuration"""
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_chart_suggestions: int = 5

@dataclass
class AppConfig:
    """Application-wide configuration"""
    debug_mode: bool = os.getenv("DEBUG", "False").lower() == "true"
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    output_directory: str = "output"
