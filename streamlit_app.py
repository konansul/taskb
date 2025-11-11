import streamlit as st
import pandas as pd
import os
import sys
import io
import traceback
import streamlit as st
from pathlib import Path
from config.database import SessionLocal, User,UserAPIKey
from utils.api_client import register_user, login_user, logout_user, reset_password, forgot_password
from utils.token_utils import decode_access_token

# Add paths for imports
current_dir = Path(__file__).parent
utils_path = current_dir / "utils"
config_path = current_dir / "config"

sys.path.extend([str(utils_path), str(config_path), str(current_dir)])

# Import configurations
try:
    from config.settings import StreamlitConfig, AIConfig, AppConfig
except ImportError:
    # Fallback configuration
    from dataclasses import dataclass
    
    @dataclass
    class StreamlitConfig:
        page_title: str = "4Sim AI Dashboard Generator"
        page_icon: str = "📊"
        layout: str = "wide"
        initial_sidebar_state: str = "expanded"
        max_upload_size_mb: int = 200
    
    @dataclass
    class AIConfig:
        google_api_key: str = ""
        model_name: str = "gemini-2.5-flash"
        temperature: float = 0.1
        max_chart_suggestions: int = 5
    
    @dataclass
    class AppConfig:
        debug_mode: bool = False
        enable_caching: bool = True
        cache_ttl_hours: int = 24
        output_directory: str = "output"

# Import utilities
try:
    from utils.ai_agents import DataAnalyst, SuggestionExtractor, ChartCodeGenerator, ChartQueryProcessor
    from utils.data_processing import DataProcessor
    from utils.chart_generation import ChartGenerator
    from utils.models import ChartSuggestion
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# Performance optimization with new caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataset_cached(file_content: bytes, file_name: str) -> dict:
    """Cache dataset loading and processing"""
    try:
        processor = DataProcessor()
        df = processor.load_uploaded_file(file_content, file_name)
        df_context = processor.get_dataframe_context(df)
        return {
            'success': True,
            'df': df,
            'df_context': df_context,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'df': None,
            'df_context': None,
            'error': str(e)
        }

@st.cache_resource
def initialize_ai_agents_cached(api_key: str, model_name: str) -> dict:
    """Cache AI agent initialization to prevent recreating on every run"""
    try:
        return {
            'analyst': DataAnalyst(api_key, model_name),
            'extractor': SuggestionExtractor(api_key, model_name),
            'code_generator': ChartCodeGenerator(api_key, model_name),
            'query_processor': ChartQueryProcessor(api_key, model_name),
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Import components
try:
    from components.file_upload import enhanced_file_uploader, display_dataframe_preview
    from components.chart_preview import chart_preview_component, display_chart_grid
    from components.dashboard_export import dashboard_export_component, display_export_status
except ImportError as e:
    st.error(f"❌ Component import error: {e}")
    st.stop()

# Configure Streamlit page
config = StreamlitConfig()
st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout=config.layout,
    initial_sidebar_state=config.initial_sidebar_state
)

# Disable the blurring effect during updates
if 'disable_blur' not in st.session_state:
    st.session_state.disable_blur = True

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    session_vars = {
        'df': None,
        'chart_suggestions': [],
        'generated_charts': [],
        'ai_analysis_complete': False,
        'df_context': {},
        'qa_generated_chart': None,
        'qa_suggestion': None,
        'raw_csv_data': None,
        'original_filename': None,
        'gemini_api_key': '',
        'detected_tables': {},  # Store all detected tables
        'selected_table_name': None,  # Track which table was selected
        'table_selection_complete': False,  # Flag to track if table selection is done
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value



def show_user_details():
    with st.sidebar:
        st.header("👤 User Details")

        user_info = st.session_state.get("user_info")
        if not user_info:
            st.warning("🚪 You are not logged in.")
            return

        st.subheader("📌 User Information")

        st.write(f"**First Name:** {user_info.get('firstName', '-')}")
        st.write(f"**Last Name:** {user_info.get('lastName', '-')}")
        st.write(f"**Email:** {user_info.get('email', '-')}")
        st.write(f"**FIN:** {user_info.get('fin', '-')}")
        st.write(f"**Role:** {user_info.get('roles', '-')}")

        st.markdown("---")

        # Logout button (if token exists)
        if "refresh_token" in st.session_state:
            if st.button("🚪 Log Out"):
                from utils.api_client import logout_user
                res = logout_user(st.session_state["refresh_token"])
                st.session_state.clear()
                st.success("You have logged out successfully.")
        else:
            st.info("User is not authorized.")




def configure_sidebar():
    """Configure sidebar with API settings only - analysis settings moved to AI Analysis tab"""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # --- 🟢 Проверяем, вошёл ли пользователь и достаём API Key из базы ---
        user_info = st.session_state.get("user_info")
        saved_api_key = None

        if user_info:
            user_fin = user_info.get("fin")
            session = SessionLocal()
            record = session.query(UserAPIKey).filter_by(user_fin=user_fin).first()
            if record:
                saved_api_key = record.api_key
                st.session_state["gemini_api_key"] = saved_api_key
                os.environ["GOOGLE_API_KEY"] = saved_api_key
            session.close()

        # --- Форма настроек API ---
        with st.form("api_config_form"):
            st.subheader("🔐 API Settings")
            api_key = st.text_input(
                "Google API Key",
                type="password",
                value=saved_api_key or "",
                help="Enter your Google Gemini API key for AI analysis"
            )

            model_name = st.selectbox(
                "AI Model",
                ["gemini-2.5-flash", "gemini-pro"],
                help="Choose the AI model for analysis"
            )

            debug_mode = st.checkbox(
                "🐛 Debug Mode",
                value=st.session_state.get('debug_mode', False)
            )

            api_submit = st.form_submit_button("🔄 Update Settings")

        # --- Сохраняем настройки ---
        if api_submit and api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.session_state.gemini_api_key = api_key
            st.session_state.model_name = model_name
            st.session_state.debug_mode = debug_mode
            st.success("✅ Settings updated")

            # 💾 Сохраняем/обновляем ключ в базе
            if user_info:
                user_fin = user_info.get("fin")
                session = SessionLocal()
                record = session.query(UserAPIKey).filter_by(user_fin=user_fin).first()
                if record:
                    record.api_key = api_key
                else:
                    session.add(UserAPIKey(user_fin=user_fin, api_key=api_key))
                session.commit()
                session.close()
        elif api_submit and not api_key:
            st.warning("⚠️ API Key required for AI analysis")

        # --- Текущие настройки ---
        st.markdown("---")
        st.markdown("**Current Settings:**")
        if st.session_state.get('gemini_api_key'):
            st.markdown("- API: ✅ Configured")
            st.markdown(f"- Model: {st.session_state.get('model_name', 'gemini-2.5-flash')}")
        else:
            st.markdown("- API: ❌ Not configured")

        if st.session_state.get('debug_mode'):
            st.markdown("- Debug: 🐛 Enabled")

        st.markdown("---")
        st.markdown("📊 **Analysis settings** are configured in the 'AI Analysis' tab")

        # --- User Details остаётся как в оригинале ---
        show_user_details()

        # Возвращаем значения
        return (
            st.session_state.get('gemini_api_key', ''),
            st.session_state.get('model_name', 'gemini-2.5-flash')
        )




def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("🤖 4Sim AI Dashboard Generator")
    st.markdown(
        """
    Welcome to the intelligent dashboard generator! Upload your data and let AI create 
    beautiful, insightful visualizations automatically.
        """
    )
    
    # Sidebar configuration with forms for performance
    api_key, model_name = configure_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload Data",
        "🔍 AI Analysis",
        "💬 Q&A with Data",
        "📄 Export & Download",
        "📊 My Files"
    ])
    
    with tab1:
        handle_data_upload()
        
    with tab2:
        if st.session_state.df is not None and api_key:
            handle_ai_analysis(api_key, model_name)
        elif not api_key:
            st.warning("⚠️ Please configure your Google API key in the sidebar first.")
        else:
            st.info("👆 Please upload a dataset first in the 'Upload Data' tab.")
            
    with tab3:
        if st.session_state.df is not None and api_key:
            handle_qa_with_data(api_key, model_name)
        elif not api_key:
            st.warning("⚠️ Please configure your Google API key in the sidebar first.")
        else:
            st.info("👆 Please upload a dataset first in the 'Upload Data' tab.")
            
    with tab4:
        if api_key:
            handle_export_download(api_key, model_name)
        else:
            st.warning("⚠️ Please configure your Google API key in the sidebar first.")

    with tab5:
        from config.database import SessionLocal, UserFile
        st.subheader("📊 My Files")

        current_fin = st.session_state.get("user_info", {}).get("fin")

        if current_fin:
            session = SessionLocal()
            files = (
                session.query(UserFile)
                .filter_by(user_fin=current_fin)
                .order_by(UserFile.upload_date.desc())
                .all()
            )

            if not files:
                st.info("You don’t have any saved files yet.")
            else:
                for i, f in enumerate(files, start=1):
                    cols = st.columns([0.5, 3, 1, 1])  # номер, имя, скачать, удалить

                    with cols[0]:
                        st.markdown(f"**{i}.**")

                    with cols[1]:
                        st.write(f"📄 **{f.filename}** — {f.upload_date.strftime('%Y-%m-%d %H:%M')}")

                    with cols[2]:
                        st.download_button(
                            label="⬇️ Download",
                            data=f.file_data,
                            file_name=f.filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_{i}"
                        )

                    with cols[3]:
                        if st.button("🗑️ Delete", key=f"delete_{i}"):
                            session.delete(f)
                            session.commit()
                            st.warning(f"File '{f.filename}' deleted.")
                            st.rerun()

            session.close()
        else:
            st.warning("⚠️ Please log in to view your saved files.")



def handle_data_upload():
    """Handle file upload and data preview with optional AI formatting"""

    uploaded_file = enhanced_file_uploader()

    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Processing your file..."):
                file_content = uploaded_file.read()

                # --- Сохраняем файл в базу, если такого ещё нет ---
                from config.database import SessionLocal, UserFile
                session = SessionLocal()

                user_info = st.session_state.get("user_info", {})
                current_fin = user_info.get("fin")

                if current_fin:
                    # Проверяем, есть ли файл с таким же именем у этого пользователя
                    existing_file = (
                        session.query(UserFile)
                        .filter_by(user_fin=current_fin, filename=uploaded_file.name)
                        .first()
                    )

                    if existing_file:
                        st.info(f"ℹ️ File '{uploaded_file.name}' already exists in your saved files. It will not be saved again.")
                    else:
                        new_file = UserFile(
                            user_fin=current_fin,
                            filename=uploaded_file.name,
                            file_data=file_content
                        )
                        session.add(new_file)
                        session.commit()
                        st.success(f"✅ File '{uploaded_file.name}' has been saved to the database!")
                else:
                    st.warning("⚠️ You are not logged in, the file will not be saved to the database.")

                session.close()

                # --- Загружаем данные для отображения ---
                df = DataProcessor.read_uploaded_file(file_content, uploaded_file.name)
                raw_csv_data = DataProcessor.extract_raw_csv_data(file_content, uploaded_file.name)

                st.session_state.df = df
                st.session_state.df_context = DataProcessor.get_dataframe_context(df)
                st.session_state.raw_csv_data = raw_csv_data
                st.session_state.original_filename = uploaded_file.name

            # --- Отображаем превью ---
            display_dataframe_preview(df)
            st.markdown("---")

            api_key = st.session_state.get('gemini_api_key', '')

            # --- Работа с таблицами ---
            if (st.session_state.get('detected_tables') and
                st.session_state.get('selected_table_name') and
                st.session_state.get('df') is not None and
                st.session_state.get('table_selection_complete', False)):

                st.success(f"✅ **Table Selection Complete:** {st.session_state.selected_table_name}")
                st.info("💾 **Selected table is ready for AI analysis!** Go to the 'AI Analysis' tab.")

                st.markdown("**📊 Selected Table Preview:**")
                display_dataframe_preview(st.session_state.df)

                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🔄 Change Table"):
                        st.session_state.table_selection_complete = False
                        from components.file_upload import multi_table_selection_component
                        selected_df, selected_name = multi_table_selection_component(st.session_state.detected_tables)
                        if selected_df is not None:
                            st.session_state.df = selected_df
                            st.session_state.selected_table_name = selected_name
                            st.session_state.df_context = DataProcessor.get_dataframe_context(selected_df)
                            st.session_state.table_selection_complete = True
                            st.rerun()

            else:
                from components.file_upload import csv_formatting_component

                formatted_df, was_formatted = csv_formatting_component(
                    df=df,
                    raw_csv_data=raw_csv_data,
                    api_key=api_key,
                    filename=uploaded_file.name
                )

                if was_formatted and formatted_df is not None:
                    st.session_state.df = formatted_df
                    st.session_state.df_context = DataProcessor.get_dataframe_context(formatted_df)

                    if not st.session_state.get('detected_tables'):
                        st.markdown("---")
                        st.subheader("📊 Updated Dataset Preview")
                        display_dataframe_preview(formatted_df)

                    st.info("✨ **Next Step:** Go to the 'AI Analysis' tab to generate chart suggestions!")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)


def handle_qa_with_data(api_key: str, model_name: str):

    """Handle Q&A with data functionality"""

    st.subheader("💬 Q&A with Data")
    st.markdown("Ask natural language questions about your data and get instant visualizations!")
    
    # Dataset info
    df = st.session_state.df
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Available Columns", len(df.columns))
    
    # Show available columns for reference
    with st.expander("📋 Available Columns", expanded=False):
        st.write("**Available columns in your dataset:**")
        for i, col in enumerate(df.columns, 1):
            col_type = str(df[col].dtype)
            st.write(f"{i}. `{col}` ({col_type})")
    
    # Query input section
    st.markdown("---")
    st.subheader("� Ask a Question")
    
    # Example queries
    example_queries = [
        "Show me the distribution of ages",
        "Compare the relationship between height and weight",
        "Show how performance varies by category",
        "Create a correlation heatmap",
        "Display the trend over time"
    ]
    
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("**Here are some example questions you can ask:**")
        for example in example_queries:
            if st.button(f"📊 {example}", key=f"example_{example}", use_container_width=True):
                st.session_state.qa_query = example
    
    # User input form
    with st.form("qa_query_form"):
        user_query = st.text_area(
            "What chart would you like to see?",
            value=st.session_state.get('qa_query', ''),
            height=100,
            placeholder="e.g., 'Show me a scatter plot of height vs weight colored by gender'",
            help="Describe the chart you want to create using natural language. Be specific about which columns to use."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_chart = st.form_submit_button(
                "🚀 Generate Chart", 
                type="primary",
                use_container_width=True
            )
        with col2:
            clear_query = st.form_submit_button(
                "�️ Clear",
                use_container_width=True
            )
    
    if clear_query:
        st.session_state.qa_generated_chart = None
        st.session_state.qa_suggestion = None
        st.session_state.qa_query = ""
        st.rerun()
    
    if generate_chart and user_query.strip():
        generate_qa_chart(user_query, api_key, model_name)
    elif generate_chart and not user_query.strip():
        st.warning("⚠️ Please enter a question about your data.")
    
    # Display generated chart if available
    if st.session_state.qa_generated_chart and st.session_state.qa_suggestion:
        display_qa_generated_chart()




def generate_qa_chart(user_query: str, api_key: str, model_name: str):
    """Generate chart from natural language query"""
    try:
        # Get AI agents
        agents_result = initialize_ai_agents_cached(api_key, model_name)
        
        if not agents_result['success']:
            st.error(f"❌ Failed to initialize AI agents: {agents_result['error']}")
            return
        
        query_processor = agents_result['query_processor']
        code_generator = agents_result['code_generator']
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Process the query
        status_text.text("🔍 Processing your query...")
        progress_bar.progress(20)
        
        suggestion = query_processor.process_query(user_query, st.session_state.df_context)
        
        progress_bar.progress(50)
        status_text.text("🎨 Generating visualization...")
        
        # Step 2: Generate chart code
        generated_code = code_generator.generate_chart_code(suggestion, st.session_state.df_context)
        cleaned_code = code_generator.clean_generated_code(generated_code)
        
        progress_bar.progress(70)
        status_text.text("⚡ Executing chart generation...")
        
        # Step 3: Execute the code
        import io
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        img_buffer = io.BytesIO()
        exec_scope = {
            'df': st.session_state.df, 'pd': pd, 'plt': plt, 'sns': sns, 
            'np': __import__('numpy'), 'io': io, 'img_buffer': img_buffer
        }
        
        exec(cleaned_code, exec_scope)
        
        progress_bar.progress(90)
        status_text.text("✅ Chart generated successfully!")
        
        # Store results
        img_buffer.seek(0)
        if img_buffer.getbuffer().nbytes > 1000:
            st.session_state.qa_generated_chart = img_buffer
            st.session_state.qa_suggestion = suggestion
            progress_bar.progress(100)
            status_text.text("🎉 Ready to view your chart!")
        else:
            st.error("❌ Chart generation failed - no image data produced")
            progress_bar.progress(0)
            status_text.text("")
        
    except Exception as e:
        st.error(f"❌ Error generating chart: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)




def display_qa_generated_chart():
    """Display the generated chart from Q&A with add to dashboard option"""
    st.markdown("---")
    st.subheader("✨ Generated Chart")
    
    suggestion = st.session_state.qa_suggestion
    chart_buffer = st.session_state.qa_generated_chart
    
    # Display chart details and image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**🎯 Question:** {suggestion.question}")
        st.markdown(f"**📊 Chart Type:** {suggestion.chart_type}")
        st.markdown(f"**💡 Description:** {suggestion.description}")
        
        if suggestion.pre_processing_steps != "None":
            st.markdown(f"**⚙️ Pre-processing:** {suggestion.pre_processing_steps}")
        
        # Column mapping details
        st.markdown("**🗂️ Column Mapping:**")
        for key, value in suggestion.column_mapping.items():
            if value:
                st.markdown(f"- **{key}:** `{value}`")
    
    with col2:
        chart_buffer.seek(0)
        st.image(chart_buffer, caption=suggestion.title, use_container_width=True)
    
    # Add to dashboard functionality
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**Add this chart to your dashboard for export:**")
    
    with col2:
        if st.button("➕ Add to Dashboard", type="primary", use_container_width=True):
            add_qa_chart_to_dashboard()
    
    with col3:
        if st.button("🔄 Generate New", use_container_width=True):
            st.session_state.qa_generated_chart = None
            st.session_state.qa_suggestion = None
            st.rerun()




def add_qa_chart_to_dashboard():

    """Add the Q&A generated chart to the main dashboard"""

    try:
        suggestion = st.session_state.qa_suggestion
        chart_buffer = st.session_state.qa_generated_chart
        
        # Check if chart already exists
        existing_titles = [title for _, title in st.session_state.generated_charts]
        if suggestion.title in existing_titles:
            st.warning(f"⚠️ Chart '{suggestion.title}' already exists in dashboard!")
            return
        
        # Add to suggestions and charts
        st.session_state.chart_suggestions.append(suggestion)
        
        # Create a new buffer for the chart (to avoid conflicts)
        chart_buffer.seek(0)
        new_buffer = io.BytesIO(chart_buffer.read())
        st.session_state.generated_charts.append((new_buffer, suggestion.title))
        
        st.success(f"✅ Chart '{suggestion.title}' added to dashboard! Check the 'Export & Download' tab.")
        
        # Clear the Q&A results
        st.session_state.qa_generated_chart = None
        st.session_state.qa_suggestion = None
        
    except Exception as e:
        st.error(f"❌ Error adding chart to dashboard: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)



def handle_ai_analysis(api_key: str, model_name: str):

    """Handle AI-powered data analysis with performance optimizations"""
    
    st.subheader("🔍 AI-Powered Data Analysis")
    
    if not api_key:
        st.error("❌ Please configure your Google API key in the sidebar to proceed.")
        return
    
    df = st.session_state.df
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Analysis configuration using form for performance
    with st.expander("⚙️ Analysis Configuration", expanded=True):
        with st.form("analysis_trigger_form"):
            st.markdown("**Configure and run your analysis:**")
            
            col1, col2 = st.columns(2)
            with col1:
                num_suggestions = st.slider(
                    "Number of Charts",
                    min_value=3,
                    max_value=12,
                    value=st.session_state.get('num_suggestions', 5),
                    help="How many chart suggestions to generate"
                )
            
            with col2:
                analysis_focus = st.selectbox(
                    "Analysis Focus",
                    ["Comprehensive Overview", "Distribution Analysis", "Correlation Analysis", "Time Series Analysis"],
                    index=st.session_state.get('analysis_focus_index', 0),
                    help="What type of analysis to prioritize"
                )
            
            run_analysis = st.form_submit_button(
                "🚀 Generate AI Dashboard",
                type="primary",
                help="Click to start the AI analysis and chart generation"
            )
        
        # Only run analysis when form is submitted
        if run_analysis:
            st.session_state.num_suggestions = num_suggestions
            st.session_state.analysis_focus = analysis_focus
            st.session_state.analysis_focus_index = ["Comprehensive Overview", "Distribution Analysis", "Correlation Analysis", "Time Series Analysis"].index(analysis_focus)
            analyze_data_with_ai(api_key, model_name, num_suggestions, analysis_focus)
            st.session_state.analysis_focus_value = "Balanced Analysis"
        
    
    # Display results if available
    if st.session_state.chart_suggestions:
        st.success(f"✅ Analysis complete! Generated {len(st.session_state.chart_suggestions)} chart suggestions with visualizations.")
        
        # Show chart suggestions with interactive preview
        st.markdown("---")
        display_chart_suggestions_with_preview()




def display_chart_suggestions_with_preview():

    """Display chart suggestions with generated charts - simplified version for AI Analysis tab"""
    
    st.subheader("📊 Generated Chart Suggestions & Visualizations")
    
    if not st.session_state.chart_suggestions:
        st.warning("No chart suggestions available.")
        return
    
    if not st.session_state.generated_charts:
        st.warning("No charts were generated. Please try running the analysis again.")
        return
    
    # Display charts in a clean format
    suggestions = st.session_state.chart_suggestions
    charts = st.session_state.generated_charts
    
    # Create a mapping of chart titles to images
    chart_map = {chart_title: chart_buffer for chart_buffer, chart_title in charts}
    
    for i, suggestion in enumerate(suggestions):
        # Handle both dict and object formats
        try:
            if isinstance(suggestion, dict):
                title = suggestion.get('title', f'Chart {i+1}')
                question = suggestion.get('question', 'No question specified')
                chart_type = suggestion.get('chart_type', 'Unknown')
                description = suggestion.get('description', 'No description available')
            else:
                title = getattr(suggestion, 'title', f'Chart {i+1}')
                question = getattr(suggestion, 'question', 'No question specified')
                chart_type = getattr(suggestion, 'chart_type', 'Unknown')
                description = getattr(suggestion, 'description', 'No description available')
            
            with st.expander(f"📈 Chart {i+1}: {title}", expanded=True):
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**🎯 Question:** {question}")
                    st.markdown(f"**📊 Chart Type:** {chart_type}")
                    st.markdown(f"**💡 Description:** {description}")
                
                with col2:
                    # Display the chart if it exists
                    if title in chart_map:
                        chart_buffer = chart_map[title]
                        chart_buffer.seek(0)
                        st.image(chart_buffer, caption=title, use_container_width=True)
                    else:
                        st.warning("⚠️ Chart image not available")
        
        except Exception as e:
            st.error(f"Error displaying suggestion {i+1}: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug - Suggestion data: {suggestion}")
                st.write(f"Debug - Suggestion type: {type(suggestion)}")




def analyze_data_with_ai(api_key: str, model_name: str, num_suggestions: int, analysis_focus: str):

    """Perform AI analysis with progress tracking and automatic chart generation using cached agents"""
    
    try:
        # Use cached AI agent initialization for better performance
        agents_result = initialize_ai_agents_cached(api_key, model_name)
        
        if not agents_result['success']:
            st.error(f"❌ Failed to initialize AI agents: {agents_result['error']}")
            return
        
        analyst = agents_result['analyst']
        extractor = agents_result['extractor']
        code_generator = agents_result['code_generator']
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate textual analysis
        status_text.text("🔍 Step 1/4: Analyzing dataset patterns...")
        progress_bar.progress(10)
        
        text_report = analyst.analyze_dataset(st.session_state.df_context, num_suggestions)
        
        progress_bar.progress(30)
        status_text.text("📊 Step 2/4: Extracting chart suggestions...")
        
        # Step 2: Extract structured suggestions with correct parameter count
        suggestions = extractor.extract_suggestions(
            text_report, 
            st.session_state.df.columns.tolist(),
            num_suggestions  # Pass the num_suggestions parameter
        )
        
        progress_bar.progress(50)
        status_text.text("🎨 Step 3/4: Generating chart visualizations...")
        
        # Step 3: Generate charts for all suggestions automatically
        generated_charts = []
        df = st.session_state.df
        df_context = st.session_state.df_context
        
        for i, suggestion in enumerate(suggestions):
            try:
                # Generate code
                generated_code = code_generator.generate_chart_code(suggestion, df_context)
                cleaned_code = code_generator.clean_generated_code(generated_code)
                
                # Execute code
                import io
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                img_buffer = io.BytesIO()
                exec_scope = {
                    'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 
                    'np': __import__('numpy'), 'io': io, 'img_buffer': img_buffer
                }
                
                exec(cleaned_code, exec_scope)
                
                # Store chart if generation was successful
                img_buffer.seek(0)
                if img_buffer.getbuffer().nbytes > 1000:
                    # Get title safely
                    if isinstance(suggestion, dict):
                        title = suggestion.get('title', f'Chart {i+1}')
                    else:
                        title = getattr(suggestion, 'title', f'Chart {i+1}')
                    
                    generated_charts.append((img_buffer, title))
                    
                # Update progress
                chart_progress = 50 + int((i + 1) / len(suggestions) * 30)
                progress_bar.progress(chart_progress)
                
            except Exception as e:
                # Continue with other charts if one fails
                if st.session_state.get('debug_mode', False):
                    st.error(f"Chart {i+1} generation failed: {str(e)}")
                continue
        
        progress_bar.progress(85)
        status_text.text("✅ Step 4/4: Finalizing results...")
        
        # Store results
        st.session_state.chart_suggestions = suggestions
        st.session_state.generated_charts = generated_charts
        st.session_state.ai_analysis_complete = True
        
        progress_bar.progress(100)
        status_text.text("🎉 Analysis and chart generation complete!")
        
        # Show success message
        st.success(f"🎉 Successfully generated {len(suggestions)} chart suggestions with {len(generated_charts)} visualizations!")
        
        # Display raw analysis if in debug mode
        if st.session_state.get('debug_mode', False):
            with st.expander("🐛 Debug: Raw AI Analysis"):
                st.text_area("Text Report", text_report, height=200)
        
    except Exception as e:
        st.error(f"❌ AI Analysis failed: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)




def handle_export_download(api_key: str, model_name: str):

    """Handle dashboard export and download"""
    
    st.subheader("� Export & Download")
    
    # Show export status
    chart_count = len(st.session_state.get('generated_charts', []))
    display_export_status(chart_count)
    
    if chart_count > 0:
        # Export component moved to the top as requested
        st.markdown("---")
        dashboard_export_component(
            st.session_state.generated_charts,
            api_key
        )
        
        # Display chart gallery below export controls
        st.markdown("---")
        st.subheader("🎨 Generated Charts Gallery")
        display_chart_grid(st.session_state.generated_charts)
    else:
        st.info("📊 No charts available for export. Generate some charts first in the 'AI Analysis' tab!")



st.title("🔐 Authentication & User Management")

tab1, tab2, tab3 = st.tabs(["Login", "Register", "Password / Logout"])

# Login
with tab1:
    fin_login = st.text_input("FIN", key="login_fin")
    password_login = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        res = login_user(fin_login, password_login)

        if "accessToken" in res:
            st.session_state["access_token"] = res["accessToken"]
            st.session_state["refresh_token"] = res["refreshToken"]

            # 🔹 Decode token
            user_info = decode_access_token(res["accessToken"])
            st.session_state["user_info"] = user_info

            st.success(f"✅ Login successful!")

# Registration
with tab2:
    first_name = st.text_input("First Name", key="reg_first")
    last_name = st.text_input("Last Name", key="reg_last")
    fin = st.text_input("FIN", key="reg_fin")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")

    if st.button("Register", key="register_btn"):
        res = register_user(first_name, last_name, fin, email, password)

        if "accessToken" in res:
            st.success("🎉 Registration successful! You can now log in to your account.")
        elif "error" in res:
            st.error(f"❌ Registration failed: {res.get('response') or res.get('error')}")
        else:
            st.warning("⚠️ Unexpected server response. Please try again later.")



# Password & Logout
with tab3:
    st.subheader("🔁 Reset Password")
    email_reset = st.text_input("Email for recovery", key="reset_email")
    otp = st.text_input("OTP (if you already have one)", key="reset_otp")
    new_pass = st.text_input("New Password", type="password", key="reset_pass")

    if st.button("Reset Password", key="reset_btn"):
        res = reset_password(email_reset, otp, new_pass)

    st.markdown("---")

    st.subheader("🚪 Logout")
    if "refresh_token" in st.session_state:
        if st.button("Log Out", key="logout_btn"):
            res = logout_user(st.session_state["refresh_token"])
            st.session_state.clear()
            st.success("👋 You have successfully logged out.")
    else:
        st.info("You are not logged in yet.")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)