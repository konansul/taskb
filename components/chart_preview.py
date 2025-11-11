import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from typing import List, Tuple, Dict
import sys
import os
from pydantic import BaseModel, Field

# Define ChartSuggestion locally to avoid import issues
class ChartSuggestion(BaseModel):
    title: str = Field(description="The concise, descriptive title from 'Chart Suggestion Title'.")
    question: str = Field(description="The analytical question from 'The Question it Answers'.")
    chart_type: str = Field(description="The single chart type from 'Chart Type', preferably from Seaborn.")
    pre_processing_steps: str = Field(description="The data manipulation steps from 'Data Pre-processing/Aggregation (if any)'. Should be 'None' if no steps are required.")
    column_mapping: Dict[str, str] = Field(description="A dictionary with keys like 'X-Axis', 'Y-Axis', 'Color/Hue (Optional)', 'Facet (Optional)' and values that are exact column names from the dataset or None.")
    description: str = Field(description="The rationale and insight from 'Rationale and Insight'.")

def chart_preview_component(
    suggestions: List, 
    df: pd.DataFrame,
    code_generator,
    df_context: Dict[str, str]
) -> List[Tuple[io.BytesIO, str]]:
    """Interactive chart preview with editing capabilities"""
    
    st.subheader("📊 Chart Suggestions & Preview")
    
    if not suggestions:
        st.warning("No chart suggestions available. Please run AI analysis first.")
        return []
    
    st.markdown(f"🎯 **{len(suggestions)} AI-generated chart suggestions ready for preview**")
    
    generated_charts = []
    
    for i, suggestion in enumerate(suggestions):
        # Handle both dict and object formats
        try:
            if isinstance(suggestion, dict):
                title = suggestion.get('title', f'Chart {i+1}')
                question = suggestion.get('question', 'No question specified')
                chart_type = suggestion.get('chart_type', 'histplot')
                description = suggestion.get('description', 'No description available')
                pre_processing_steps = suggestion.get('pre_processing_steps', 'None')
                column_mapping = suggestion.get('column_mapping', {})
            else:
                title = getattr(suggestion, 'title', f'Chart {i+1}')
                question = getattr(suggestion, 'question', 'No question specified')
                chart_type = getattr(suggestion, 'chart_type', 'histplot')
                description = getattr(suggestion, 'description', 'No description available')
                pre_processing_steps = getattr(suggestion, 'pre_processing_steps', 'None')
                column_mapping = getattr(suggestion, 'column_mapping', {})
                
        except Exception as e:
            st.error(f"Error processing suggestion {i+1}: {str(e)}")
            # Show debug info if available
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug - Suggestion {i+1} data:", suggestion)
                st.write(f"Debug - Suggestion type:", type(suggestion))
            continue
            
        # Add individual error boundary for each chart
        try:
            with st.expander(f"📈 Chart {i+1}: {title}", expanded=i == 0):
                
                # Display suggestion details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🎯 Question:** {question}")
                    st.markdown(f"**📊 Chart Type:** {chart_type}")
                    st.markdown(f"**💡 Description:** {description}")
                    
                    if pre_processing_steps != "None":
                        st.markdown(f"**⚙️ Pre-processing:** {pre_processing_steps}")
                
                with col2:
                    st.markdown("**🗂️ Column Mapping:**")
                    for key, value in column_mapping.items():
                        if value:
                            st.markdown(f"• **{key}:** `{value}`")
                        else:
                            st.markdown(f"• **{key}:** _None_")
                
                # Chart generation controls
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    generate_btn = st.button(
                        f"🚀 Generate Chart {i+1}", 
                        key=f"generate_{i}",
                        type="primary"
                    )
                
                with col2:
                    edit_btn = st.button(
                        f"✏️ Edit", 
                        key=f"edit_{i}",
                        help="Edit chart parameters"
                    )
                
                with col3:
                    if f"chart_{i}_generated" in st.session_state:
                        st.success("✅ Generated")
                
                # Edit mode
                if edit_btn or f"edit_mode_{i}" in st.session_state:
                    st.session_state[f"edit_mode_{i}"] = True
                    
                    with st.form(f"edit_form_{i}"):
                        st.markdown("**Edit Chart Parameters:**")
                        
                        # Editable fields
                        new_title = st.text_input("Chart Title", value=title)
                        new_chart_type = st.selectbox(
                            "Chart Type", 
                            ["histplot", "scatterplot", "boxplot", "violinplot", "barplot", "countplot", "lineplot", "heatmap"],
                            index=0
                        )
                        
                        # Column mapping edits
                        available_columns = ["None"] + list(df.columns)
                        
                        # Helper function to safely get index
                        def safe_index(value, options_list):
                            if value and value in options_list:
                                return options_list.index(value)
                            return 0  # Default to "None"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            x_axis = st.selectbox("X-Axis", available_columns, 
                                                index=safe_index(column_mapping.get("X-Axis"), available_columns))
                            y_axis = st.selectbox("Y-Axis", available_columns,
                                                index=safe_index(column_mapping.get("Y-Axis"), available_columns))
                        
                        with col2:
                            hue = st.selectbox("Color/Hue", available_columns,
                                             index=safe_index(column_mapping.get("Color/Hue (Optional)"), available_columns))
                            facet = st.selectbox("Facet", available_columns,
                                               index=safe_index(column_mapping.get("Facet (Optional)"), available_columns))
                        
                        # Submit button for the form
                        submitted = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
                        
                        if submitted:
                            # Update suggestion
                            if isinstance(suggestion, dict):
                                suggestion['title'] = new_title
                                suggestion['chart_type'] = new_chart_type
                                suggestion['column_mapping'] = {
                                    "X-Axis": x_axis if x_axis != "None" else None,
                                    "Y-Axis": y_axis if y_axis != "None" else None,
                                    "Color/Hue (Optional)": hue if hue != "None" else None,
                                    "Facet (Optional)": facet if facet != "None" else None
                                }
                            else:
                                suggestion.title = new_title
                                suggestion.chart_type = new_chart_type
                                suggestion.column_mapping = {
                                    "X-Axis": x_axis if x_axis != "None" else None,
                                    "Y-Axis": y_axis if y_axis != "None" else None,
                                    "Color/Hue (Optional)": hue if hue != "None" else None,
                                    "Facet (Optional)": facet if facet != "None" else None
                                }
                            st.success("✅ Changes saved!")
                            del st.session_state[f"edit_mode_{i}"]
                            st.rerun()
                
                # Generate chart
                if generate_btn:
                    with st.spinner(f"Generating chart {i+1}..."):
                        try:
                            # Generate code
                            generated_code = code_generator.generate_chart_code(suggestion, df_context)
                            cleaned_code = code_generator.clean_generated_code(generated_code)
                            
                            # Execute code
                            img_buffer = io.BytesIO()
                            exec_scope = {
                                'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 
                                'np': __import__('numpy'), 'io': io, 'img_buffer': img_buffer
                            }
                            
                            exec(cleaned_code, exec_scope)
                            
                            # Display chart
                            img_buffer.seek(0)
                            if img_buffer.getbuffer().nbytes > 1000:
                                st.image(img_buffer, caption=title, use_container_width=True)
                                generated_charts.append((img_buffer, title))
                                st.session_state[f"chart_{i}_generated"] = True
                                st.success(f"✅ Chart {i+1} generated successfully!")
                            else:
                                st.error("❌ Chart generation failed - no output produced")
                                
                        except Exception as e:
                            st.error(f"❌ Chart generation failed: {str(e)}")
                            if st.session_state.get('debug_mode', False):
                                st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error displaying chart {i+1}: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)
    
    return generated_charts

def display_chart_grid(chart_images: List[Tuple[io.BytesIO, str]]):
    """Display generated charts in a grid layout"""
    
    if not chart_images:
        st.info("No charts generated yet. Generate some charts in the preview section above.")
        return
    
    st.subheader("🎨 Generated Charts Gallery")
    
    # Display charts in a grid
    cols_per_row = 2
    for i in range(0, len(chart_images), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(chart_images):
                chart_buffer, title = chart_images[i + j]
                chart_buffer.seek(0)
                with col:
                    st.image(chart_buffer, caption=title, use_container_width=True)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from typing import List, Tuple, Dict
import sys
import os
from pydantic import BaseModel, Field

# Define ChartSuggestion locally to avoid import issues
class ChartSuggestion(BaseModel):
    title: str = Field(description="The concise, descriptive title from 'Chart Suggestion Title'.")
    question: str = Field(description="The analytical question from 'The Question it Answers'.")
    chart_type: str = Field(description="The single chart type from 'Chart Type', preferably from Seaborn.")
    pre_processing_steps: str = Field(description="The data manipulation steps from 'Data Pre-processing/Aggregation (if any)'. Should be 'None' if no steps are required.")
    column_mapping: Dict[str, str] = Field(description="A dictionary with keys like 'X-Axis', 'Y-Axis', 'Color/Hue (Optional)', 'Facet (Optional)' and values that are exact column names from the dataset or None.")
    description: str = Field(description="The rationale and insight from 'Rationale and Insight'.")

def chart_preview_component(
    suggestions: List,
    df: pd.DataFrame,
    code_generator,
    df_context: Dict[str, str]
) -> List[Tuple[io.BytesIO, str]]:
    """Interactive chart preview with editing capabilities"""

    st.subheader("📊 Chart Suggestions & Preview")

    if not suggestions:
        st.warning("No chart suggestions available. Please run AI analysis first.")
        return []

    st.markdown(f"🎯 **{len(suggestions)} AI-generated chart suggestions ready for preview**")

    generated_charts = []

    for i, suggestion in enumerate(suggestions):
        # Handle both dict and object formats
        try:
            if isinstance(suggestion, dict):
                title = suggestion.get('title', f'Chart {i+1}')
                question = suggestion.get('question', 'No question specified')
                chart_type = suggestion.get('chart_type', 'histplot')
                description = suggestion.get('description', 'No description available')
                pre_processing_steps = suggestion.get('pre_processing_steps', 'None')
                column_mapping = suggestion.get('column_mapping', {})
            else:
                title = getattr(suggestion, 'title', f'Chart {i+1}')
                question = getattr(suggestion, 'question', 'No question specified')
                chart_type = getattr(suggestion, 'chart_type', 'histplot')
                description = getattr(suggestion, 'description', 'No description available')
                pre_processing_steps = getattr(suggestion, 'pre_processing_steps', 'None')
                column_mapping = getattr(suggestion, 'column_mapping', {})

        except Exception as e:
            st.error(f"Error processing suggestion {i+1}: {str(e)}")
            # Show debug info if available
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug - Suggestion {i+1} data:", suggestion)
                st.write(f"Debug - Suggestion type:", type(suggestion))
            continue

        # Add individual error boundary for each chart
        try:
            with st.expander(f"📈 Chart {i+1}: {title}", expanded=i == 0):

                # Display suggestion details
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**🎯 Question:** {question}")
                    st.markdown(f"**📊 Chart Type:** {chart_type}")
                    st.markdown(f"**💡 Description:** {description}")

                    if pre_processing_steps != "None":
                        st.markdown(f"**⚙️ Pre-processing:** {pre_processing_steps}")

                with col2:
                    st.markdown("**🗂️ Column Mapping:**")
                    for key, value in column_mapping.items():
                        if value:
                            st.markdown(f"• **{key}:** `{value}`")
                        else:
                            st.markdown(f"• **{key}:** _None_")

                # Chart generation controls
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    generate_btn = st.button(
                        f"🚀 Generate Chart {i+1}",
                        key=f"generate_{i}",
                        type="primary"
                    )

                with col2:
                    edit_btn = st.button(
                        f"✏️ Edit",
                        key=f"edit_{i}",
                        help="Edit chart parameters"
                    )

                with col3:
                    if f"chart_{i}_generated" in st.session_state:
                        st.success("✅ Generated")

                # Edit mode
                if edit_btn or f"edit_mode_{i}" in st.session_state:
                    st.session_state[f"edit_mode_{i}"] = True

                    with st.form(f"edit_form_{i}"):
                        st.markdown("**Edit Chart Parameters:**")

                        # Editable fields
                        new_title = st.text_input("Chart Title", value=title)
                        new_chart_type = st.selectbox(
                            "Chart Type",
                            ["histplot", "scatterplot", "boxplot", "violinplot", "barplot", "countplot", "lineplot", "heatmap"],
                            index=0
                        )

                        # Column mapping edits
                        available_columns = ["None"] + list(df.columns)

                        # Helper function to safely get index
                        def safe_index(value, options_list):
                            if value and value in options_list:
                                return options_list.index(value)
                            return 0  # Default to "None"

                        col1, col2 = st.columns(2)
                        with col1:
                            x_axis = st.selectbox("X-Axis", available_columns,
                                                index=safe_index(column_mapping.get("X-Axis"), available_columns))
                            y_axis = st.selectbox("Y-Axis", available_columns,
                                                index=safe_index(column_mapping.get("Y-Axis"), available_columns))

                        with col2:
                            hue = st.selectbox("Color/Hue", available_columns,
                                             index=safe_index(column_mapping.get("Color/Hue (Optional)"), available_columns))
                            facet = st.selectbox("Facet", available_columns,
                                               index=safe_index(column_mapping.get("Facet (Optional)"), available_columns))

                        # Submit button for the form
                        submitted = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")

                        if submitted:
                            # Update suggestion
                            if isinstance(suggestion, dict):
                                suggestion['title'] = new_title
                                suggestion['chart_type'] = new_chart_type
                                suggestion['column_mapping'] = {
                                    "X-Axis": x_axis if x_axis != "None" else None,
                                    "Y-Axis": y_axis if y_axis != "None" else None,
                                    "Color/Hue (Optional)": hue if hue != "None" else None,
                                    "Facet (Optional)": facet if facet != "None" else None
                                }
                            else:
                                suggestion.title = new_title
                                suggestion.chart_type = new_chart_type
                                suggestion.column_mapping = {
                                    "X-Axis": x_axis if x_axis != "None" else None,
                                    "Y-Axis": y_axis if y_axis != "None" else None,
                                    "Color/Hue (Optional)": hue if hue != "None" else None,
                                    "Facet (Optional)": facet if facet != "None" else None
                                }
                            st.success("✅ Changes saved!")
                            del st.session_state[f"edit_mode_{i}"]
                            st.rerun()

                # Generate chart
                if generate_btn:
                    with st.spinner(f"Generating chart {i+1}..."):
                        try:
                            # Generate code
                            generated_code = code_generator.generate_chart_code(suggestion, df_context)
                            cleaned_code = code_generator.clean_generated_code(generated_code)

                            # Execute code
                            img_buffer = io.BytesIO()
                            exec_scope = {
                                'df': df, 'pd': pd, 'plt': plt, 'sns': sns,
                                'np': __import__('numpy'), 'io': io, 'img_buffer': img_buffer
                            }

                            exec(cleaned_code, exec_scope)

                            # Display chart
                            img_buffer.seek(0)
                            if img_buffer.getbuffer().nbytes > 1000:
                                st.image(img_buffer, caption=title, use_container_width=True)
                                generated_charts.append((img_buffer, title))
                                st.session_state[f"chart_{i}_generated"] = True
                                st.success(f"✅ Chart {i+1} generated successfully!")
                            else:
                                st.error("❌ Chart generation failed - no output produced")

                        except Exception as e:
                            st.error(f"❌ Chart generation failed: {str(e)}")
                            if st.session_state.get('debug_mode', False):
                                st.exception(e)

        except Exception as e:
            st.error(f"❌ Error displaying chart {i+1}: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)

    return generated_charts

def display_chart_grid(chart_images: List[Tuple[io.BytesIO, str]]):
    """Display generated charts in a grid layout"""

    if not chart_images:
        st.info("No charts generated yet. Generate some charts in the preview section above.")
        return

    st.subheader("🎨 Generated Charts Gallery")

    # Display charts in a grid
    cols_per_row = 2
    for i in range(0, len(chart_images), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(chart_images):
                chart_buffer, title = chart_images[i + j]
                chart_buffer.seek(0)
                with col:
                    st.image(chart_buffer, caption=title, use_container_width=True)
