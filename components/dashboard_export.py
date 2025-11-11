import streamlit as st
import io
from typing import List, Tuple
import datetime
import zipfile
import os

def dashboard_export_component(
    chart_images: List[Tuple[io.BytesIO, str]],
    api_key: str = ""
) -> None:
    """Dashboard export with multiple format options and performance optimizations"""
    
    st.subheader("📄 Export & Download Dashboard")
    
    if not chart_images:
        st.warning("⚠️ No charts available for export. Please generate some charts first.")
        return
    
    st.markdown(f"📊 **{len(chart_images)} charts ready for export**")
    
    # Initialize session state for export settings
    if 'export_format_index' not in st.session_state:
        st.session_state.export_format_index = 0
    if 'include_metadata' not in st.session_state:
        st.session_state.include_metadata = True
    
    # Use form to prevent reruns when changing export settings
    with st.form("export_settings_form"):
        st.markdown("**Configure your export settings:**")
        
        # Move export format to the top as requested
        export_format = st.selectbox(
            "📋 Export Format",
            ["Individual Images (ZIP)", "Simple PDF", "Enhanced PDF (with AI Analysis)"],
            index=st.session_state.export_format_index,
            help="Choose how you want to export your dashboard"
        )
        
        include_metadata = st.checkbox(
            "📝 Include Metadata",
            value=st.session_state.include_metadata,
            help="Include chart titles and descriptions"
        )
        
        # Export button
        export_btn = st.form_submit_button(
            "🚀 Export Dashboard",
            type="primary",
            help="Generate and download your dashboard"
        )
    
    # Display current settings
    st.info(f"📋 **Current:** {['ZIP', 'Simple PDF', 'Enhanced PDF'][st.session_state.export_format_index]} format, Metadata: {'✅' if st.session_state.include_metadata else '❌'}")
    
    # Only export when form is submitted
    if export_btn:
        # Update session state with current settings
        st.session_state.export_format_index = ["Individual Images (ZIP)", "Simple PDF", "Enhanced PDF (with AI Analysis)"].index(export_format)
        st.session_state.include_metadata = include_metadata
        
        export_dashboard(chart_images, export_format, include_metadata, api_key)

def export_dashboard(
    chart_images: List[Tuple[io.BytesIO, str]], 
    export_format: str,
    include_metadata: bool,
    api_key: str
):
    """Handle dashboard export process"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with st.spinner("🔄 Preparing your dashboard for export..."):
        
        if export_format == "Individual Images (ZIP)":
            export_images_zip(chart_images, timestamp, include_metadata)
            
        elif export_format == "Simple PDF":
            export_simple_pdf(chart_images, timestamp, include_metadata)
            
        elif export_format == "Enhanced PDF (with AI Analysis)":
            if not api_key:
                st.error("❌ Google API key required for enhanced PDF export. Please configure it in the sidebar.")
                return
            export_enhanced_pdf(chart_images, timestamp, api_key)

def export_images_zip(
    chart_images: List[Tuple[io.BytesIO, str]], 
    timestamp: str,
    include_metadata: bool
):
    """Export individual chart images as ZIP file"""
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # Add each chart image
        for i, (img_buffer, title) in enumerate(chart_images):
            img_buffer.seek(0)
            
            # Clean filename
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"chart_{i+1:02d}_{clean_title[:30]}.png"
            
            zip_file.writestr(filename, img_buffer.read())
        
        # Add metadata file if requested
        if include_metadata:
            metadata = []
            for i, (_, title) in enumerate(chart_images):
                metadata.append(f"Chart {i+1}: {title}")
            
            metadata_content = "\n".join(metadata)
            zip_file.writestr("chart_metadata.txt", metadata_content.encode('utf-8'))
    
    zip_buffer.seek(0)
    
    # Offer download
    st.download_button(
        label="📥 Download Charts (ZIP)",
        data=zip_buffer.read(),
        file_name=f"dashboard_charts_{timestamp}.zip",
        mime="application/zip"
    )
    
    st.success("✅ Charts exported successfully!")

def export_simple_pdf(
    chart_images: List[Tuple[io.BytesIO, str]], 
    timestamp: str,
    include_metadata: bool
):
    """Export charts as simple PDF"""
    
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("AI-Generated Dashboard", styles['Title']))
        story.append(Spacer(1, 0.5 * inch))
        
        # Add each chart
        for i, (img_buffer, title) in enumerate(chart_images):
            if include_metadata:
                story.append(Paragraph(f"Chart {i+1}: {title}", styles['Heading2']))
            
            img_buffer.seek(0)
            img = Image(img_buffer, width=6*inch, height=3.75*inch, kind='proportional')
            story.append(img)
            story.append(Spacer(1, 0.3 * inch))
        
        doc.build(story)
        pdf_buffer.seek(0)
        
        # Offer download
        st.download_button(
            label="📥 Download PDF",
            data=pdf_buffer.read(),
            file_name=f"dashboard_{timestamp}.pdf",
            mime="application/pdf"
        )
        
        st.success("✅ PDF exported successfully!")
        
    except ImportError:
        st.error("❌ ReportLab not available. Please install it to export PDF.")
    except Exception as e:
        st.error(f"❌ PDF export failed: {str(e)}")

def export_enhanced_pdf(
    chart_images: List[Tuple[io.BytesIO, str]], 
    timestamp: str,
    api_key: str
):
    """Export enhanced PDF with AI analysis"""
    
    try:
        from utils.pdf_utils import StreamlitPDFGenerator, PDFAnalyzer
        import tempfile
        import os
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Create a simple PDF first
        status_text.text("📄 Step 1/3: Creating initial PDF...")
        progress_bar.progress(20)
        
        pdf_generator = StreamlitPDFGenerator()
        
        # Create temporary file for the simple PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf_path = temp_pdf.name
        
        # Extract just the image buffers for PDF generation
        image_buffers = [img_buffer for img_buffer, _ in chart_images]
        
        # Create simple PDF first
        simple_filename = f"temp_dashboard_{timestamp}.pdf"
        pdf_generator.generate_enhanced_pdf(
            image_buffers, 
            # Create a dummy analysis for now - we'll analyze the actual PDF
            type('MockAnalysis', (), {
                'overall_trend_summary': 'Initial analysis pending...',
                'charts': []
            })(),
            simple_filename
        )
        
        # Step 2: Analyze the PDF with AI
        status_text.text("🤖 Step 2/3: Analyzing PDF with AI...")
        progress_bar.progress(60)
        
        analyzer = PDFAnalyzer(api_key)
        pdf_analysis = analyzer.analyze_pdf_with_genai(simple_filename)
        
        # Step 3: Generate enhanced PDF with analysis
        status_text.text("✨ Step 3/3: Creating enhanced PDF...")
        progress_bar.progress(90)
        
        enhanced_filename = f"enhanced_dashboard_{timestamp}.pdf"
        final_pdf = pdf_generator.generate_enhanced_pdf(
            image_buffers, 
            pdf_analysis, 
            enhanced_filename
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Enhanced PDF generated successfully!")
        
        # Read the final PDF for download
        with open(enhanced_filename, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        # Offer download
        st.download_button(
            label="📥 Download Enhanced PDF",
            data=pdf_data,
            file_name=f"enhanced_dashboard_{timestamp}.pdf",
            mime="application/pdf"
        )
        
        # Show analysis summary
        st.success("✅ Enhanced PDF with AI analysis generated successfully!")
        
        with st.expander("🤖 AI Analysis Summary", expanded=True):
            st.markdown(f"**📊 Total Charts Analyzed:** {pdf_analysis.total_charts}")
            st.markdown(f"**💡 Overall Summary:** {pdf_analysis.overall_trend_summary}")
            
            if pdf_analysis.charts:
                st.markdown("**📈 Chart Details:**")
                for i, chart in enumerate(pdf_analysis.charts[:3]):  # Show first 3
                    st.markdown(f"- **Chart {chart.chart_number}:** {chart.chart_title} ({chart.chart_type})")
                
                if len(pdf_analysis.charts) > 3:
                    st.markdown(f"... and {len(pdf_analysis.charts) - 3} more charts")
        
        # Clean up temporary files
        try:
            os.unlink(simple_filename)
            os.unlink(enhanced_filename)
        except:
            pass
        
    except Exception as e:
        st.error(f"❌ Enhanced PDF export failed: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)

def display_export_status(chart_count: int):
    """Display export status - cleaned up version without quick export buttons"""
    
    if chart_count == 0:
        st.info("📊 Generate some charts first to enable export options.")
    else:
        st.success(f"✅ {chart_count} charts ready for export!")
        
        # Only keep the reset charts option
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:  # Center the reset button
            if st.button("🔄 Reset Charts", help="Clear all generated charts"):
                if 'generated_charts' in st.session_state:
                    st.session_state.generated_charts = []
                    st.success("Charts cleared!")
                    st.rerun()
