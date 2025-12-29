import streamlit as st
import numpy as np
import plotly.graph_objects as go

import tempfile
import os
import time
import trimesh
import matplotlib.pyplot as plt
from skimage import measure
import io
from streamlit_option_menu import option_menu as om
from backend import Feature_Predictor

# Page configuration
st.set_page_config(
    page_title="CAM Feature Recognition",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dashboard - White and Dark Blue theme
# Custom CSS for professional dashboard - White and Dark Blue theme
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #FFFFFF;
        color: #1F2937;
    }
    
    /* Header styling - professional and clean */
    .main-header {
        font-size: 2.2rem;
        color: #2563EB;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1.2rem;
        background: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
        border: 1px solid #E5E7EB;
    }
    
    /* Tab container styling */
    .tab-container {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 15px;
    }
    
    /* Tab button styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #FFFFFF;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F8FAFC;
        color: #6B7280;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        border: 1px solid #E5E7EB;
        margin-right: 2px;
        font-size: 14px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 500;
        border: 1px solid #2563EB;
    }
    
    /* Upload section styling */
    .upload-section {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 8px;
        border: 2px dashed #D1D5DB;
        margin-bottom: 20px;
        text-align: center;
        color: #6B7280;
    }
    
    .upload-section:hover {
        border-color: #2563EB;
        background-color: #F0F5FF;
    }
    
    /* File info styling */
    .file-info {
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #2563EB;
        margin: 12px 0;
        color: #4B5563;
        font-size: 14px;
        border: 1px solid #E5E7EB;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    /* Progress container */
    .progress-container {
        background-color: #F8FAFC;
        padding: 16px;
        border-radius: 8px;
        border-left: 3px solid #2563EB;
        margin: 16px 0;
        color: #4B5563;
        border: 1px solid #E5E7EB;
    }
    
    /* Status badge */
    .status-badge {
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 6px 0;
    }
    
    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 20px;
        border: 1px solid #2563EB;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        color: #1F2937;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Model selection - Transparent */
    .model-selection {
        background-color: transparent;
        padding: 16px 0;
        color: #1F2937;
        font-size: 14px;
        border: none;
        box-shadow: none;
    }
    
    /* Radio button styling - Clean and minimal */
    .stRadio [role="radiogroup"] {
        background-color: transparent;
        padding: 8px 0;
        border: none;
        border-radius: 0;
        gap: 8px;
    }
    
    .stRadio [data-baseweb="radio"] {
        margin: 0;
        padding: 0;
    }
    
    .stRadio label {
        color: #1F2937 !important;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
        background-color: transparent;
        border: 2px solid #E5E7EB;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        min-height: 44px;
    }
    
    .stRadio label:hover {
        background-color: #F0F5FF;
        color: #2563EB !important;
        border-color: #2563EB;
    }
    
    .stRadio [data-baseweb="radio"][aria-checked="true"] + span {
        color: #2563EB !important;
        font-weight: 600;
    }
    
    /* Selected radio button styling */
    .stRadio [data-baseweb="radio"][aria-checked="true"] + span {
        background-color: #F0F5FF;
        border-color: #2563EB;
    }
    
    .stRadio [data-baseweb="radio"] div:first-child {
        background-color: #FFFFFF;
        border-color: #D1D5DB;
        border-width: 2px;
    }
    
    .stRadio [data-baseweb="radio"][aria-checked="true"] div:first-child {
        background-color: #2563EB;
        border-color: #2563EB;
        box-shadow: 0 0 0 3px #F0F5FF;
    }
    
    /* Radio button inner dot */
    .stRadio [data-baseweb="radio"][aria-checked="true"] div:first-child div {
        background-color: #FFFFFF;
        transform: scale(0.6);
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background-color: #E5E7EB;
    }
    
    .stSlider [data-baseweb="slider"] div:first-child {
        background-color: #2563EB;
    }
    
    .stSlider [data-baseweb="slider"] div:nth-child(2) {
        background-color: #93C5FD;
    }
    
    /* Slider thumb */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    
    /* Slider value display */
    .stSlider [data-baseweb="slider"] + div {
        color: #6B7280;
    }
    
    /* Checkbox styling */
    .stCheckbox [data-baseweb="checkbox"] {
        background-color: #FFFFFF;
        border-color: #D1D5DB;
    }
    
    .stCheckbox [data-baseweb="checkbox"]:checked {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F4F6;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9CA3AF;
    }
    
    /* Make sure all text is visible */
    .stMarkdown, .stText, .stTitle, .stHeader {
        color: #1F2937 !important;
    }
    
    /* Section headers */
    .stSubheader {
        color: #2563EB !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Confidence threshold value display */
    .stSlider [data-baseweb="slider"] + div > div {
        color: #2563EB !important;
        font-weight: 500;
    }
    
    /* Expandable section styling */
    .streamlit-expanderHeader {
        background-color: transparent;
        border: 2px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px 16px;
        color: #1F2937;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #2563EB;
        background-color: #F0F5FF;
    }
    
    .streamlit-expanderContent {
        background-color: transparent;
        border: none;
        padding: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

#
model_path = '.'
# Your voxelization functions (same as before)
def load_and_preprocess_stl(stl_path):
    """Load STL file and preprocess it for voxelization"""
    mesh = trimesh.load(stl_path)
    if not mesh.is_watertight:
        st.warning("Mesh is not watertight, attempting repair...")
        mesh.fill_holes()
        mesh.fix_normals()
    mesh.apply_translation(-mesh.bounds[0])
    mesh.apply_scale(1 / np.max(mesh.extents))
    return mesh

def voxelize_mesh(mesh, resolution=100):
    """High-quality voxelization using trimesh"""
    voxels = mesh.voxelized(pitch=1/resolution)
    voxel_matrix = voxels.matrix
    return voxels, voxel_matrix

def visualize_voxels_matplotlib(voxel_matrix, downsample_factor=2):
    """Visualize voxels using matplotlib and return the figure"""
    if downsample_factor > 1:
        voxel_matrix = voxel_matrix[::downsample_factor,
                                  ::downsample_factor,
                                  ::downsample_factor]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
    ax.voxels(voxel_matrix, edgecolor='k', alpha=0.5, facecolors='#2563EB')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Voxel Grid Visualization", fontsize=14, color='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(colors='black')
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    return fig

def save_voxel_data(voxel_matrix, output_path):
    """Save voxel data to numpy file"""
    np.save(output_path, voxel_matrix.astype(bool))

def full_pipeline(stl_path, resolution=100, store_path='.'):
    """Complete voxelization pipeline"""
    mesh = load_and_preprocess_stl(stl_path)
    voxels, voxel_matrix = voxelize_mesh(mesh, resolution)
    save_voxel_data(voxel_matrix, store_path)
    return voxels, voxel_matrix

def plot_voxel_grid_plotly(voxel_data, title="Voxel Grid"):
    """Create 3D visualization of voxel grid using Plotly"""
    sample_rate = max(1, voxel_data.shape[0] // 32)
    x, y, z = np.where(voxel_data[::sample_rate, ::sample_rate, ::sample_rate] > 0)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=3, color='#2563EB', opacity=0.8, symbol='square')
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            bgcolor='#FFFFFF',
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(backgroundcolor="#FFFFFF", color='black'),
            yaxis=dict(backgroundcolor="#FFFFFF", color='black'),
            zaxis=dict(backgroundcolor="#FFFFFF", color='black')
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='black'),
        width=800,
        height=600
    )
    return fig

def plot_stl_mesh(stl_mesh, title="STL Model"):
    """Create 3D visualization of STL mesh"""
    fig = go.Figure(data=[
        go.Mesh3d(
            x=stl_mesh.vertices[:, 0],
            y=stl_mesh.vertices[:, 1],
            z=stl_mesh.vertices[:, 2],
            i=stl_mesh.faces[:, 0],
            j=stl_mesh.faces[:, 1],
            k=stl_mesh.faces[:, 2],
            opacity=0.9,
            color='#2563EB',
            flatshading=True
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            bgcolor='#FFFFFF',
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(backgroundcolor="#FFFFFF", color='black'),
            yaxis=dict(backgroundcolor="#FFFFFF", color='black'),
            zaxis=dict(backgroundcolor="#FFFFFF", color='black')
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='black'),
        width=800,
        height=600
    )
    return fig

# Header
st.markdown('<h1 class="main-header">🧊 CAM Feature Recognition Dashboard</h1>', unsafe_allow_html=True)

# Sidebar - Model Selection
with st.sidebar:
    st.markdown("### 🎯 Model Selection")
    with st.expander("Select Model Architecture", expanded=True):
        selected_model = om(
            "Choose a model:",
            ["ResNet", "InceptionLite", "FeatureNet CNN"],
            menu_icon='robot',
            #help="Select the model architecture for feature recognition",
            orientation= 'vertical'
        )
                # selected_model
        if selected_model == 'ResNet':
            current_model_name = 'ResNet_3D_mod_1'
            label_map_filename = f"{current_model_name}_label_map.json"
            model_type = 2
        elif selected_model == 'InceptionLite':
            current_model_name = 'LiteInceptionNet3D_2'
            label_map_filename = f"{current_model_name}_label_map.json"
            model_type = 1
        elif selected_model == 'FeatureNet CNN':
            current_model_name = 'Featurenet_2'
            label_map_filename = f"{current_model_name}_label_map.json"
            model_type = 3
        else:
            st.warning(f"Unknown selected_model '{selected_model}'. Using default values.")
            current_model_name = None
            label_map_filename = None

        # Update predictor in session state when model selection changes
        if current_model_name:
            label_map_path = os.path.join(model_path, label_map_filename) if current_model_name else None
            st.session_state.predictor = Feature_Predictor(model_path, current_model_name, label_map_path, model_type)
            
    st.markdown("---")
    st.markdown(f"""
    <div class="model-selection">
        <strong>Selected Model:</strong><br>
        <span style='color: #2563EB; font-weight: bold;'>{selected_model}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # st.markdown("---")
    # st.info("""
    # **Model Capabilities:**
    # - 3D feature recognition
    # - Real-time prediction
    # - High accuracy
    # - Professional grade analysis
    # """)

# Create tabs with dashboard style
tab1, tab2 = st.tabs(["📦 Voxelization Engine", "🔮 Prediction Engine"])

# Voxelization Engine Tab
with tab1:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔄 STL File Voxelization")
        
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload STL File", 
            type=['stl'],
            key="voxelize_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.markdown(f'''
            <div class="file-info">
                <strong>📄 File Name:</strong> {uploaded_file.name}<br>
                <strong>📊 File Size:</strong> {file_size:.2f} KB<br>
                <strong>📋 File Type:</strong> STL
            </div>
            ''', unsafe_allow_html=True)
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                stl_mesh = trimesh.load(tmp_path)
                st.subheader("📐 Original STL Model")
                stl_fig = plot_stl_mesh(stl_mesh)
                st.plotly_chart(stl_fig, use_container_width=True)
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing STL file: {str(e)}")
            
            if st.button("🚀 Start Voxelization", key="voxelize_btn"):
                with st.spinner("Initializing voxelization process..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_stl_path = tmp_file.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_output:
                            tmp_npy_path = tmp_output.name
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        steps = [
                            "Loading and preprocessing STL...",
                            "Voxelizing mesh...",
                            "Generating voxel grid...",
                            "Saving results..."
                        ]
                        
                        for i, step in enumerate(steps):
                            status_text.markdown(f'<div class="status-badge">Step {i+1}/{len(steps)}</div> {step}', unsafe_allow_html=True)
                            time.sleep(0.5)
                            progress_bar.progress((i + 1) / len(steps))
                        
                        voxels, voxel_matrix = full_pipeline(tmp_stl_path, resolution=128, store_path=tmp_npy_path)
                        
                        status_text.markdown('<div class="status-badge">✅ Complete</div> Voxelization successful!', unsafe_allow_html=True)
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.subheader("🎯 Voxelized Result")
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            st.markdown("**🔄 Interactive 3D View**")
                            voxel_fig = plot_voxel_grid_plotly(voxel_matrix, "Voxel Grid")
                            st.plotly_chart(voxel_fig, use_container_width=True)
                        
                        with col_viz2:
                            st.markdown("**📊 Detailed View**")
                            matplotlib_fig = visualize_voxels_matplotlib(voxel_matrix)
                            st.pyplot(matplotlib_fig)
                        
                        with open(tmp_npy_path, 'rb') as f:
                            npy_data = f.read()
                        
                        st.download_button(
                            label="📥 Download NPY File",
                            data=npy_data,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_voxelized.npy",
                            mime="application/octet-stream"
                        )
                        
                        os.unlink(tmp_stl_path)
                        os.unlink(tmp_npy_path)
                        
                    except Exception as e:
                        st.error(f"Error during voxelization: {str(e)}")
    
    with col2:
        # Voxelization settings removed as requested
        st.markdown("---")
        st.info("""
        **Voxelization Process:**
        - Converts 3D mesh to volumetric grid
        - Preserves geometric features
        - Optimized for deep learning
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Engine Tab
with tab2:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔮 Feature Prediction")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload NPY or STL File", 
            type=['npy', 'stl'],
            key="predict_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / 1024
            file_ext = uploaded_file.name.split('.')[-1].upper()
            st.markdown(f'''
            <div class="file-info">
                <strong>📄 File Name:</strong> {uploaded_file.name}<br>
                <strong>📊 File Size:</strong> {file_size:.2f} KB<br>
                <strong>📋 File Type:</strong> {file_ext}
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button("🔍 Predict Features", key="predict_btn"):
                with st.spinner("Analyzing features..."):
                    try:
                        # Handle file processing
                        if file_ext == 'STL':
                            # Process STL to NPY (dummy implementation - replace with your actual conversion)
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_stl:
                                tmp_stl.write(uploaded_file.getvalue())
                                tmp_stl_path = tmp_stl.name
                                mesh = load_and_preprocess_stl(tmp_stl_path)
                            
                            #  STL to NPY conversion logic here
                            # This is a placeholder - replace with your actual conversion
                            _, np_sample_data = voxelize_mesh(mesh, resolution=128)  # Dummy data
                            
                        else:  # NPY file
                            np_sample_data = np.load(io.BytesIO(uploaded_file.getvalue()))
                        
                        # Perform prediction using session state model
                        if 'predictor' in st.session_state:
                            top1_str, filtered_probs_dict, top1_np = st.session_state.predictor.predict(np_sample_data, threshold=0.01)
                            
                            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                            st.markdown(f"**Predicted Feature:** {top1_str}")
                            st.markdown(f"**Confidence:** {filtered_probs_dict.get(top1_str, 0)*100:.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.subheader("📈 Confidence Scores")
                            for feature, prob in filtered_probs_dict.items():
                                st.progress(float(prob), text=f"{feature}: {prob*100:.2f}%")
                        else:
                            st.error("Model not loaded. Please select a model first.")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        st.markdown("---")
        st.info(f"""
        **Selected Model:** {selected_model}
        
        **Status:** {'Ready for prediction' if 'predictor' in st.session_state else 'Model not loaded'}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #2563EB;'>"
    "🧊 CAM Feature Recognition Dashboard | Professional 3D Analysis System"
    "</div>",
    unsafe_allow_html=True
)