import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline

# Page config
st.set_page_config(
    page_title="DeepShield - Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 DeepShield")
st.subheader("AI-powered Deepfake Detection System")
st.divider()

# Upload section
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
)

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    # Analyze button
    if st.button("🔬 Analyze", use_container_width=True):
        with st.spinner("Analyzing..."):
            
            if file_type == 'image':
                # Image process karo
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                pipeline = ImagePipeline()
                result = pipeline.run(image_np)
            
            else:
                # Video process karo
                import tempfile
                with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    pipeline = VideoPipeline()
                    result = pipeline.run(tmp.name)
        
        # Results dikhao
        if 'error' in result:
            st.error(result['error'])
        else:
            st.divider()
            
            # Verdict
            if result['label'] == 'FAKE':
                st.error(f"⚠ FAKE DETECTED")
            else:
                st.success(f"✓ REAL")
            
            # Confidence
            st.metric("Confidence", f"{result['confidence']}%")
            st.progress(result['confidence'] / 100)
            
            # Heatmap
            if result['heatmap'] is not None:
                st.subheader("Manipulation Heatmap")
                st.image(result['heatmap'], caption="Red = Manipulated area")
            
            # Extra info for video
            if 'frames_analyzed' in result:
                st.info(f"Frames analyzed: {result['frames_analyzed']}")