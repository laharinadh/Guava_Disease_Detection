import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Guava Disease Detector",
    layout="wide",
    page_icon=":guava:"
)

# Load models
@st.cache_resource
def load_models():
    yolo = YOLO('C:\\Users\\rlaha\\OneDrive\\Desktop\\Guava_de\\models\\best.pt')  # Part detection model
    densenet = tf.keras.models.load_model('C:\\Users\\rlaha\\OneDrive\\Desktop\\Guava_de\\models\\densenet201_guava_epoch_10.h5')  # Disease classifier
    return yolo, densenet

yolo_model, densenet_model = load_models()

# Class mappings
YOLO_CLASSES = ["Guava Fruit", "Guava Leaf", "Guava Stem"]
DENSENET_CLASSES = [
    "Stem Canker", "Fruit Phytopthora", "Fruit Scab", "Fruit Styler End Root",
    "Fruit Healthy", "Leaf Anthracnose", "Leaf Canker", "Leaf Mummification",
    "Leaf Red Rust", "Leaf Rust", "Leaf Healthy", "Stem Healthy", "Stem Wilt"
]

# Treatment recommendations
TREATMENTS = {
    "Stem Canker": "1. Remove infected branches\n2. Apply copper-based fungicide\n3. Improve air circulation",
    "Fruit Phytopthora": "1. Avoid waterlogging\n2. Apply potassium phosphonate\n3. Remove infected fruits",
    "Fruit Scab": "1. Prune affected areas\n2. Use sulfur-based spray\n3. Maintain proper spacing",
    "Fruit Styler End Root": "1. Improve drainage\n2. Reduce nitrogen fertilizer\n3. Apply biocontrol agents",
    "Fruit Healthy": "No treatment needed. Maintain regular care.",
    "Leaf Anthracnose": "1. Remove infected leaves\n2. Apply neem oil\n3. Ensure proper sunlight",
    "Leaf Canker": "1. Prune infected leaves\n2. Apply Bordeaux mixture\n3. Avoid overhead watering",
    "Leaf Mummification": "1. Remove mummified fruits\n2. Apply fungicide\n3. Maintain orchard hygiene",
    "Leaf Red Rust": "1. Apply fungicide\n2. Increase potassium fertilization\n3. Remove alternate hosts",
    "Leaf Rust": "1. Use resistant varieties\n2. Apply triazole fungicides\n3. Improve air circulation",
    "Leaf Healthy": "No treatment needed. Maintain regular care.",
    "Stem Healthy": "No treatment needed. Maintain regular care.",
    "Stem Wilt": "1. Remove severely infected plants\n2. Apply soil drench fungicides\n3. Solarize soil before planting"
}

# Image enhancement functions
def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = image.copy()
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
    return img

# UI setup
st.title("Guava Plant Disease Diagnosis & Treatment 🍐")
st.sidebar.header("Analysis Settings")

confidence_threshold = st.sidebar.slider(
    "Detection Threshold", 
    0.0, 1.0, 
    0.5,
    help="Minimum confidence for object detection"
)

# Image enhancement options
with st.sidebar.expander("Image Enhancement"):
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
    enhance = st.checkbox("Apply Enhancements", value=False)

uploaded_files = st.sidebar.file_uploader(
    "Upload guava plant images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Processing pipeline
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read and display original image
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.sidebar.image(image, caption="Original Image", width=256)
        except Exception as e:
            st.error(f"File error: {str(e)}")
            continue
        
        # Apply enhancements if enabled
        if enhance:
            try:
                enhanced_image = enhance_image(
                    image,
                    brightness=brightness,
                    contrast=contrast,
                    sharpness=sharpness
                )
                img_array = np.array(enhanced_image)
                st.sidebar.image(enhanced_image, caption="Enhanced Image", width=256)
            except Exception as e:
                st.warning(f"Enhancement error: {str(e)}")
        
        # YOLOv8 part detection
        try:
            results = yolo_model.predict(
                source=img_array,
                conf=confidence_threshold,
                save=False,
                imgsz=640,
                verbose=False
            )
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            continue
        
        # Check for valid detections
        if not results[0].boxes:
            st.warning("No plant parts detected")
            continue
            
        # Process best detection
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        best_idx = np.argmax(confidences)
        best_conf = confidences[best_idx]
        best_box = boxes[best_idx]
        part_class = YOLO_CLASSES[class_ids[best_idx]]
        
        # Crop detected region
        x1, y1, x2, y2 = map(int, best_box)
        cropped_img = img_array[y1:y2, x1:x2]
        
        # Disease classification with DenseNet
        try:
            # Preprocess for DenseNet
            resized = cv2.resize(cropped_img, (224, 224)) / 255.0
            batch = np.expand_dims(resized, axis=0)
            
            # Predict
            predictions = densenet_model.predict(batch)
            disease_class_id = np.argmax(predictions)
            confidence = np.max(predictions)
            disease_name = DENSENET_CLASSES[disease_class_id]
            
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            continue
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detected Region")
            st.image(
                cropped_img,
                use_container_width=True
            )
            
        with col2:
            st.subheader("Diagnostic Report")
            st.metric("Plant Part", part_class)
            st.metric("Disease", disease_name)
            st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Health status styling
            if "Healthy" in disease_name:
                st.success(f"✓ {part_class} appears healthy")
            else:
                st.error(f"⚠️ {disease_name} detected")
                
            # Treatment recommendations
            st.subheader("Treatment Recommendations")
            st.markdown(f"**For {disease_name}:**")
            st.markdown(TREATMENTS.get(disease_name, "Consult agricultural expert"))
                
            st.markdown(
                f"**Diagnosis:** {part_class} shows symptoms of "
                f"{disease_name.lower()} with {confidence*100:.1f}% confidence"
            )

else:
    st.info("Upload guava plant images for disease diagnosis")