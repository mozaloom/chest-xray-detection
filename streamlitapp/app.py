import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import io
import matplotlib.pyplot as plt
import time
import traceback

# Set page configuration
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide"
)

# App title and description
st.title("Chest X-Ray Classification")
st.write("""
This application analyzes chest X-ray images to detect abnormalities including COVID-19, 
bacterial pneumonia, and viral pneumonia using a Vision Transformer deep learning model.
""")

# Define class names
class_names = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image preprocessing
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Load model
@st.cache_resource
def load_model():
    try:
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try different possible model paths
        possible_paths = [
            os.path.join(script_dir, "models", "VitFinal30_model.pth"),
            os.path.join(script_dir, "..", "models", "VitFinal30_model.pth"),
            os.path.abspath("./models/VitFinal30_model.pth"),
            os.path.abspath("../models/VitFinal30_model.pth")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                st.sidebar.success(f"Model found at: {model_path}")
                break
        
        if model_path is None:
            st.error("Model file not found. Please place the model file in a 'models' directory.")
            st.info("Expected model file: VitFinal30_model.pth")
            return None
        
        # Allow VisionTransformer class to be loaded
        from torchvision.models.vision_transformer import VisionTransformer
        torch.serialization.add_safe_globals([VisionTransformer])
        
        # Try loading the saved model directly first
        try:
            st.info("Attempting to load full model directly...")
            with torch.serialization.safe_globals([VisionTransformer]):
                loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            
            # Check if loaded_model is already a model instance
            if isinstance(loaded_model, torch.nn.Module):
                st.success("Full model loaded successfully!")
                loaded_model.to(device)
                loaded_model.eval()
                return loaded_model
            else:
                st.info("Loaded file is not a full model, trying as state dict...")
        except Exception as e:
            st.warning(f"Could not load as full model: {e}")
        
        # Initialize a new model with the correct architecture
        st.info("Creating model architecture...")
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        
        # Replace classification head
        num_classes = len(class_names)
        num_features = model.heads.head.in_features
        model.heads = torch.nn.Linear(num_features, num_classes)
        
        # Try loading as state dict
        try:
            st.info("Attempting to load model weights as state dict...")
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            
            # Check if it's a dict before trying to load
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict, strict=False)
                st.success("Model weights loaded successfully!")
            else:
                st.error("Model file is not a state dictionary or a full model.")
                return None
                
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            st.code(traceback.format_exc(), language="python")
            return None
            
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.code(traceback.format_exc(), language="python")
        return None

# Function to make prediction
def predict(image, model):
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
        return None
    
    try:
        # Preprocess image
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Add diagnostic info
        st.sidebar.info(f"Input tensor shape: {image_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            try:
                outputs = model(image_tensor)
                st.sidebar.info(f"Output tensor shape: {outputs.shape}")
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                
                # Get all probabilities as a list
                probs = probabilities[0].cpu().numpy().tolist()
                
                result = {
                    "prediction": class_names[predicted_class_idx],
                    "confidence": float(probs[predicted_class_idx]),
                    "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                }
                
                return result
            except Exception as model_error:
                st.error(f"Error during model inference: {model_error}")
                st.code(traceback.format_exc(), language="python")
                return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.code(traceback.format_exc(), language="python")
        return None

# Sidebar for model info and status
with st.sidebar:
    st.header("Model Information")
    st.write("Model: Vision Transformer (ViT-B/16)")
    st.write(f"Running on: {device}")
    st.write("Classes:")
    for cls in class_names:
        st.write(f"- {cls}")
    
    st.header("About")
    st.write("""
    This application uses a fine-tuned Vision Transformer model to classify chest X-ray images.
    The model can detect COVID-19, normal conditions, bacterial pneumonia, and viral pneumonia.
    """)

# Create two columns for the main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload X-ray Image")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
        
        # Button to perform prediction
        if st.button("Analyze Image"):
            with st.spinner('Processing...'):
                # Add a slight delay to show the spinner
                time.sleep(1)
                
                # Load model
                model = load_model()
                
                # Make prediction
                if model is None:
                    st.error("Model could not be loaded. Please check that the model file exists and is accessible.")
                    with st.expander("Troubleshooting Tips"):
                        st.write("""
                        1. Make sure the model file 'VitFinal30_model.pth' is in the 'models' directory
                        2. Check file permissions on the model file
                        3. Verify that you have the correct version of PyTorch and torchvision installed
                        """)
                else:
                    result = predict(image, model)
                    
                    if result:
                        # Store result for display in the next column
                        st.session_state.result = result
                        st.success("Analysis complete!")
                    else:
                        st.error("Failed to analyze the image. Please try again.")

with col2:
    st.header("Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state.result
        
        # Display the prediction with color coding
        prediction = result["prediction"]
        confidence = result["confidence"] * 100
        
        # Choose color based on prediction
        if prediction == "Normal":
            pred_color = "green"
        elif prediction == "COVID-19":
            pred_color = "red"
        else:
            pred_color = "orange"
        
        st.markdown(f"### Diagnosis: <span style='color:{pred_color}'>{prediction}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Display probabilities as a bar chart
        st.subheader("Probability Distribution")
        probs = result["probabilities"]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create bar chart
        classes = list(probs.keys())
        values = list(probs.values())
        colors = ['red', 'green', 'orange', 'purple']
        
        bars = ax.bar(classes, [v * 100 for v in values], color=colors)
        
        # Add percentage labels above bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val*100:.1f}%', ha='center', va='bottom')
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Classification Probabilities')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Add interpretation
        st.subheader("Interpretation")
        if prediction == "Normal":
            st.write("The X-ray appears normal with no significant abnormalities detected.")
        elif prediction == "COVID-19":
            st.write("The X-ray shows patterns consistent with COVID-19 infection. Common findings include ground-glass opacities and consolidation.")
        elif prediction == "Pneumonia-Bacterial":
            st.write("The X-ray shows patterns consistent with bacterial pneumonia, which typically presents with lobar consolidation.")
        elif prediction == "Pneumonia-Viral":
            st.write("The X-ray shows patterns consistent with viral pneumonia, which typically presents with interstitial patterns and diffuse haziness.")
        
        st.write("**Note:** This is an AI-assisted analysis and should not replace professional medical diagnosis.")
    else:
        st.info("Upload and analyze an image to see results here.")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.*")