from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms, models
import io
from PIL import Image
import uvicorn
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Chest X-Ray Classification API",
              description="API for classifying chest X-ray images using Vision Transformer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def load_model():
    try:
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build path to model relative to the script
        model_path = os.path.join(script_dir, "..", "models", "VitFinal30_model.pth")
        print(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            # Try alternate paths
            alt_paths = [
                os.path.abspath("../models/VitFinal30_model.pth"),
                os.path.join(script_dir, "models", "VitFinal30_model.pth"),
                os.path.abspath("./models/VitFinal30_model.pth")
            ]
            
            for alt_path in alt_paths:
                print(f"Trying alternate path: {alt_path}")
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"Found model at: {model_path}")
                    break
            else:
                print("ERROR: Model file not found in any location")
                return None
        
        # Load the model architecture
        print("Creating model architecture...")
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        
        # Print model architecture info
        num_classes = len(class_names)
        print(f"Model heads structure: {model.heads}")
        
        # Access the correct in_features (using head layer inside Sequential)
        num_features = model.heads.head.in_features
        print(f"Head input features: {num_features}")
        
        # Replace classification head
        model.heads = torch.nn.Linear(num_features, num_classes)
        
        # Load weights with improved error handling
        print(f"Loading model weights from {model_path}")
        
        # Allow VisionTransformer class to be loaded
        from torchvision.models.vision_transformer import VisionTransformer
        torch.serialization.add_safe_globals([VisionTransformer])
        
        try:
            # Explicitly set weights_only to False
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            print(f"Model state dict loaded, keys: {list(state_dict.keys())[:5]}...")
            
            model.load_state_dict(state_dict)
            print("Standard loading succeeded")
        except Exception as e:
            print(f"Standard loading failed: {e}")
            try:
                # Try with weights_only=True as a backup
                print("Trying with weights_only=True...")
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                print("Non-strict loading succeeded")
            except Exception as e2:
                print(f"Non-strict loading also failed: {e2}")
                
                # Try one more approach with full model loading
                try:
                    print("Trying to load full model...")
                    loaded_model = torch.load(model_path, map_location=device, weights_only=False)
                    model = loaded_model
                    print("Full model loading succeeded")
                except Exception as e3:
                    print(f"All loading approaches failed: {e3}")
                    return None
            
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

model = load_model()

@app.get("/")
def home():
    return {"message": "Chest X-Ray Classification API is running. Use /predict endpoint to classify images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded")
    
    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Preprocess image
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            
            # Get all probabilities as a list
            probs = probabilities[0].cpu().numpy().tolist()
            
            result = {
                "prediction": class_names[predicted_class_idx],
                "confidence": float(probs[predicted_class_idx]),
                "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            }
            
            return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
