# PulmoVision: Chest X-Ray AI Classifier

PulmoVision is a deep learning system for automated classification of chest X-ray images. The system can detect and differentiate between COVID-19, bacterial pneumonia, viral pneumonia, and normal conditions using a Vision Transformer (ViT) neural network architecture.

## Features

- Classification of chest X-rays into four categories: COVID-19, Normal, Bacterial Pneumonia, and Viral Pneumonia
- Web interface built with Streamlit for easy interaction with the model
- API endpoint built with FastAPI for integration into other applications
- Docker support for simplified deployment
- Detailed visualization of classification results with confidence scores

## Project Structure

```
.
├── Dockerfile          # Docker configuration for containerization
├── LICENSE             # Project license information
├── Makefile            # Automation for common tasks
├── README.md           # This file
├── models/             # Directory containing trained models
├── notebooks/          # Jupyter notebooks for development and training
├── requirements.txt    # Python dependencies
├── streamlitapp/       # Streamlit web application
└── webapp/             # FastAPI backend service
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Access to the trained model file (VitFinal30_model.pth)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/mozaloom/pulmo-vision.git
   cd PulmoVision
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Streamlit Application

The Streamlit app provides a user-friendly interface for uploading and analyzing chest X-ray images:

```
cd streamlitapp
streamlit run app.py
```

Then access the application at http://localhost:8501

### FastAPI Service

The FastAPI service provides a REST API for programmatic access to the model:

```
cd webapp
uvicorn app:app --reload
```

Then access the API documentation at http://localhost:8000/docs

### Docker Deployment

Build and run the Docker container:

```
docker build -t pulmovision .
docker run -p 8000:8000 -p 8501:8501 pulmovision
```

## Model Information

- Architecture: Vision Transformer (ViT-B/16)
- Input: Chest X-ray images (resized to 224x224)
- Output: Classification probabilities for four classes
- Training notebooks available in the `notebooks` directory

## Development

You can extend the model or application using the included notebooks:
- `vision-transformer.ipynb`: Implementation and training of the ViT model
- `cleancode.ipynb`: Data preprocessing and exploration

## License

See the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```
@software{PulmoVision2025,
  author = {Mohammed Zaloom},
  title = {PulmoVision: Chest X-Ray AI Classifier},
  year = {2025},
  url = {https://github.com/mozaloom/pulmo-vision}
}
```