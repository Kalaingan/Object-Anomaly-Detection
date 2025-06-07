# Object-Anomaly-Detection
This application leverages the DINO-ViT-B16 model from Hugging Face, fine-tuned on the MVTec Anomaly Detection dataset, to identify anomalies in industrial objects.
## Features

- Detects anomalies in industrial objects
- Locates and highlights defects within images
- Utilizes the DINO-ViT-B16 model with KMeans clustering
- Supports GPU acceleration for faster processing
- Allows model export for offline deployment
- Provides an interactive web interface via Streamlit

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train and evaluate the model and run:

```
python app.py
```

This will:
1. Load the DINO-ViT-B16 model
2. Collect and preprocess images from the dataset
3. Extract features and train KMeans clustering
4. Fine-tune the model for anomaly detection
5. Evaluate the model on test data
6. Save the trained model for future use

### Launching the Web Interface

Once the model is trained, you can use the Streamlit web interface:

```
streamlit run streamlit_app.py
```

This will open a web interface where you can:
1. Upload images for anomaly detection
2. View detection results and defect localization
3. Explore anomaly heatmaps for identified defects

## Model Details

- Base model: DINO-ViT-B16 from Hugging Face
- Image preprocessing: Resize to 224x224, normalize, convert to tensor
- Feature extraction: Uses CLS token from the DINO model
- Clustering: KMeans with 2 clusters (normal and defective)
- Defect localization: Sliding window approach with anomaly heatmap

## Exported Models

The trained models are saved in the `models/` directory:
- `fine_tuned_model.pth`: The fine-tuned PyTorch model
- `kmeans_model.pkl`: The trained KMeans clustering model

These files can be used for offline inference or integration into other applications. 
