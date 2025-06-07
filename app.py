import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torchvision.transforms.functional as TF
from transformers import ViTFeatureExtractor, ViTModel
import glob
import random
import cv2
from tqdm import tqdm
import joblib
import json
import pandas as pd
import sys
import traceback

# Set up error logging
def log_error(message):
    with open("error_log.txt", "a") as f:
        f.write(f"{message}\n")

# Capture uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Uncaught exception: {error_message}")
    log_error(f"Uncaught exception: {error_message}")

sys.excepthook = handle_exception

print("ALL IMPORTS SUCCESSFUL")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add more debug at the script level
print("DEBUG: Script starting execution")
log_error("DEBUG: Script starting execution")

# Image preprocessing
def preprocess_image(image_path):
    """Preprocess image for the DINO model"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Load and initialize the DINO model
@st.cache_resource
def load_model():
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    model = ViTModel.from_pretrained('facebook/dino-vitb16')
    model = model.to(device)
    model.eval()
    return feature_extractor, model

# Helper function to collect dataset images
def collect_dataset_images(dataset_path):
    """
    Navigate through dataset folders and collect images
    Returns a dictionary with category, path, and label information
    """
    dataset = {'train': [], 'test': []}
    categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d not in ['license.txt', 'readme.txt']]
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        
        # Training data (only 'good' images)
        train_good_path = os.path.join(category_path, 'train', 'good')
        if os.path.exists(train_good_path):
            for img_path in glob.glob(os.path.join(train_good_path, '*.png')):
                dataset['train'].append({
                    'path': img_path,
                    'category': category,
                    'label': 'good',
                    'defect_type': None
                })
        
        # Test data (both 'good' and defective images)
        test_path = os.path.join(category_path, 'test')
        if os.path.exists(test_path):
            for defect_type in os.listdir(test_path):
                defect_path = os.path.join(test_path, defect_type)
                if os.path.isdir(defect_path):
                    for img_path in glob.glob(os.path.join(defect_path, '*.png')):
                        label = 'good' if defect_type == 'good' else 'defective'
                        dataset['test'].append({
                            'path': img_path,
                            'category': category,
                            'label': label,
                            'defect_type': None if defect_type == 'good' else defect_type
                        })
                        
    print(f"Collected {len(dataset['train'])} training images and {len(dataset['test'])} test images")
    return dataset

# Extract features using the DINO model
def extract_features(model, image_paths, batch_size=16):
    """Extract features from images using the DINO model"""
    features = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = torch.cat([preprocess_image(path) for path in batch_paths])
        batch_images = batch_images.to(device)
        
        with torch.no_grad():
            batch_features = model(batch_images).last_hidden_state[:, 0, :].cpu().numpy()
        
        features.append(batch_features)
    
    return np.vstack(features)

# Train KMeans clustering model
def train_kmeans(features, n_clusters=2):
    """Train KMeans clustering on the extracted features"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

# Evaluate the model
def evaluate_model(model, kmeans, test_data):
    """Evaluate the model on test data"""
    test_paths = [item['path'] for item in test_data]
    true_labels = [0 if item['label'] == 'good' else 1 for item in test_data]
    
    # Extract features
    test_features = extract_features(model, test_paths)
    
    # Predict clusters
    predicted_clusters = kmeans.predict(test_features)
    
    # Map clusters to labels (0: good, 1: defective)
    # We need to determine which cluster corresponds to which label
    cluster_label_counts = {}
    for true_label, pred_cluster in zip(true_labels, predicted_clusters):
        if pred_cluster not in cluster_label_counts:
            cluster_label_counts[pred_cluster] = [0, 0]
        cluster_label_counts[pred_cluster][true_label] += 1
    
    # Determine which cluster should be mapped to which label
    cluster_to_label = {}
    for cluster, counts in cluster_label_counts.items():
        # Assign the cluster to the label with the highest count
        cluster_to_label[cluster] = 0 if counts[0] > counts[1] else 1
    
    # Map predictions
    predictions = [cluster_to_label[cluster] for cluster in predicted_clusters]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['good', 'defective'])
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels
    }

# Localize defects in an image
def localize_defects(image_path, model, kmeans, cluster_to_label, patch_size=24, stride=12):
    """
    Localize defects in an image by sliding a window and classifying patches
    Returns the original image with bounding boxes around defective regions
    
    Parameters:
    - image_path: Path to the image
    - model: Base DINO model for feature extraction
    - kmeans: Trained KMeans model
    - cluster_to_label: Mapping from cluster IDs to labels (0=good, 1=defective)
    - patch_size: Size of patches to analyze (smaller size for finer detection)
    - stride: Stride for sliding window (smaller stride for more overlap)
    """
    # Load the original image
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape
    
    # Create a heatmap to store anomaly scores
    heatmap = np.zeros((h, w))
    counts = np.zeros((h, w))
    
    # Store distances for all patches
    all_distances = []
    patch_positions = []
    
    # Create sliding windows
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            # Extract patch
            patch = orig_img[y:y+patch_size, x:x+patch_size]
            
            # Convert to PIL Image and preprocess
            patch_pil = Image.fromarray(patch)
            patch_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(patch_pil).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = model(patch_tensor).last_hidden_state[:, 0, :].cpu().numpy()
            
            # Get cluster and distance information
            cluster = kmeans.predict(features)[0]
            distances = kmeans.transform(features)[0]
            min_distance = distances[cluster]
            
            # Use normalized distance as anomaly score (higher = more anomalous)
            anomaly_score = min_distance / (np.sum(distances) + 1e-10)
            
            # Store distance information
            all_distances.append(anomaly_score)
            patch_positions.append((y, x))
            
            # Standard binary prediction for detection
            is_defective = cluster_to_label[cluster] == 1
            
            # Update heatmap with anomaly score rather than binary value
            if is_defective:
                heatmap[y:y+patch_size, x:x+patch_size] += anomaly_score * 2  # Weight defective patches more
            else:
                # Still add a small value for non-defective patches based on distance
                heatmap[y:y+patch_size, x:x+patch_size] += anomaly_score * 0.5
                
            counts[y:y+patch_size, x:x+patch_size] += 1
    
    # Normalize heatmap
    heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts!=0)
    
    # Scale heatmap to 0-1 range for better visualization
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Use a fixed threshold for detection (middle ground value)
    threshold = 0.5
    binary_map = (heatmap > threshold).astype(np.uint8)
    
    # Apply morphological operations to clean up the binary map
    kernel = np.ones((3, 3), np.uint8)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    binary_map = cv2.dilate(binary_map, np.ones((5, 5), np.uint8), iterations=1)
    
    # Find contours in the binary map
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around defective regions
    result_img = orig_img.copy()
    has_significant_defect = False
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter very small contours
            x, y, w, h = cv2.boundingRect(contour)
            # Larger/more anomalous areas get red, smaller/less anomalous get yellow
            color = (255, 0, 0) if area > 200 else (255, 255, 0)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            has_significant_defect = True
    
    # If no significant defects found, but we have high anomaly areas, mark them
    if not has_significant_defect and np.max(heatmap) > threshold/2:
        # Find the most anomalous region
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Draw a dashed circle around it
        cv2.circle(result_img, (x, y), 25, (0, 255, 255), 2, cv2.LINE_AA)
        # Add text
        cv2.putText(result_img, "Potential anomaly", (x-30, y-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return result_img, heatmap

# Fine-tune the model using PyTorch
def fine_tune_model(base_model, train_data, epochs=5, learning_rate=1e-5):
    """
    Fine-tune the DINO model on our specific dataset
    """
    # We'll fine-tune the model by adding a classification head
    class AnomalyDetector(torch.nn.Module):
        def __init__(self, base_model):
            super(AnomalyDetector, self).__init__()
            self.base_model = base_model
            # Freeze the base model weights
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Add a classification head
            self.classifier = torch.nn.Linear(base_model.config.hidden_size, 2)
            
        def forward(self, x):
            outputs = self.base_model(x)
            cls_token = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_token)
            return logits
    
    # Create the model
    model = AnomalyDetector(base_model)
    model = model.to(device)
    
    # Prepare data
    train_paths = [item['path'] for item in train_data]
    # Since we only have good samples in training, we'll create artificial defects
    # by applying random transformations to some of the good samples
    train_labels = []
    train_tensors = []
    
    # Data augmentation for "defective" samples
    augment_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    normal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For each good image, create one normal and one augmented version
    for path in tqdm(train_paths, desc="Preparing training data"):
        img = Image.open(path).convert('RGB')
        
        # Normal sample (label 0 = good)
        normal_tensor = normal_transform(img)
        train_tensors.append(normal_tensor)
        train_labels.append(0)
        
        # Augmented sample (label 1 = defective)
        augmented_tensor = augment_transform(img)
        train_tensors.append(augmented_tensor)
        train_labels.append(1)
    
    # Convert to tensors
    train_tensors = torch.stack(train_tensors)
    train_labels = torch.tensor(train_labels)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_tensors, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / len(progress_bar),
                'accuracy': 100 * correct / total
            })
    
    print(f"Final training accuracy: {100 * correct / total:.2f}%")
    return model

# Save the trained model
def save_model(model, kmeans, model_dir='models'):
    """Save the trained model and KMeans clustering model"""
    print("\n\n--------ATTEMPTING TO SAVE MODELS--------")
    print(f"Current directory: {os.getcwd()}")
    print(f"Saving to directory: {os.path.abspath(model_dir)}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Save the fine-tuned model
        model_path = os.path.join(model_dir, 'fine_tuned_model.pth')
        print(f"Saving model to: {model_path}")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully: {os.path.exists(model_path)}")
        
        # Save the KMeans model
        kmeans_path = os.path.join(model_dir, 'kmeans_model.pkl')
        print(f"Saving KMeans to: {kmeans_path}")
        joblib.dump(kmeans, kmeans_path)
        print(f"KMeans saved successfully: {os.path.exists(kmeans_path)}")
        
        print(f"Models saved to {model_dir}")
        print("--------SAVE COMPLETE--------\n\n")
    except Exception as e:
        print(f"ERROR SAVING MODELS: {str(e)}")
        import traceback
        traceback.print_exc()

# Load the trained model
@st.cache_resource
def load_trained_model(_base_model, model_dir='models'):
    """Load the fine-tuned model and KMeans clustering model"""
    class AnomalyDetector(torch.nn.Module):
        def __init__(self, base_model):
            super(AnomalyDetector, self).__init__()
            self.base_model = base_model
            # Freeze the base model weights
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Add a classification head
            self.classifier = torch.nn.Linear(base_model.config.hidden_size, 2)
            
        def forward(self, x):
            outputs = self.base_model(x)
            cls_token = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_token)
            return logits
    
    # Create the model
    model = AnomalyDetector(_base_model)
    
    # Load the state dict
    model.load_state_dict(torch.load(os.path.join(model_dir, 'fine_tuned_model.pth')))
    model = model.to(device)
    model.eval()
    
    # Load the KMeans model
    kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
    
    return model, kmeans

# Main function for the CLI application
def main():
    print("Industrial Anomaly Detection using DINO-ViT-B16")
    
    try:
        print("Loading the DINO model...")
        # Load the DINO model
        feature_extractor, base_model = load_model()
        print("DINO model loaded successfully")
        
        print("Collecting dataset images...")
        # Collect dataset images
        dataset = collect_dataset_images("Dataset")
        
        print("Extracting features for training...")
        # Extract features for training (use only good samples)
        train_paths = [item['path'] for item in dataset['train']]
        print(f"Extracting features from {len(train_paths)} training images...")
        train_features = extract_features(base_model, train_paths)
        print("Feature extraction complete")
        
        print("Training KMeans clustering model...")
        # Train KMeans clustering
        kmeans = train_kmeans(train_features)
        print("KMeans training complete")
        
        print("Fine-tuning the model...")
        # Fine-tune the model
        fine_tuned_model = fine_tune_model(base_model, dataset['train'])
        print("Fine-tuning complete")
        
        print("Saving the model...")
        # Save the model
        save_model(fine_tuned_model, kmeans)
        print("Model saving complete")
        
        print("Evaluating on test data...")
        # Evaluate on test data
        metrics = evaluate_model(base_model, kmeans, dataset['test'])
        
        # Print evaluation metrics
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Sample defect localization on a few test images
        print("\nPerforming defect localization on sample images...")
        
        # Determine cluster to label mapping
        cluster_label_counts = {}
        for true_label, pred_cluster in zip(metrics['true_labels'], kmeans.predict(train_features)):
            if pred_cluster not in cluster_label_counts:
                cluster_label_counts[pred_cluster] = [0, 0]
            cluster_label_counts[pred_cluster][true_label] += 1
        
        cluster_to_label = {}
        for cluster, counts in cluster_label_counts.items():
            cluster_to_label[cluster] = 0 if counts[0] > counts[1] else 1
        
        # Get some defective test samples
        defective_samples = [item for item in dataset['test'] if item['label'] == 'defective']
        if defective_samples:
            sample_indices = random.sample(range(len(defective_samples)), min(5, len(defective_samples)))
            
            for idx in sample_indices:
                sample = defective_samples[idx]
                print(f"\nProcessing sample {idx+1}: {os.path.basename(sample['path'])}")
                print(f"Category: {sample['category']}, Defect type: {sample['defect_type']}")
                
                # Localize defects
                result_img, _ = localize_defects(sample['path'], base_model, kmeans, cluster_to_label)
                
                # Save the result
                result_path = f"result_sample_{idx+1}.png"
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.imread(sample['path'])[:, :, ::-1])
                plt.title("Original")
                plt.subplot(1, 2, 2)
                plt.imshow(result_img)
                plt.title("Defect Localization")
                plt.tight_layout()
                plt.savefig(result_path)
                print(f"Result saved to {result_path}")
        else:
            print("No defective samples found in the test dataset")
        
        print("\nProcess completed successfully")
        
    except Exception as e:
        print(f"\n\nERROR IN MAIN: {str(e)}")
        import traceback
        traceback.print_exc()

# Streamlit app
def streamlit_app():
    st.title("Industrial Anomaly Detection")
    st.write("Upload an image to detect anomalies using DINO-ViT-B16 model")
    
    # Add a progress indicator for model loading
    with st.spinner("Loading base DINO model..."):
        # Load the model
        feature_extractor, base_model = load_model()
    
    # Check if the model is already fine-tuned
    model_dir = 'models'
    if os.path.exists(os.path.join(model_dir, 'fine_tuned_model.pth')):
        with st.spinner("Loading fine-tuned model... This might take a few seconds"):
            # Use leading underscore for base_model when passing to load_trained_model
            fine_tuned_model, kmeans = load_trained_model(_base_model=base_model)
        st.success("Model loaded successfully!")
    else:
        st.error("Fine-tuned model not found. Please run the CLI application first.")
        return
    
    # Determine cluster to label mapping
    # We'll use a simple approach for the demo: cluster 0 = good, cluster 1 = defective
    cluster_to_label = {0: 0, 1: 1}
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(temp_path, caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        with st.spinner("Detecting anomalies..."):
            # Preprocess and extract features
            img_tensor = preprocess_image(temp_path).to(device)
            
            # Model prediction (standard binary approach)
            with torch.no_grad():
                outputs = fine_tuned_model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
                is_defective = predicted.item() == 1
        
        if is_defective:
            st.error("⚠️ Defect detected!")
            
            # Localize defects with enhanced visualization
            with st.spinner("Localizing defects..."):
                result_img, heatmap = localize_defects(
                    temp_path, 
                    base_model, 
                    kmeans, 
                    cluster_to_label
                )
                
                # Display results
                st.image(result_img, caption="Defect Localization", use_column_width=True)
                
                # Display heatmap
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(heatmap, cmap='hot')
                plt.colorbar(im, label='Anomaly Score')
                plt.title("Anomaly Heatmap")
                plt.tight_layout()
                plt.savefig("heatmap.png")
                st.image("heatmap.png", caption="Anomaly Heatmap", use_column_width=True)
        else:
            st.success("✅ No defects detected. This item appears to be good.")
            
            # Even if classified as good, show a light analysis
            with st.expander("View Analysis Details"):
                # Generate the heatmap anyway to see if there are any 
                # borderline issues that didn't reach the threshold
                result_img, heatmap = localize_defects(
                    temp_path, 
                    base_model, 
                    kmeans, 
                    cluster_to_label
                )
                
                # Display the heatmap at lower alpha
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(heatmap, cmap='hot', alpha=0.7)
                plt.colorbar(im, label='Anomaly Score')
                plt.title("Inspection Heatmap (No Significant Anomalies)")
                plt.tight_layout()
                plt.savefig("good_heatmap.png")
                st.image("good_heatmap.png", caption="Inspection Heatmap", use_column_width=True)
                st.write(f"Maximum anomaly score: {np.max(heatmap):.4f} (below threshold)")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists("heatmap.png"):
            os.remove("heatmap.png")
        if os.path.exists("good_heatmap.png"):
            os.remove("good_heatmap.png")

if __name__ == "__main__":
    # A better way to check if running within Streamlit
    import sys
    
    # Check if the script was launched through streamlit
    # This is a more reliable way to check if we're running in Streamlit
    streamlit_cmd = any('streamlit' in arg for arg in sys.argv)
    
    if not streamlit_cmd:
        # Normal mode - run the training and evaluation
        main()
    # If running in Streamlit mode, do nothing - Streamlit will call streamlit_app() 
