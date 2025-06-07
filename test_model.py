import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from app import load_model, preprocess_image, load_trained_model, localize_defects

def main():
    parser = argparse.ArgumentParser(description='Test the anomaly detection model on a single image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the trained models')
    parser.add_argument('--output', type=str, default='result.png', help='Path to save the result image')
    args = parser.parse_args()

    # Check if the image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        sys.exit(1)

    # Check if the model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist. Please train the model first.")
        sys.exit(1)

    # Load the models
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, base_model = load_model()
    
    # Check if the fine-tuned model exists
    if not os.path.exists(os.path.join(args.model_dir, 'fine_tuned_model.pth')):
        print(f"Error: Fine-tuned model not found in {args.model_dir}. Please train the model first.")
        sys.exit(1)
    
    # Load the fine-tuned model
    fine_tuned_model, kmeans = load_trained_model(base_model, args.model_dir)
    
    # Simple cluster to label mapping (can be improved with proper mapping)
    cluster_to_label = {0: 0, 1: 1}
    
    # Process the image
    print(f"Processing image: {args.image_path}")
    
    # Preprocess and extract features
    img_tensor = preprocess_image(args.image_path).to(device)
    
    # Fine-tuned model prediction
    with torch.no_grad():
        outputs = fine_tuned_model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        is_defective = predicted.item() == 1
    
    if is_defective:
        print("RESULT: Defect detected!")
        
        # Localize defects
        result_img, heatmap = localize_defects(args.image_path, base_model, kmeans, cluster_to_label)
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.imread(args.image_path)[:, :, ::-1])
        plt.title("Original")
        plt.axis('off')
        
        # Defect localization
        plt.subplot(1, 3, 2)
        plt.imshow(result_img)
        plt.title("Defect Localization")
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 3)
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar(label='Anomaly Score')
        plt.title("Anomaly Heatmap")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"Result saved to {args.output}")
    else:
        print("RESULT: No defects detected. This item appears to be good.")
        
        # Just show the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.imread(args.image_path)[:, :, ::-1])
        plt.title("No Defects Detected")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main() 
