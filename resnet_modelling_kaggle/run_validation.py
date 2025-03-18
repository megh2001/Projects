# valid.py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import the model architecture and utility functions from train.py
from train import (
    BasicBlock, WideResNet, build_wide_resnet, 
    load_cifar_batch, load_cifar_data, CIFAR10Dataset
)

def validate_model(model, val_loader, device='cuda', return_predictions=False):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect statistics
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    if return_predictions:
        return accuracy, all_preds, all_targets
    return accuracy

def visualize_results(all_preds, all_targets, class_names):
    """
    Visualize the validation results with confusion matrix and other metrics.
    """
    # Convert lists to numpy arrays if they aren't already
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    
    # Plot class-wise accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy * 100)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.xticks(rotation=45)
    
    # Add accuracy numbers on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.5, 
                f"{class_accuracy[i]*100:.1f}%", 
                ha='center')
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.show()
    
    # Return the overall accuracy
    return np.mean(class_accuracy) * 100

def benchmark_speed(model, input_size=(1, 3, 32, 32), device='cuda', num_runs=100):
    """
    Benchmark the inference speed of the model.
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nInference Speed Benchmark:")
    print(f"  Average inference time per image: {avg_time:.2f} ms (± {std_time:.2f})")
    print(f"  Throughput: {1000/avg_time:.1f} images/second")
    
    return avg_time

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set paths and parameters
    data_dir = '/content/cifar-10-python/cifar-10-batches-py'
    model_path = 'best_model.pth'
    batch_size = 128
    width = 1.0
    dropout = 0.2
    
    # Build the model with the same architecture used during training
    model = build_wide_resnet(
        width_multiplier=width, 
        dropout_rate=dropout
    )
    
    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Display model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters")
    
    # Load CIFAR-10 test set (to use as validation)
    print(f"Loading CIFAR-10 validation data from {data_dir}...")
    _, _, val_data, val_labels, class_names = load_cifar_data(data_dir)
    print(f"Validation data loaded with {len(val_labels)} samples")
    
    # Define validation transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])
    
    # Create validation dataset and loader
    val_dataset = CIFAR10Dataset(val_data, val_labels, transform=val_transform, is_train=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    # Validate the model with visualizations
    accuracy, all_preds, all_targets = validate_model(
        model, val_loader, device, return_predictions=True
    )
    
    # Visualize results
    visualize_results(all_preds, all_targets, class_names)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Run inference speed benchmark
    benchmark_speed(model, device=device)
    
    # Plot per-class validation examples with predictions
    # Visualize model feature activations
    class_counts = {i: 0 for i in range(10)}
    max_examples_per_class = 5
    examples_to_show = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            if all(count >= max_examples_per_class for count in class_counts.values()):
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predictions[i].item()
                
                if class_counts[label] < max_examples_per_class:
                    class_counts[label] += 1
                    examples_to_show.append((images[i].cpu(), label, pred))

    # Plot some predictions
    rows = 2
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(15, 6))
    
    for i, (image, label, pred) in enumerate(examples_to_show[:rows*cols]):
        r, c = i // cols, i % cols
        img = image.permute(1, 2, 0).numpy()
        # Denormalize image
        img = img * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)
        
        axs[r, c].imshow(img)
        title_color = "green" if label == pred else "red"
        axs[r, c].set_title(f"True: {class_names[label]}\nPred: {class_names[pred]}", 
                          color=title_color)
        axs[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()
    
if _name_ == "__main__":
    main()
