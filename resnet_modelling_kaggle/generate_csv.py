import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from train import build_wide_resnet
from train import load_cifar_batch


def predict_test_set(model_path, test_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = build_wide_resnet(width_multiplier=1.0, dropout_rate=0.2)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters")

    # Load the custom test data
    print(f"Loading test data from {test_file_path}")
    test_data = load_cifar_batch(test_file_path)
    test_images = test_data[b'data']
    print(f"Test data shape: {test_images.shape}")

    # Convert to PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Make predictions
    print("Making predictions on test data...")
    predictions = []
    with torch.no_grad():
        for i, image in enumerate(test_images):
            # Convert to 3D image format
            image = image.reshape(32, 32, 3)
            
            # Transform and predict
            img = transform(image).unsqueeze(0).to(device)
            output = model(img)
            _, predicted = output.max(1)
            predictions.append(predicted.item())
            
            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}/{len(test_images)} images")

    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    predict_test_set('best_model.pth', 'cifar_test_nolabel.pkl')
