import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
from model import create_model

class CaptchaPredictor:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the captcha predictor
        
        Args:
            model_path: Path to the saved model
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model, self.class_names = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (should match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model info
        class_names = checkpoint['class_names']
        model_type = checkpoint.get('model_type', 'lightweight')
        
        # Create model
        model = create_model(model_type, len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, class_names
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed tensor
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict_single(self, image_path, return_probabilities=False):
        """
        Predict a single captcha image
        
        Args:
            image_path: Path to the image file
            return_probabilities: If True, return class probabilities
            
        Returns:
            Predicted class and optionally probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
        
        if return_probabilities:
            prob_dict = {
                self.class_names[i]: probabilities[0][i].item() 
                for i in range(len(self.class_names))
            }
            return predicted_class, confidence_score, prob_dict
        else:
            return predicted_class, confidence_score
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict multiple captcha images
        
        Args:
            image_paths: List of image file paths
            return_probabilities: If True, return class probabilities
            
        Returns:
            List of predictions
        """
        results = []
        
        for image_path in image_paths:
            try:
                if return_probabilities:
                    pred_class, confidence, probabilities = self.predict_single(
                        image_path, return_probabilities=True
                    )
                    results.append({
                        'image_path': image_path,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'probabilities': probabilities
                    })
                else:
                    pred_class, confidence = self.predict_single(image_path)
                    results.append({
                        'image_path': image_path,
                        'predicted_class': pred_class,
                        'confidence': confidence
                    })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'predicted_class': None,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_directory(self, directory_path, file_extension='.png', return_probabilities=False):
        """
        Predict all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            file_extension: File extension to filter (default: '.png')
            return_probabilities: If True, return class probabilities
            
        Returns:
            List of predictions
        """
        # Get all image files
        image_files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.lower().endswith(file_extension.lower())
        ]
        
        print(f"Found {len(image_files)} images in {directory_path}")
        
        return self.predict_batch(image_files, return_probabilities)
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with confidence scores
        
        Args:
            image_path: Path to the image
            save_path: Optional path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        # Make prediction
        pred_class, confidence, probabilities = self.predict_single(
            image_path, return_probabilities=True
        )
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f'Predicted: {pred_class} (Confidence: {confidence:.3f})')
        ax1.axis('off')
        
        # Show probability distribution
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        bars = ax2.bar(classes, probs)
        ax2.set_title('Class Probabilities')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Highlight the predicted class
        pred_idx = classes.index(pred_class)
        bars[pred_idx].set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Captcha Prediction')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--image', type=str, help='Path to a single image to predict')
    parser.add_argument('--directory', type=str, help='Path to directory containing images')
    parser.add_argument('--output', type=str, help='Output file for batch predictions')
    parser.add_argument('--visualize', action='store_true', help='Visualize prediction')
    parser.add_argument('--probabilities', action='store_true', help='Show class probabilities')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = CaptchaPredictor(args.model)
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Please train a model first using train.py")
        return
    
    if args.image:
        # Predict single image
        print(f"Predicting image: {args.image}")
        
        if args.visualize:
            predictor.visualize_prediction(args.image)
        else:
            if args.probabilities:
                pred_class, confidence, probabilities = predictor.predict_single(
                    args.image, return_probabilities=True
                )
                print(f"Predicted class: {pred_class}")
                print(f"Confidence: {confidence:.3f}")
                print("Class probabilities:")
                for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {prob:.3f}")
            else:
                pred_class, confidence = predictor.predict_single(args.image)
                print(f"Predicted class: {pred_class}")
                print(f"Confidence: {confidence:.3f}")
    
    elif args.directory:
        # Predict all images in directory
        print(f"Predicting images in directory: {args.directory}")
        results = predictor.predict_from_directory(
            args.directory, return_probabilities=args.probabilities
        )
        
        # Print results
        for result in results:
            if 'error' in result:
                print(f"{result['image_path']}: ERROR - {result['error']}")
            else:
                print(f"{result['image_path']}: {result['predicted_class']} "
                      f"(Confidence: {result['confidence']:.3f})")
        
        # Save results to file if specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
    
    else:
        print("Please specify either --image or --directory")
        parser.print_help()

if __name__ == "__main__":
    main()
