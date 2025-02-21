import sys
sys.path.append("models")  # Ensure models/ directory is accessible

from inference import classify_animal

img_path = "test_images/any.jpg"  # Replace with a real image path
predicted_label = classify_animal(img_path)

print(f" Predicted Animal: {predicted_label}")
