import os
import sys
sys.path.append("models")  # Ensure the script finds `inference.py`

from inference import classify_animal

# Define the test images directory
test_images_dir = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\test_images"

# Check if the directory exists
if not os.path.exists(test_images_dir):
    print(f" Error: Directory not found: {test_images_dir}")
    sys.exit(1)

# Loop through all images in the folder
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)
    
    # Ensure it’s an image file
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        predicted_label = classify_animal(img_path)
        print(f" Image: {img_name} ->  Predicted: {predicted_label}")
    else:
        print(f" Skipping non-image file: {img_name}")
