import argparse
import os
import random
from inference_ner import extract_animal_name
from inference import classify_animal  # Import CNN classification function

# Function to process both text and image
def process_pipeline(image_path, text):
    detected_animal_text = extract_animal_name(text)  # Extract animal name from text
    detected_animal_image = classify_animal(image_path)  # Classify animal in image
    
    # Compare extracted text vs classified image
    print(f"Extracted Animal from Text: {detected_animal_text}")
    print(f"Predicted Animal from Image: {detected_animal_image}")

    match = detected_animal_image.strip().lower() in [animal.strip().lower() for animal in detected_animal_text]

    print(f"Text: {detected_animal_text} | Image: {detected_animal_image} | Match: {match}")
    return match

# CLI for testing & batch mode
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--text", type=str, help="Text describing the image")
    
    args = parser.parse_args()

    if args.image and args.text:
        # Manual single test case
        result = process_pipeline(args.image, args.text)
        print(f"Prediction: {result}")
    else:
        # Automated Random Image Selection from Dataset
        test_dir = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\datasets_split\test"

        # Get random images from each category
        random_test_cases = []
        for category in os.listdir(test_dir):
            category_path = os.path.join(test_dir, category)
            if os.path.isdir(category_path):
                images = os.listdir(category_path)
                if images:
                    random_image = random.choice(images)
                    random_test_cases.append((f"I see a {category} in the image.", os.path.join(category_path, random_image)))

        # Run the test for each randomly selected image
        for text, img in random_test_cases:
            print("\n-----------------------------")
            print(f"Testing: {text} with {img}")
            process_pipeline(img, text)
