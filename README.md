Winstars AI DS Internship Test 2025 - Task 2 
 Named Entity Recognition + Image Classification Pipeline
 Project Overview
This project builds a Machine Learning pipeline that:
•	Extracts animal names from text using a Named Entity Recognition (NER) model.
•	Classifies images of animals using a CNN-based model.
•	Verifies if the user-provided text matches the image content.
________________________________________
 Project Structure
 Task_2
 datasets_split/   # Training & validation datasets
 models/           # Saved trained models
 test_images/      # Sample images for inference
 utils/            # Helper functions (data processing, training)
 train.py          # Train the image classification model
 inference.py      # Inference script for the classifier
 test_inference.py # Test classification model with sample images
 ner_model.py      # NER model implementation
 pipeline.py       # Final end-to-end text + image verification
 requirements.txt  # Dependencies list
 README.md         # Project documentation
________________________________________
 Installation & Setup
Clone the repository:
git clone https://github.com/YOUR_GITHUB_USERNAME/Winstars_AI_DS_Test.git
cd Winstars_AI_DS_Test/Task_2
Install dependencies:
pip install -r requirements.txt
Download the necessary transformer model for NLP:
python -m spacy download en_core_web_trf
________________________________________
 Model Training
Train Image Classifier
To train the CNN model:
python train.py
The trained model will be saved in:
models/cnn_model.keras
Train Named Entity Recognition (NER) Model
To fine-tune the NER model:
python ner_train.py
The trained NER model will be saved in:
models/ner_model.bin
________________________________________
 Inference
Classify a Single Image
To predict an image's class:
python test_inference.py
Example Output:
 Image: tiger.jpg ->  Predicted: Tiger
 Image: elephant.png ->  Predicted: Elephant
Extract Animals from Text
To test the NER model:
from ner_model import extract_animal
text = "There is a lion in the picture."
print(extract_animal(text))
Example Output:
Detected Animal: Lion
________________________________________
 Final Pipeline (Text + Image Matching) To verify if text matches an image:
python pipeline.py
Expected Output
Input Text: "There is a zebra in the picture."
Predicted Image Class: Zebra
 Match: True
OR (if incorrect prediction)
 Mismatch: Expected "Giraffe", but found "Deer".
________________________________________
 Future Fine-Tuning Options
1. Improve Model Generalization
✔ Apply stronger augmentation (e.g., more rotations, flips, brightness shifts) to improve performance on unseen images. ✔ Increase training epochs if necessary.
2. Enhance NER Model
✔ Fine-tune the Named Entity Recognition (NER) model on a larger domain-specific dataset. ✔ Extend support for multiple animals in a sentence.
3. Optimize for Deployment
✔ Convert model to TensorFlow Lite (.tflite) for mobile/edge deployment. ✔ Use model pruning/quantization to reduce size and increase inference speed.
________________________________________
 References
•	Hugging Face Transformers for Named Entity Recognition
•	TensorFlow/Keras for CNN Image Classification
•	Seaborn & Matplotlib for Visualization
________________________________________
 Credits Developed by [Your Name] for Winstars AI DS Internship 2025

