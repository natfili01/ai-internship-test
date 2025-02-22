# **Winstars AI DS Internship Test 2025**

## **Project Overview**
This project consists of two major tasks, covering different aspects of Machine Learning:
1. **MNIST Classification (Task 1)** – Comparing three models (Random Forest, FeedForward NN, CNN) for handwritten digit classification.
2. **Named Entity Recognition (NER) + Image Classification (Task 2)** – Building an ML pipeline that extracts animal names from text and verifies if they match the provided image.

---

## **Repository Structure**
```plaintext
Winstars_AI_DS_Test/
│── Task_1/                  # MNIST Classification (Handwritten Digits)
│   │── models/              # Implemented classifiers (Random Forest, Neural Network, CNN)
│   │── compare_models.py    # Script to compare models' performance
│   │── data/                # MNIST dataset
│   │── test_cnn.py          # CNN model testing
│   │── test_neural_network.py # Neural Network model testing
│   │── test_random_forest.py # Random Forest model testing
│   │── README.md            # Task 1 documentation
│
│── Task_2/                  # NER + Image Classification Pipeline
│   │── datasets_split/      # Training & validation datasets
│   │── models/              # Saved trained models
│   │── test_images/         # Sample images for inference
│   │── utils/               # Helper functions (data processing, training)
│   │── train.py             # Train the image classification model
│   │── inference.py         # Classifier inference script
│   │── ner_model.py         # Named Entity Recognition model implementation
│   │── pipeline.py          # End-to-end text + image verification pipeline
│   │── README.md            # Task 2 documentation
│
│── requirements.txt         # Project dependencies
│── README.md                # General project documentation
```

---

## **Installation & Setup**
### **1. Clone the Repository**
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/Winstars_AI_DS_Test.git
cd Winstars_AI_DS_Test
```

### **2. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3. Download Required NLP Model (for Task 2)**
```sh
python -m spacy download en_core_web_trf
```

---

## **Running the Project**
### **Task 1: MNIST Classification**
To compare the performance of all three models on MNIST:
```sh
cd Task_1
python compare_models.py
```
Expected output: Accuracy scores for Random Forest, Neural Network, and CNN.

### **Task 2: NER + Image Classification Pipeline**
1. **Train the Image Classifier (CNN)**
   ```sh
   cd Task_2
   python train.py
   ```
   The trained model will be saved in:
   ```
   models/cnn_model.keras
   ```

2. **Train the Named Entity Recognition (NER) Model**
   ```sh
   python ner_train.py
   ```
   The trained model will be saved in:
   ```
   models/ner_model.bin
   ```

3. **Run Inference for Image Classification**
   ```sh
   python test_inference.py
   ```
   Example Output:
   ```
   Image: tiger.jpg -> Predicted: Tiger
   Image: elephant.png -> Predicted: Elephant
   ```

4. **Test NER Model for Extracting Animals from Text**
   ```python
   from ner_model import extract_animal
   text = "There is a lion in the picture."
   print(extract_animal(text))
   ```
   Expected Output:
   ```
   Detected Animal: Lion
   ```

5. **Run Full Pipeline (Text + Image Matching)**
   ```sh
   python pipeline.py
   ```
   Example Output:
   ```
   Input Text: "There is a zebra in the picture."
   Predicted Image Class: Zebra
   Match: True
   ```
   OR (if incorrect prediction):
   ```
   Mismatch: Expected "Giraffe", but found "Deer".
   ```

---

## **Future Improvements**
### **Task 1 (MNIST Classification)**
- Fine-tuning CNN hyperparameters
- Exploring more advanced architectures (e.g., ResNet)
- Evaluating on more complex datasets

### **Task 2 (NER + Image Classification)**
1. **Improve Model Generalization**
   - Apply stronger augmentation (e.g., more rotations, flips, brightness shifts)
   - Increase training epochs if necessary
2. **Enhance NER Model**
   - Fine-tune the NER model on a larger domain-specific dataset
   - Extend support for multiple animals in a sentence
3. **Optimize for Deployment**
   - Convert models to TensorFlow Lite (.tflite) for mobile deployment
   - Use model pruning/quantization for faster inference

---

## **Technologies Used**
- **Machine Learning:** Scikit-Learn, TensorFlow/Keras, PyTorch
- **NLP:** SpaCy, Hugging Face Transformers
- **Data Processing:** Pandas, NumPy, OpenCV
- **Visualization:** Matplotlib, Seaborn

---

## **References**
- Hugging Face Transformers for Named Entity Recognition
- TensorFlow/Keras for CNN Image Classification
- Scikit-Learn for traditional ML models
- Seaborn & Matplotlib for data visualization

---

## **Developed by**  
**Nataliia Lauria** for Winstars AI DS Internship 2025.
