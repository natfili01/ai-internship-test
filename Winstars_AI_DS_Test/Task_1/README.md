# MNIST Classification Project - Task 1

##  Project Overview
This project evaluates three different models on the MNIST dataset to classify handwritten digits:
- Random Forest
- FeedForward Neural Network
- Convolutional Neural Network (CNN)

The goal is to compare traditional machine learning (Random Forest) with deep learning models (FeedForward NN & CNN) to analyze performance differences.

---

##  Final Accuracy Scores:
| Model                   | Accuracy on MNIST |
|-------------------------|------------------|
| Random Forest          | 72% |
| FeedForward NN        | 74% |
| Convolutional Neural Network (CNN) | 85% |

The CNN model achieves the highest accuracy, as expected for an image classification task.

---

##  How to Run the Project

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Run Model Comparison

```sh
python compare_models.py
```

### Project Structure

Winstars_AI_DS_Test/
│── Task_1/
│   │── models/
│   │   │── __init__.py
│   │   │── mnist_classifier_interface.py
│   │   │── random_forest_classifier.py
│   │   │── neural_network_classifier.py
│   │   │── cnn_classifier.py
│   │── compare_models.py
│   │── README.md
│   │── requirements.txt
│   │── data/  # Folder containing MNIST dataset
│   │── test_cnn.py
│   │── test_neural_network.py
│   │── test_random_forest.py


### Model Details & Key Improvements

### Random Forest


    Uses flattened MNIST images as input.
    Works well but is outperformed by neural networks on image tasks.

###  FeedForward Neural Network

    Fully connected layers with ReLU activation.
    Initially underperformed but improved with optimized hyperparameters.

### Convolutional Neural Network (CNN)

    Uses 4 convolutional layers for feature extraction.
    Optimized with:
        AdamW optimizer (better weight decay handling)
        Learning rate scheduler (StepLR) for stable training
        Increased batch size (improves stability)
        Proper label formatting (integer labels instead of one-hot encoding)

### Fixes & Optimizations Applied

 - Used AdamW optimizer (improves weight decay handling).
 - Implemented Learning Rate Scheduler (improves convergence).
 - Normalized MNIST images (ensures stable training).
 - Fixed label formatting (CNN required integer labels, not one-hot).
 - Increased batch size (better stability).

### Possible improvements:

  Fine-tuning hyperparameters for CNN
  Exploring different architectures (e.g., ResNet)
  Testing on more complex datasets

### Final Notes

This project successfully compares traditional ML vs. deep learning for digit classification. CNN achieves the best performance, proving the effectiveness of deep learning for image tasks.
