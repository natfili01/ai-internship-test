from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from models.mnist_classifier_interface import MnistClassifierInterface  
import numpy as np

class RandomForestMnist(MnistClassifierInterface):  
    def __init__(self, n_estimators=200):#Initializes the Random Forest model.
        self.model = RandomForestClassifier(n_estimators=n_estimators) #n_estimators – Number of decision trees in the forest. 

    #Trains the model.
    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)  
    #Makes predictions on test data.
    def predict(self, test_data):
        return self.model.predict(test_data)

    #Evaluates the model and returns accuracy.
    def evaluate(self, test_data, test_labels):
        predictions = self.predict(test_data)
        return accuracy_score(test_labels, predictions)

