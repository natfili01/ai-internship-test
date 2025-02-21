from abc import ABC, abstractmethod  

class MnistClassifierInterface(ABC):  
    @abstractmethod
    def train(self, train_data, train_labels):
        
        pass  

    @abstractmethod
    def predict(self, test_data):
        
        pass  

