import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

#  Load the trained NER model
MODEL_PATH = "./ner_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

#  Create an inference pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

#  Function to extract animal names from text
def extract_animal_name(text):
    ner_results = ner_pipeline(text)
    animals_detected = [res["word"] for res in ner_results if res["score"] > 0.85]
    return animals_detected

#  Example Usage
if __name__ == "__main__":
    text = "There is a lion in the picture."
    detected_animals = extract_animal_name(text)
    print(f" Detected animals: {detected_animals}")
