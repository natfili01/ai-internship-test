import random
import json

#  Define animals in dataset
animals = ["lion", "tiger", "elephant", "zebra", "giraffe", "dog", "cat", "wolf", "rabbit", "bear"]

#  Define sentence templates
templates = [
    "I saw a {} in the jungle.",
    "A {} is running in the forest.",
    "There is a {} in the picture.",
    "Have you ever seen a {} at the zoo?",
    "The {} was sleeping under a tree.",
    "A {} appeared in my dream last night.",
    "I love watching {}s in nature documentaries.",
    "A {} was spotted near the river today.",
    "The {} is a beautiful and majestic animal.",
    "A group of {}s were walking together."
]

#  Generate synthetic dataset
dataset = []
for animal in animals:
    for template in templates:
        sentence = template.format(animal)
        tokens = sentence.split()
        labels = [1 if word.strip(".") == animal else 0 for word in tokens]  # Mark only animal names
        
        dataset.append({"tokens": tokens, "ner_tags": labels})

#  Save dataset as JSON
with open("synthetic_ner_data.json", "w") as f:
    json.dump(dataset, f, indent=4)

print(f" Generated {len(dataset)} synthetic NER sentences and saved to 'synthetic_ner_data.json'")
