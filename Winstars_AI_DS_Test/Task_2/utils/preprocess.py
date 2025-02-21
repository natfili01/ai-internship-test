import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_dir):
    # Отримуємо список класів
    classes = sorted(os.listdir(train_dir))

    # Оновлюємо кількість зображень у кожному класі
    class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in classes}

    # Формуємо список міток
    y_labels = []
    for idx, cls in enumerate(classes):
        y_labels.extend([idx] * class_counts[cls])  

    # Обчислюємо ваги класів
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(range(len(classes))),  # Використовуємо числові індекси
        y=y_labels
    )

    return {i: weight for i, weight in enumerate(class_weights)}
