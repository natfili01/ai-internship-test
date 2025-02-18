from models.random_forest_classifier import RandomForestMnist
import numpy as np

# Створюємо об'єкт моделі
rf_classifier = RandomForestMnist(n_estimators=10)

# Генеруємо тестові дані (10 зразків по 5 ознак)
X_train = np.random.rand(10, 5)  # Навчальні дані
y_train = np.random.randint(0, 2, 10)  # Випадкові мітки (0 або 1)

X_test = np.random.rand(3, 5)  # Тестові дані

# Навчаємо модель
rf_classifier.train(X_train, y_train)

# Робимо передбачення
predictions = rf_classifier.predict(X_test)
print(predictions)  # Виведе передбачені класи (0 або 1)
