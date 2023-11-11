from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split dataset into training and testing set
# Ensure 50 images per class in the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, stratify=y, random_state=42)

# Nearest Neighbor Classifier
nn_classifier = KNeighborsClassifier(n_neighbors=1)
nn_classifier.fit(X_train, y_train)
nn_predictions = nn_classifier.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print(f"Nearest Neighbor Accuracy: {nn_accuracy}")

# k-Nearest Neighbors Classifier for k=3, 5, 7
for k in [3, 5, 7]:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_predictions = knn_classifier.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f"k-Nearest Neighbors Accuracy for k={k}: {knn_accuracy}")
