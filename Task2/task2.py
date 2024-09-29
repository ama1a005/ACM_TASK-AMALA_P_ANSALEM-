#dataset selection and preprocessing

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#building knn classifier from scratch
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label among the k neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

#building same model using standard libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create and train the KNN model using Scikit-learn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Predict on the test set
y_pred = knn.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Scikit-learn KNN Accuracy: {accuracy:.2f}")

#comparison result

# Evaluate the scratch-built model
knn_scratch = KNNClassifier(k=3)
knn_scratch.fit(X_train, y_train)
y_pred_scratch = knn_scratch.predict(X_test)

# Calculate accuracy for the scratch-built model
accuracy_scratch = np.mean(y_pred_scratch == y_test)
print(f"Scratch-built KNN Accuracy: {accuracy_scratch:.2f}")

# Compare
print(f"Difference in Accuracy: {accuracy - accuracy_scratch:.2f}")


'''Comparison of results
ccuracy/Performance: Both implementations provide the same accuracy since KNN is a simple algorithm, and both models perform similarly for small datasets.

Training Time: Scikit-learn is generally faster due to its use of optimized libraries.

Code Complexity: Scikit-learn abstracts away complexity, making the implementation simpler and less prone to errors.
.
Scalability: Scikit-learn scales much better with larger datasets due to built-in optimizations like KD-Trees or Ball Trees.'''





'''Scikit-learn's KNN is more practical in real world scenarios especially when dealing with large datasets while model built from scratch is
a great learning exercise to understand how knn actually works'''
