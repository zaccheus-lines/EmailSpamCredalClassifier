import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

class NaiveCredalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        """
        Initialize the Naive Credal Classifier with Laplace smoothing.
        """
        self.N =None
        self.k=None
        self.n_c=None
        self.C = None
        self.label_encoder_ = None
        self.word_counts_per_class={}

    def fit(self, X, y):
        """
        Fit the Naive Credal model according to the given training data.
        """
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.C= self.label_encoder_.classes_

        self.N, self.k = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.n_c = dict(zip(unique_classes, class_counts))

        word_counts_per_class = {}

        for c in self.C:
            X_c = X[y == c]
        
            word_counts = np.array(X_c.sum(axis=0)).flatten()
            self.word_counts_per_class[c] = word_counts

        return self

    def inf(self, x, row, c1, c2):
        q = self.k * np.log((self.n_c[c2] + x) / (self.n_c[c1] + (1 - x)))
        prod = 0
        for i in range(self.k):
            if row[i][c1] != 0:
                prod += np.log((row[i][c1])) - np.log((row[i][c2] + x))
        return (q + prod)  # Negate if you're maximizing

    def predict(self, X, c1, c2):
        m, n = X.shape
        predictions = []  # Array to store predictions for each row
        
        for i in range(20):  # Loop over all rows
            row_vectors = []
            current_row_vector = []
            
            for j in range(n):
                n_i = {}
                
                for c in self.C:
                    if X[i, j] == 1:
                        n_i[c] = self.word_counts_per_class[c][j]
                    else:
                        n_i[c] = self.n_c[c] - self.word_counts_per_class[c][j]

                current_row_vector.append(n_i)
            
            row_vectors.append(current_row_vector)

            classes = [c1, c2]
            for attempt in range(2):  # Allows up to two attempts
                for row in row_vectors:
                    # Optimize 'inf' for the current row using the current classes
                    res = minimize(self.inf, x0=0.5, args=(row, classes[0], classes[1]), bounds=[(0.1, 0.9)])
                    optimal_x = res.x

                    # Use optimal_x to calculate the optimized 'inf' value for the row
                    optimized_inf = self.inf(optimal_x, row, classes[0], classes[1])

                # Check the condition and assign the predicted class to the row
                if optimized_inf >= np.log(1):
                    predictions.append(classes[0])  # Add the prediction for this row
                    break  # Exit the attempt loop since we have a prediction
                else:
                    # Swap classes for the next attempt
                    classes.reverse()
            else:
                # If neither class met the condition after all attempts, handle accordingly
                # For example, append None or a default class to predictions
                predictions.append(None)

        return np.array(predictions)  # Return an array of predictions for all rows
