import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

class NaiveCredalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon = 0.01, s=0.5, dominance='credal'):
        """
        Initialize the Naive Credal Classifier with Laplace smoothing.
        """
        self.N =None
        self.k=None
        self.n_c=None
        self.classes_ = None
        self.label_encoder_ = None
        self.word_counts_per_class={}
        self.epsilon = epsilon
        self.s = s
        self.dominance = dominance

    def inf(self, x, row, c1, c2):
        quot = (self.k - 1) * (np.log((self.n_c[c2] + x)) - np.log(self.n_c[c1] + self.s - x))
        term1 = np.log(row[c1] + self.s*self.epsilon)
        term2 = np.log(row[c2] + x)
        prod = np.sum(term1 - term2)
        return quot + prod

    
    def fit(self, X, y):
        """
        Fit the Naive Credal model according to the given training data.
        """
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_= self.label_encoder_.classes_

        self.N, self.k = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.n_c = dict(zip(unique_classes, class_counts))

        word_counts_per_class = {}

        for c in self.classes_:
            X_c = X[y == c]
        
            word_counts = np.array(X_c.sum(axis=0)).flatten()
            self.word_counts_per_class[c] = word_counts

        return self

    def predict(self, X, c1, c2):
        if self.dominance == 'stoch':
            return self.stoch_predict(X, c1, c2)
        elif self.dominance == 'credal':
            return self.credal_predict(X, c1, c2)
        else:
            raise ValueError("Unknown dominance type specified. Choose 'stoch' or 'credal'.")

    def credal_predict(self, X, c1, c2):
        m, n = X.shape
        predictions = [] 
        classes = [c1, c2]
        for i in range(m):
            row_array = X[i, :].toarray().flatten()
            #print(row_array,i) 

            n_i = {}
            for c in classes:
                a = self.word_counts_per_class[c]*row_array
                b = (self.n_c[c] - self.word_counts_per_class[c])*(1-row_array)
                n_i[c] =a+b

            for attempt in range(2):  
                # Optimise 'inf' for the current row using the current classes
                res = minimize(self.inf, x0=0.75, args=(n_i, classes[0], classes[1]), bounds=[(self.s*self.epsilon, self.s)])
                optimal_x = res.x

                # Use optimal_x to calculate the optimised 'inf' value for the row
                optimised_inf = self.inf(optimal_x, n_i, classes[0], classes[1])

                # Check the condition and assign the predicted class to the row
                if optimised_inf >= np.log(1):
                    predictions.append(classes[0])
                    break 
                else:
                    # Swap classes for the next attempt
                    classes.reverse()
            else:
                predictions.append(None)

    
        return np.array(predictions)  # Return an array of predictions for all rows
    
    def stoch_predict(self, X, c1, c2):
        m, n = X.shape
        predictions = [] 
        classes = [c1, c2]
        for i in range(m):
            row_array = X[i, :].toarray().flatten()

            n_i = {}
            for c in classes:
                a = self.word_counts_per_class[c] * row_array
                b = (self.n_c[c] - self.word_counts_per_class[c]) * (1 - row_array)
                n_i[c] = a + b

            for attempt in range(2):
                # Calculate the bounds for the first class
                lower_bound_c1 = -(self.k - 1)* np.log((self.n_c[classes[0]] + self.s)) + np.sum(np.log(n_i[classes[0]]+ self.s*self.epsilon))
                # Calculate the bounds for the second class
                upper_bound_c2 = -(self.k - 1)* np.log((self.n_c[classes[1]] + self.s)) + np.sum(np.log(n_i[classes[1]]+ self.s))

                # Compare the bounds to make a prediction
                if lower_bound_c1 > upper_bound_c2:
                    predictions.append(classes[0])
                    break
                else:
                    # Swap the classes for the second attempt
                    classes.reverse()
            else:
                # Append None if no class dominates after two attempts
                predictions.append(None)

        return np.array(predictions)  # Return an array of predictions for all rows


    def get_params(self, deep=True):
        return {'epsilon': self.epsilon, 's': self.s}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self