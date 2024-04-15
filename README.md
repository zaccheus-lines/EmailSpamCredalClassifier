# Email Spam Filtering using Naive Credal Classifiers

# Naive Credal Classifier

The Naive Credal Classifier (NCC) is an extension of the traditional Naive Bayes classifier that incorporates elements of imprecise probabilities to better handle uncertainty in classification tasks. This implementation is particularly useful in scenarios where the class distributions are not well defined or when dealing with sparse data sets.

## Features

- **Robust to Sparse Data:** Ideal for datasets where classes are not perfectly separable or have overlapping features.
- **Imprecise Probabilities:** Utilises credal sets for managing uncertainty in class definitions.
- **Customizable Smoothing:** Supports Laplace smoothing with customizable `epsilon` and `s` parameters to stabilize likelihood estimates in the face of sparse data.

## Installation

Before you can use the Naive Credal Classifier, you need to install the required Python libraries. Ensure you have Python installed on your system, then run the following command to install dependencies:

```bash
# Clone the repository
git clone https://github.com/zaccheus-lines/EmailSpamCredalClassifier

# Navigate to the project directory
cd EmailSpamCredalClassifier

# Install dependencies
pip install -r requirements.txt
```
## Usage
To run the spam filter, use the following command:

```bash
python spam_filter.py
```
## Usage

To integrate and use the Naive Credal Classifier in your Python projects, follow these steps:

```python
# Import the classifier
from NCC import NaiveCredalClassifier

# Initialize the classifier with custom epsilon and s values
classifier = NaiveCredalClassifier(e=0.5, s=1)

# Fit the model to your data (assume X_train and y_train are your features and labels)
classifier.fit(X_train, y_train)

# Make predictions (assuming X_test is your test feature matrix)
predictions = classifier.predict(X_test, 'class1', 'class2')
print(predictions)


## Dataset
The project uses a publicly available email dataset, which contains a mix of spam and non-spam emails. The dataset has been preprocessed to suit the requirements of Naive Credal Classifiers.

## Methodology
The project utilises Naive Credal Classifiers for spam detection. This method extends traditional probability-based classifiers by allowing for imprecise probabilities, making it more suitable for real-world applications where data may be incomplete or uncertain.


## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## Contact
For any queries or collaborations, feel free to contact me at zaccheus.lines@hotmail.com

## Acknowledgements
Special thanks to my dissertation supervisor, Prof. Matthias Troffaes , and the Durham University Mathematics department for their support and guidance throughout this project.


