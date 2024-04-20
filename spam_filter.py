import pandas as pd
import os
import re
from email.parser import Parser
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from NCC import NaiveCredalClassifier
import random
import numpy as np 
from scipy.stats import norm
from sklearn.model_selection import KFold


def safe_decode(payload, encoding='ISO-8859-1'):
    """Safely decode the payload with the given encoding."""
    try:
        return payload.decode(encoding)
    except UnicodeDecodeError:
        return payload.decode(encoding, errors='ignore')

def read_folder(folder_path):
    """Read all txt files in the given folder and return their content in a DataFrame."""
    data = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r', encoding='ISO-8859-1') as f:
            data.append({'email_content': f.read()})
    return pd.DataFrame(data)

def preprocess(text):
    """Preprocess text by lowercasing and removing non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_email(email_text, label):
    """Parse email content and return a DataFrame with extracted information."""
    msg = Parser().parsestr(email_text)
    content_type = msg.get_content_type()
    payload = msg.get_payload(decode=True)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in ['text/plain', 'text/html']:
                payload = safe_decode(part.get_payload(decode=True))
                break
    else:
        payload = safe_decode(payload)
    text_content = BeautifulSoup(payload, 'lxml').get_text() if 'html' in content_type else payload
    text_content = preprocess(text_content)
    return pd.DataFrame({
        'From': [msg['Return-Path'] or msg['From'] or msg['Sender']],
        'Subject': [msg['subject']],
        'Date': [msg['date']],
        'Content': [text_content],
        'Label': [label]
    })

def prepare_data(easy_ham_path = 'SpamCorpus/easy_ham', hard_ham_path='SpamCorpus/hard_ham', spam_path= 'SpamCorpus/spam'):
    """Prepare the dataset by reading emails, parsing, and vectorising the content."""
    # Read and label the datasets from specified folders
    easy_ham_emails = read_folder(easy_ham_path).assign(label='ham')
    hard_ham_emails = read_folder(hard_ham_path).assign(label='ham')
    spam_emails = read_folder(spam_path).assign(label='spam')

    # Combine all emails into a single DataFrame
    all_emails = pd.concat([easy_ham_emails, hard_ham_emails, spam_emails])

    # Parse emails
    parsed_emails = all_emails.apply(lambda row: parse_email(row['email_content'], row['label']), axis=1)
    df = pd.concat(parsed_emails.tolist())

    # Vsctorise the email content
    vectoriser = CountVectorizer(analyzer='word', binary=True)
    X = vectoriser.fit_transform(df['Content'])
    y = df['Label']

    return X,y

def model_predict(model, X, default_classes=None):
    """Adapts the predict call based on the model type."""
    if isinstance(model, NaiveCredalClassifier):
        if default_classes is None:
            raise ValueError("default_classes must be provided for NaiveCredalClassifier")
        return model.predict(X, *default_classes)
    else:
        return model.predict(X)

def evaluate_model(model, X_test, y_test, default_classes=('ham', 'spam')):
    """Evaluate the given model on the test set and print out performance metrics."""

    # Assuming model_predict is a function that handles model prediction
    y_pred = model_predict(model, X_test, default_classes=default_classes)

    # Identify valid (non-None) predictions
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_pred_filtered = [y_pred[i] for i in valid_indices] if valid_indices else y_pred
    y_test_filtered = y_test.iloc[valid_indices] if valid_indices else y_test

    # Calculate Accuracy
    accuracy = accuracy_score(y_test_filtered, y_pred_filtered) if valid_indices else 0
    n = len(y_pred_filtered)  # Number of non-None predictions
    if n > 0:
        acc_std_error = np.sqrt(accuracy * (1 - accuracy) / n)
        acc_confidence_interval = 1.96 * acc_std_error

    # Calculate Determinacy
    total_predictions = len(y_pred)
    determinacy = len(valid_indices) / total_predictions
    det_std_error = np.sqrt(determinacy * (1 - determinacy) / total_predictions)
    det_confidence_interval = 1.96 * det_std_error

    # Print Results
    #print(f"\nEvaluating {model.__class__.__name__}")
    print(f"Single Accuracy: {accuracy * 100:.2f}% ± {acc_confidence_interval*100:.2f}")
    print(f"Determinacy: {determinacy * 100:.2f}% ± {det_confidence_interval*100:.2f}")

    #print("Confusion Matrix (excluding 'None' predictions):\n", confusion_matrix(y_test_filtered, y_pred_filtered))
    #print("Classification Report (excluding 'None' predictions):\n", classification_report(y_test_filtered, y_pred_filtered))
    return accuracy, determinacy

def k_fold(X, y, model, n_splits=10, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    accuracies = []
    determinacies =[]

    # Ensure y is suitable for indexing
    y = pd.Series(y).reset_index(drop=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        accuracy, determinacy = evaluate_model(model, X_test,y_test)
        accuracies.append(accuracy)
        determinacies.append(determinacy)

    return np.mean(accuracies), np.std(accuracies), np.mean(determinacies), np.std(determinacies)

def epsilon_test(X,y, seed=50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    for e in range(1,11):
        print(f'epislon: {e/20}') 
        ncc = NaiveCredalClassifier(epsilon= e/20, s=0.5)                                        
        ncc.fit(X_train, y_train)
        evaluate_model(ncc, X_test, y_test)

def s_test(X,y, seed=35):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    for s in range(6,22,2):
        print(f's: {s/10}') 
        ncc = NaiveCredalClassifier(epsilon= 0.1, s=s/10)                                        
        ncc.fit(X_train, y_train)
        evaluate_model(ncc, X_test, y_test)

def main():

    start_time = time.time()
    r = random.randint(1,50)
    X, y = prepare_data()
    
    epsilon_test(X,y,50)
    
    #print(k_fold(X,y,ncc))
    
    

    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()