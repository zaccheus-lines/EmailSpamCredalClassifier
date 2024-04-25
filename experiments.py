import os
import re
import time
import numpy as np
import pandas as pd
from email.parser import Parser
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from NCC import NaiveCredalClassifier
from sklearn.naive_bayes import MultinomialNB

# Constants
ENCODING = 'ISO-8859-1'
DATA_PATHS = {
    'easy_ham': 'SpamCorpus/easy_ham',
    'hard_ham': 'SpamCorpus/hard_ham',
    'spam': 'SpamCorpus/spam'
}

def safe_decode(payload, encoding=ENCODING):
    """Safely decode the payload with the given encoding."""
    try:
        return payload.decode(encoding)
    except UnicodeDecodeError:
        return payload.decode(encoding, errors='ignore')

def read_folder(folder_path):
    """Read all txt files in the given folder and return their content in a DataFrame."""
    data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding=ENCODING) as file:
            data.append({'email_content': file.read()})
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
    text_content = payload

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in ['text/plain', 'text/html']:
                text_content = safe_decode(part.get_payload(decode=True))
                break
    else:
        text_content = safe_decode(payload)

    if 'html' in content_type:
        text_content = BeautifulSoup(text_content, 'lxml').get_text()
    
    text_content = preprocess(text_content)
    return pd.DataFrame({
        'From': [msg.get('Return-Path', msg.get('From', msg.get('Sender')))],
        'Subject': [msg.get('Subject', '')],
        'Date': [msg.get('Date', '')],
        'Content': [text_content],
        'Label': [label]
    })

def prepare_data(df=DATA_PATHS):
    """Prepare the dataset by reading emails, parsing, and vectorising the content."""
    # Read and label the datasets from specified folders
    easy_ham_emails = read_folder(df['easy_ham']).assign(label='ham')
    hard_ham_emails = read_folder(df['hard_ham']).assign(label='ham')
    spam_emails = read_folder(df['spam']).assign(label='spam')

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
    """Evaluate the given model on the test set and return performance metrics along with indeterminate cases."""

    # Assuming model_predict is a function that handles model prediction
    y_pred = model_predict(model, X_test, default_classes=default_classes)

    # Identify valid (non-None) predictions
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    indeterminate_indices = [i for i, pred in enumerate(y_pred) if pred is None]

    # Filter valid predictions for evaluation
    y_pred_filtered = [y_pred[i] for i in valid_indices]
    y_test_filtered = y_test.iloc[valid_indices]

    # Calculate Accuracy
    accuracy = accuracy_score(y_test_filtered, y_pred_filtered) if valid_indices else 0

    # Calculate Determinacy
    determinacy = len(valid_indices) / len(y_pred)

    # Retrieve indeterminate cases from test set
    indeterminate_X = X_test[indeterminate_indices]
    indeterminate_y = y_test.iloc[indeterminate_indices]

    return accuracy, determinacy, indeterminate_X, indeterminate_y


def k_fold_cross_validation(X, y, model, seed=1, n_splits=5, shuffle=True):
    """Perform K-fold cross validation."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    accuracies = []
    determinacies = []
    y = pd.Series(y).reset_index(drop=True)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        accuracy, determinacy, indeterminate_X, indeterminate_y = evaluate_model(model, X_test,y_test)
        accuracies.append(accuracy)
        determinacies.append(determinacy)

    return np.mean(accuracies), np.std(accuracies), np.mean(determinacies), np.std(determinacies), indeterminate_X, indeterminate_y

def epsilon_test(X, y, seed=50, dominance = 'credal'):
    for e in range(1,11):
        print(f'epsilon: {e/40}') 

        print('Credal')
        ncc = NaiveCredalClassifier(epsilon= e/40, s=1, dominance='credal')   
        acc, acc_sd, det, det_sd,_ ,_ = k_fold_cross_validation(X,y, ncc, seed = seed)    
        print(f"K-fold Single Accuracy: {acc:.2%} ± {acc_sd:.2%}")
        print(f"K-fold Deteminacy: {det:.2%} ± {det_sd:.2%}")

        print('Interval')
        ncc = NaiveCredalClassifier(epsilon= e/20, s=1, dominance='stoch')   
        acc, acc_sd, det, det_sd,_ ,_ = k_fold_cross_validation(X,y, ncc, seed = seed)    
        print(f"K-fold Single Accuracy: {acc:.2%} ± {acc_sd:.2%}")
        print(f"K-fold Deteminacy: {det:.2%} ± {det_sd:.2%}")

def s_test(X,y, seed = 50):
    for s in range(1,11):
        print(f's: {s/2}')

        print('Credal') 
        ncc = NaiveCredalClassifier(epsilon= 0.25, s=s/2, dominance = 'credal') 
        acc, acc_sd, det, det_sd, _, _ = k_fold_cross_validation(X,y, ncc, seed = seed)    
        print(f"K-fold Single Accuracy: {acc:.2%} ± {acc_sd:.2%}")
        print(f"K-fold Deteminacy: {det:.2%} ± {det_sd:.2%}")

        print('Stochastic')   
        ncc = NaiveCredalClassifier(epsilon= 0.25, s=s/2, dominance = 'stoch') 
        acc, acc_sd, det, det_sd, _, _ = k_fold_cross_validation(X,y, ncc, seed = seed)    
        print(f"K-fold Single Accuracy: {acc:.2%} ± {acc_sd:.2%}")
        print(f"K-fold Deteminacy: {det:.2%} ± {det_sd:.2%}")

def comparison(X, y, seed=50):
    # Initialise classifiers
    ncc_credal = NaiveCredalClassifier(epsilon=0.25, s=1, dominance = "credal")
    ncc_stochastic = NaiveCredalClassifier(epsilon=0.25, s=1, dominance="stoch")
    nbc = MultinomialNB()

    # Comparing Naive Credal Classifier (Credal Dominance) results
    acc_ncc_credal, acc_sd_ncc_credal, det_ncc_credal, det_sd_ncc_credal, credal_indeterminate_X, credal_indeterminate_y = k_fold_cross_validation(X, y, ncc_credal, seed=seed)
    print(f"NCC (Credal Dominance) - K-fold Single Accuracy: {acc_ncc_credal:.2%} ± {acc_sd_ncc_credal:.2%}")
    print(f"NCC (Credal Dominance) - K-fold Determinacy: {det_ncc_credal:.2%} ± {det_sd_ncc_credal:.2%}")

    # Comparing Naive Credal Classifier (Stochastic Dominance) results
    acc_ncc_stochastic, acc_sd_ncc_stochastic, det_ncc_stochastic, det_sd_ncc_stochastic, _, _ = k_fold_cross_validation(X, y, ncc_stochastic, seed=seed)
    print(f"NCC (Stochastic Dominance) - K-fold Single Accuracy: {acc_ncc_stochastic:.2%} ± {acc_sd_ncc_stochastic:.2%}")
    print(f"NCC (Stochastic Dominance) - K-fold Determinacy: {det_ncc_stochastic:.2%} ± {det_sd_ncc_stochastic:.2%}")

    # Comparing Multinomial Naive Bayes results
    acc_nbc, acc_sd_nbc, det_nbc, det_sd_nbc, _, _ = k_fold_cross_validation(X, y, nbc, seed=seed)
    print(f"NBC - K-fold Single Accuracy: {acc_nbc:.2%} ± {acc_sd_nbc:.2%}")
    print(f"NBC - K-fold Determinacy: {det_nbc:.2%} ± {det_sd_nbc:.2%}")

    # Analysing cases where NCC (Credal Dominance) was indeterminate
    ind_acc_nbc, ind_acc_sd_nbc, ind_det_nbc, ind_det_sd_nbc, _, _= k_fold_cross_validation(credal_indeterminate_X, credal_indeterminate_y, nbc, seed=seed)
    print(f"NBC - K-fold Single Accuracy (Credal Indeterminate Cases): {ind_acc_nbc:.2%} ± {ind_acc_sd_nbc:.2%}")
    print(f"NBC - K-fold Determinacy (Credal Indeterminate Cases): {ind_det_nbc:.2%} ± {ind_det_sd_nbc:.2%}")


def main():
    """Main function to run processes."""
    start_time = time.time()
    X, y = prepare_data()

    #epsilon_test(X,y, seed = 5)

    #s_test(X,y,seed=5)

    comparison(X,y, seed=5)

    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
