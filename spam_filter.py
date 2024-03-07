import pandas as pd
import numpy as np
from email.parser import Parser
import os
from bs4 import BeautifulSoup
import time
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


start_time = time.time()

def safe_decode(payload, encoding='ISO-8859-1'):
    try:
        return payload.decode(encoding)
    except UnicodeDecodeError:
        return payload.decode(encoding, errors='ignore')

def preprocess(text):
    # Lowercasing, removing non-alphabetic characters.
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_email(email_text, label):
    # Parse the email
    msg = Parser().parsestr(email_text)

    # Extract details
    from_ = msg['Return-Path'] or msg['From'] or msg['Sender']
    subject_ = msg['subject']
    date_ = msg['date']
    content_type_ = msg.get_content_type()
    payload = msg.get_payload()

    # Get the email body
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' or part.get_content_type() == 'text/html':
                payload = safe_decode(part.get_payload(decode=True))
                break
    else:
        payload = safe_decode(msg.get_payload(decode=True))

    # Parse HTML content to extract text if it's HTML
    if 'html' in content_type_:
        soup = BeautifulSoup(payload, 'lxml')
        text_content = soup.get_text()
    else:
        text_content = payload
    
    text_content= preprocess(text_content)



    # Create a DataFrame
    df = pd.DataFrame({
        'From': [from_],
        'Subject': [subject_],
        'Date': [date_],
        'Content': [text_content],
        'Label': [label]
    })

    return df

# Function to read all txt files in a folder and return a DataFrame
def read_folder(folder_path):
    data = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                content = f.read()
                data.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(data, columns=['email_content'])



# Paths to the folders
spam_path = 'SpamCorpus/spam'
easy_ham_path = 'SpamCorpus/easy_ham'
hard_ham_path = 'SpamCorpus/hard_ham'

# Reading each folder
hard_spam = read_folder(hard_ham_path)
easy_ham = read_folder(easy_ham_path)
hard_ham = read_folder(hard_ham_path)

# Add a label column
hard_spam['label'] = 'spam'
easy_ham['label'] = 'ham'
hard_ham['label'] = 'ham'

# Combine all data into a single DataFrame
corpus = pd.concat([hard_spam, easy_ham, hard_ham])

# Apply the parse_email function to the 'email_content' column of the first 10 rows
parsed_rows = corpus.apply(lambda row: parse_email(row['email_content'], row['label']), axis=1)

# Concatenate the results
concatenated_df = pd.concat(parsed_rows.tolist())
print(concatenated_df)
# Vectorize the preprocessed text
vectorizer = CountVectorizer(analyzer='word', binary=True)

X = vectorizer.fit_transform(concatenated_df['Content'])
y = concatenated_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

nb_classifier = MultinomialNB()

# 3. Train the classifier
nb_classifier.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{report}')

word_counts = np.array(X.sum(axis=0)).flatten()
print(word_counts)
elapsed = time.time() - start_time
print(f"Time elapsed: {elapsed} seconds")
