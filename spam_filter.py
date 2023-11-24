from typing import Iterable
import pandas as pd
import email
from email.parser import Parser
import os
from bs4 import BeautifulSoup
import time
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

start_time = time.time()

def safe_decode(payload, encoding='ISO-8859-1'):
    try:
        return payload.decode(encoding)
    except UnicodeDecodeError:
        return payload.decode(encoding, errors='ignore')

def tokenize(text: str) -> set[str]:
    # Normalize text: lowercase and replace newlines with spaces
    text = text.lower().replace('\n', ' ')

    text = re.sub(r'<[^>]+>', ' ', text)

    #pattern = r'\b\w+\b(?:\s+\b\w+\b)?'
    pattern = r'\b\w+\b'
    words = re.findall(pattern, text)
    
    tokens = set(words)
    return tokens

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

    # Create a DataFrame
    df = pd.DataFrame({
        'From': [from_],
        'Subject': [subject_],
        'Content': [payload],
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


hard_spam['label'] = 'spam'
easy_ham['label'] = 'ham'
hard_ham['label'] = 'ham'

# Combine all data into a single DataFrame
corpus = pd.concat([hard_spam, easy_ham, hard_ham])

# Apply the parse_email function to the 'email_content' column of the first 10 rows
parsed_rows = corpus.iloc[:].apply(lambda row: parse_email(row['email_content'], row['label']), axis=1)

# Concatenate the results
concatenated_df = pd.concat(parsed_rows.tolist())

union_set = set()
for email in concatenated_df['Content']:
    tokens = tokenize(email)
    union_set.update(tokens)


# Prepare a list to store the vectors and labels
data = []

# Iterate over each email and its label
for email, label in zip(concatenated_df['Content'], concatenated_df['Label']):
    # Tokenize the email
    email_tokens = set(tokenize(email))

    # Create a binary vector for this email
    vector = [int(token in email_tokens) for token in union_set]

    # Append the vector and label as a dictionary to the data list
    data.append({'vector': np.array(vector), 'label': label})

# Create a new DataFrame from the list
vector_df = pd.DataFrame(data)

print(vector_df)

elapsed = time.time() - start_time
print(f"Time elapsed: {elapsed} seconds")
