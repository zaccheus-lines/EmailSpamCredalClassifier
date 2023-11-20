import pandas as pd
import email
from email.parser import Parser
import os
from bs4 import BeautifulSoup
import time
import re

start_time = time.time()

def safe_decode(payload, encoding='ISO-8859-1'):
    try:
        return payload.decode(encoding)
    except UnicodeDecodeError:
        return payload.decode(encoding, errors='ignore')

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

    text_content = re.sub(r'[^a-zA-Z0-9]', ' ', text_content)  # Replace with space
    text_content = re.sub(r'\s+', ' ', text_content).strip()  # Replace multiple spaces with a single space
    text_content = text_content.lower()

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

# Optionally, add a label column
hard_spam['label'] = 'hard_spam'
easy_ham['label'] = 'easy_ham'
hard_ham['label'] = 'ham'

# Combine all data into a single DataFrame
corpus = pd.concat([hard_spam, easy_ham, hard_ham])

# Apply the parse_email function to the 'email_content' column of the first 10 rows
parsed_rows = corpus.iloc[:10].apply(lambda row: parse_email(row['email_content'], row['label']), axis=1)

# Concatenate the results
concatenated_df = pd.concat(parsed_rows.tolist())

print(concatenated_df)

elapsed = time.time() - start_time
print(f"Time elapsed: {elapsed} seconds")
