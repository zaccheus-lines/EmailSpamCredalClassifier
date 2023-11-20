import pandas as pd
import email
from email.parser import Parser
import os
from bs4 import BeautifulSoup

# Function to read all txt files in a folder and return a DataFrame
def read_folder(folder_path):
    data = []
    for file in os.listdir(folder_path):
        #if file.endswith(".txt"): 
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            content = f.read()
            data.append(content)
    return pd.DataFrame(data, columns=['email_content'])

def parse_email(email_text):
    # Parse the email
    msg = Parser().parsestr(email_text)

    # Extract details
    from_ = msg['Return-Path']
    subject_ = msg['subject']
    date_ = msg['date']
    content_type_ = msg.get_content_type()
    payload = msg.get_payload()

    # Get the email body
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' or part.get_content_type() == 'text/html':
                payload = part.get_payload(decode=True).decode()
                break
    else:
        payload_ = msg.get_payload(decode=True).decode('ISO-8859-1')

    # Parse HTML content to extract text if it's HTML
    if 'html' in content_type_:
        soup = BeautifulSoup(payload, 'lxml')
        text_content = soup.get_text()
    else:
        text_content = payload

    # Create a DataFrame
    df = pd.DataFrame({
        'From': [from_],
        'Subject': [subject_],
        'Date': [date_],
        'Content': [text_content]
    })

    return df

# Paths to the folders
hard_spam_path = 'SpamCorpus/spam'
easy_spam_path = 'SpamCorpus/easy_ham'
ham_path = 'SpamCorpus/hard_ham'

# Reading each folder
hard_spam = read_folder(hard_spam_path)
easy_spam = read_folder(easy_spam_path)
ham = read_folder(ham_path)

# Optionally, add a label column
hard_spam['label'] = 'hard_spam'
easy_spam['label'] = 'easy_spam'
ham['label'] = 'ham'

# Combine all data into a single DataFrame
corpus = pd.concat([hard_spam, easy_spam, ham])

# Apply the parse_email function to the 'email_content' column of the first 10 rows
parsed_rows = corpus.iloc[:10]['email_content'].apply(parse_email)

# Concatenate the results
concatenated_df = pd.concat(parsed_rows.tolist())

print(concatenated_df)