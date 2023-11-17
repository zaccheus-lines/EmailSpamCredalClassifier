import pandas as pd
import os

# Function to read all txt files in a folder and return a DataFrame
def read_folder(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data.append(content)
    return pd.DataFrame(data, columns=['email_content'])

# Paths to the folders
hard_spam_path = 'SpamAssassin Public Corpus/hard spam'
easy_spam_path = 'SpamAssassin Public Corpus/easy spam'
not_spam_path = 'SpamAssassin Public Corpus/not spam'

# Reading each folder
hard_spam = read_folder(hard_spam_path)
easy_spam = read_folder(easy_spam_path)
not_spam = read_folder(not_spam_path)

# Optionally, add a label column
hard_spam['label'] = 'hard_spam'
easy_spam['label'] = 'easy_spam'
not_spam['label'] = 'not_spam'

# Combine all data into a single DataFrame
all_emails = pd.concat([hard_spam, easy_spam, not_spam])

# Your processing code goes here

# Save or use the DataFrame as needed
