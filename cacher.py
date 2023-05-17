import os
import sqlite3
import PyPDF2
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def cache_pdf_encodings(folder_path):
    # Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Connect to a SQLite database or create one if it does not exist
    conn = sqlite3.connect('pdf_encodings.db')
    c = conn.cursor()

    # Create a table to store the PDF encodings if it does not exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS encodings (
            id INTEGER PRIMARY KEY,
            book_name TEXT,
            page_number INTEGER,
            paragraph TEXT,
            encoding BLOB
        )
    ''')

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            # Extract book name from file name
            book_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)

            # Read the PDF file
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                # Iterate through each page in the PDF
                for page_number in range(len(reader.pages)):
                    page = reader.pages[page_number]
                    text = page.extract_text()

                    # Split the text into paragraphs
                    paragraphs = text.split('\n\n')

                    # Encode each paragraph using BERT
                    for paragraph in paragraphs:
                        # If the paragraph is too long, we split it into chunks of 512 tokens
                        inputs = tokenizer(paragraph, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
                        with torch.no_grad():
                            outputs = model(**inputs)
                        encoding = outputs.last_hidden_state[:, 0, :].numpy()

                        # Insert the encoding into the database
                        c.execute('''
                            INSERT INTO encodings (
                                book_name, page_number, paragraph, encoding
                            ) VALUES (?, ?, ?, ?)
                        ''', (book_name, page_number, paragraph, encoding.tobytes()))

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

# Example usage:
cache_pdf_encodings('path/to/directory/will/be/cached/')
