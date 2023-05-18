import os
import sqlite3
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class AppCLI:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def run_cosine_similarity(self, text, top_n):
        # Encode input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        input_encoding = outputs.last_hidden_state[:, 0, :].numpy()

        # Connect to the database
        conn = sqlite3.connect('pdf_encodings.db')
        c = conn.cursor()

        # Fetch all encodings
        c.execute('SELECT book_name, page_number, paragraph, encoding FROM encodings')
        rows = c.fetchall()

        # Calculate cosine similarity for each encoding
        similarities = []
        for row in rows:
            encoding = np.frombuffer(row[3], dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(input_encoding, encoding)
            similarities.append((similarity, row[0], row[1], row[2]))

        # Sort by cosine similarity
        similarities.sort(reverse=True)

        # Display the top N most similar paragraphs
        result_text = ''
        for i, (similarity, book_name, page_number, paragraph) in enumerate(similarities[:top_n]):
            result_text += f'\n\n{"="*50}\n{i+1}. (Similarity: {similarity[0][0]}) {book_name} - page {page_number}:\n{"-"*50}\n{paragraph}\n'
        
        return result_text

if __name__ == '__main__':
    app = AppCLI()
    text = input('Enter your text: ')
    top_n = int(input('Enter the number of top entries: '))
    result = app.run_cosine_similarity(text, top_n)
    print(result)

