import os
import sqlite3
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QSpinBox, QSplitter
from PyQt5.QtCore import Qt
from sklearn.metrics.pairwise import cosine_similarity

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.title = 'Meaningfull Search'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Use QSplitter for flexible layout
        self.splitter = QSplitter(Qt.Vertical)
        self.layout.addWidget(self.splitter)

        self.input_text = QTextEdit()
        self.splitter.addWidget(self.input_text)

        # Spinner for top N entries
        self.layout.addWidget(QLabel("Top N entries:"))
        self.spinner = QSpinBox()
        self.spinner.setMinimum(1)
        self.spinner.setMaximum(100)
        self.spinner.setValue(10)
        self.layout.addWidget(self.spinner)

        self.run_button = QPushButton('Search')
        self.run_button.clicked.connect(self.run_cosine_similarity)
        self.layout.addWidget(self.run_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.splitter.addWidget(self.result_text)

    def run_cosine_similarity(self):
        # Get input text
        text = self.input_text.toPlainText()

        # Get top N entries
        top_n = self.spinner.value()

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
            result_text += f'<hr><b>{i+1}.</b> (Similarity: {similarity[0][0]}) {book_name} - page {page_number}:<br>{paragraph}\n\n'
        self.result_text.setHtml(result_text)

        conn.close()

if __name__ == '__main__':
    app = QApplication([])
    ex = App()
    ex.show()
    app.exec_()
