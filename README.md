# Meaningful Search

Meaningful Search is a powerful tool designed to find the most similar paragraphs in a large collection of PDF files based on the input text. It uses BERT (Bidirectional Encoder Representations from Transformers), a transformer-based machine learning model to create semantic representations of the input and the text from the PDF files. These representations are then compared using cosine similarity to find the most similar paragraphs.

<img src="https://github.com/alihakimtaskiran/SemanticSearch/blob/main/screenshoot.png" width="400">

## Getting Started

### Prerequisites

To use the Meaningful Search, you will need to have the following installed:

* Python 3.7 or newer
* PyPDF2
* PyQt5
* transformers
* torch
* numpy
* sqlite3
* sklearn

You can install these packages using pip:

```
pip install PyPDF2 PyQt5 transformers torch numpy sqlite3 sklearn
```

### Installation

1. Clone the repository
```
git clone https://github.com/your_username_/Meaningful_Search.git
```
2. Navigate to the repository
```
cd Meaningful_Search
```

### Usage

1. Use `cacher.py` to cache the semantic representations of the text in your PDF files into a SQLite database. You can do this by calling the `cache_pdf_encodings` function with the path to the directory containing your PDF files as an argument:
```python
from cacher import cache_pdf_encodings
cache_pdf_encodings('path/to/your/pdf/files/')
```
This will create a SQLite database named `pdf_encodings.db` containing the semantic representations of the text in your PDF files.

2. Run `search.py` to start the Meaningful Search application:
```python
python search.py
```

### How to use the App

1. Enter the text you want to search in the input field.
2. Specify the number of most similar paragraphs you want to display using the spinner.
3. Click the 'Search' button to start the search. The application will display the most similar paragraphs along with their similarity scores, the names of the books they came from, and their page numbers.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU GPL 3.0 License. See `LICENSE` for more information.

## Contact

Ali Hakim Taskiran - [alihakimxyz@gmail.com](mailto:alihakimxyz@gmail.com)

Project Link: [https://github.com/alihakimtaskiran/SemanticSearch](https://github.com/alihakimtaskiran/SemanticSearch)
## Acknowledgments

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro)
* [PyPDF2](https://github.com/mstamy2/PyPDF2)
