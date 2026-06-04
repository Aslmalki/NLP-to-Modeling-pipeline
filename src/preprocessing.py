"""
Preprocessing module for NLP Topic Modeling Pipeline.
ODE-compliant: pre-compiled regex, config paths.
"""
import os
import re

from collections import Counter
from itertools import chain

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Pre-compiled regex patterns for efficiency
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'\S+@\S+')
_RE_SPECIAL = re.compile(r'[^a-zA-Z\s]')
_RE_WHITESPACE = re.compile(r'\s+')


def _ensure_nltk_data():
    import nltk
    for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            if pkg == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif pkg == 'averaged_perceptron_tagger':
                nltk.data.find(f'taggers/{pkg}')
            else:
                nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)


def get_custom_stopwords():
    """Create an enhanced stopwords list that includes common non-informative terms."""
    _ensure_nltk_data()
    custom_stopwords = set(stopwords.words('english'))

    additional_stopwords = {
        'may', 'also', 'one', 'two', 'time', 'result', 'level', 'word',
        'get', 'make', 'use', 'take', 'give', 'find', 'come', 'go',
        'many', 'much', 'several', 'various', 'different',
        'first', 'second', 'third', 'last', 'next', 'previous',
        'year', 'month', 'week', 'day', 'hour', 'minute',
        'etc', 'ie', 'eg', 'et', 'al', 'fig', 'figure', 'table',
        'number', 'example', 'however', 'thus', 'therefore'
    }
    custom_stopwords.update(additional_stopwords)

    domain_terms = {
        'stress', 'adaptation', 'team', 'agent', 'model', 'task', 'group',
        'performance', 'study', 'change', 'desert', 'work', 'social',
        'individual', 'norm', 'research', 'system'
    }
    custom_stopwords = custom_stopwords - domain_terms

    return custom_stopwords


def load_data_from_drive(folder_path):
    """Load .txt files from a folder. Returns documents and filenames in sorted order."""
    documents = []
    filenames = []
    txt_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.txt'))
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                filenames.append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return documents, filenames


def preprocess_text(docs, min_word_length=3, custom_stopwords=None):
    """Preprocess documents: clean, tokenize, lemmatize, and remove stop words."""
    if custom_stopwords is None:
        custom_stopwords = get_custom_stopwords()

    lemmatizer = WordNetLemmatizer()
    preprocessed_docs = []

    for doc in docs:
        doc = doc.lower()
        doc = _RE_URL.sub('', doc)
        doc = _RE_EMAIL.sub('', doc)
        doc = _RE_SPECIAL.sub(' ', doc)
        doc = _RE_WHITESPACE.sub(' ', doc).strip()

        tokens = word_tokenize(doc)
        filtered_tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token not in custom_stopwords and len(token) >= min_word_length
        ]
        preprocessed_docs.append(' '.join(filtered_tokens))

    return preprocessed_docs


def calculate_dataset_statistics(preprocessed_docs):
    """Calculate and print statistics about the dataset after preprocessing."""
    num_documents = len(preprocessed_docs)
    print("\nDataset Statistics:")
    print(f"Total number of documents: {num_documents}")

    all_tokens = list(chain.from_iterable(doc.split() for doc in preprocessed_docs))
    vocabulary = set(all_tokens)
    vocabulary_size = len(vocabulary)
    print(f"Vocabulary size (number of unique words): {vocabulary_size}")

    total_words = len(all_tokens)
    print(f"Total number of words in the corpus: {total_words}")

    document_lengths = [len(doc.split()) for doc in preprocessed_docs]
    average_document_length = np.mean(document_lengths)
    max_document_length = np.max(document_lengths)
    min_document_length = np.min(document_lengths)
    std_document_length = np.std(document_lengths)

    print(f"Average document length: {average_document_length:.2f} words")
    print(f"Maximum document length: {max_document_length} words")
    print(f"Minimum document length: {min_document_length} words")
    print(f"Standard deviation of document lengths: {std_document_length:.2f} words")

    word_counts = Counter(all_tokens)
    top_words = word_counts.most_common(20)

    print("\nTop 20 most frequent words:")
    for word, count in top_words:
        print(f"  {word}: {count}")

    return word_counts, top_words
