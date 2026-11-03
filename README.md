# Movie Review Sentiment Analysis
## CS 3810 – Introduction to Artificial Intelligence
### Mini Project 2
#### Hannah Dees and Alexis Stacy

---

## Overview

This project explores sentiment analysis using multiple Natural Language Processing (NLP) techniques. The goal is to classify movie reviews as either positive or negative and compare the effectiveness of traditional machine learning approaches, word embeddings, transformer models, and prompt-based large language models.

The dataset used is the NLTK Movie Reviews corpus, containing 2,000 labeled movie reviews.

---

## Dataset

The project uses the NLTK Movie Reviews dataset.

- 2,000 movie reviews
- 1,000 positive reviews
- 1,000 negative reviews
- 80% training data
- 20% testing data

```python
from nltk.corpus import movie_reviews
```

---

# Project Components

## Part 1: Data Preprocessing

Reviews were cleaned and normalized before model training.

### Preprocessing Steps

- Convert text to lowercase
- Expand contractions
- Remove punctuation
- Remove numbers
- Tokenize text
- Create clean training and testing datasets

---

## Part 2: Classical Machine Learning

### Bag of Words + Naive Bayes

A Bag of Words representation was generated from the movie reviews and used to train a Naive Bayes classifier.

### TF-IDF + Logistic Regression

TF-IDF weighting was applied to improve feature quality and reduce the impact of common words.

### TF-IDF + Support Vector Machine

Two different feature sets were compared:

#### Unigrams

Single-word features.

Examples:

```text
good
bad
excellent
```

#### Unigrams + Bigrams

Word pairs and phrases.

Examples:

```text
not good
very bad
waste of
```

---

## Part 3: Word Embeddings and Transformers

### Word2Vec

Word2Vec creates dense vector representations of words based on context and co-occurrence patterns.

Models tested:

- Logistic Regression
- Multi-Layer Perceptron (MLP)

### BERT Embeddings

Pre-trained BERT embeddings were used to create contextual document representations.

Models tested:

- Logistic Regression
- Multi-Layer Perceptron (MLP)

---

## Part 4: TinyLlama Prompting

TinyLlama was used to perform sentiment classification through prompting rather than traditional training.

Prompting approaches explored:

### Zero-Shot Prompting

No examples provided.

### Few-Shot Prompting

Several labeled examples are included with the prompt.

### Persona Prompting

The model is instructed to act as an experienced film critic.

### Chain-of-Thought Prompting

The model reasons through the review before reaching a final conclusion.

---

# Technologies Used

- Python
- NLTK
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- PyTorch
- Transformers
- TinyLlama
- Jupyter Notebook
- VS Code

---

# Challenges Encountered

Several technical challenges were encountered while setting up the environment and executing the project.

## ARM64 Python Compatibility Issues

The original development environment used Python 3.12 ARM64.

This created compatibility problems with several required packages.

### Gensim Installation Issues

The Word2Vec portion of the project required:

```python
from gensim.models import Word2Vec
```

Unfortunately, the package could not be successfully compiled in the ARM64 environment because of dependency and architecture compatibility issues.

### PyTorch Installation Issues

The project also required:

```python
import torch
```

PyTorch could not initially be installed in the ARM64 virtual environment and repeatedly produced installation errors.

Example error:

```text
No matching distribution found for torch
```

### Windows Smart App Control

Windows Smart App Control blocked multiple scientific libraries from loading correctly.

Examples included:

- Pandas
- SciPy
- Scikit-Learn

Typical error:

```text
An Application Control policy has blocked this file.
```

---

## Resolution

The issues were eventually resolved by:

1. Moving the project to a new development directory.
2. Creating a new virtual environment.
3. Switching from ARM64 Python to x64 Python.
4. Reinstalling all required libraries.
5. Registering a new Jupyter kernel.
6. Running the notebook inside the x64 environment.

Final environment:

```text
Python 3.14 (x64)
```

This allowed:

- PyTorch installation
- Transformers installation
- BERT execution
- TinyLlama execution

---

# Learning Outcomes

This project provided practical experience with:

- Text preprocessing
- Feature extraction
- Sentiment classification
- Naive Bayes
- Logistic Regression
- Support Vector Machines
- Word Embeddings
- BERT
- Prompt Engineering
- Large Language Models
- Environment troubleshooting
- Dependency management

---
