# 08. Natural Language Processing

## 🎯 Learning Objectives
- Master text preprocessing and feature extraction techniques
- Understand language models and word representations
- Learn modern NLP architectures (Transformers, BERT, GPT)
- Apply NLP to real-world text analysis problems

---

## 1. Introduction to Natural Language Processing

**Natural Language Processing (NLP)** is a branch of AI that helps computers understand, interpret, and manipulate human language.

### 1.1 What is NLP? 🟢

#### Core Challenges:
- **Ambiguity**: Words and sentences can have multiple meanings
- **Context dependency**: Meaning changes based on context
- **Variability**: Different ways to express the same idea
- **Idioms and metaphors**: Non-literal language use
- **Cultural and domain-specific knowledge**: Background knowledge required

#### NLP Pipeline:
```
Raw Text → Preprocessing → Feature Extraction → Model → Prediction/Analysis
```

#### Applications:
- **Machine Translation**: Google Translate, DeepL
- **Sentiment Analysis**: Social media monitoring, reviews
- **Chatbots**: Customer service, virtual assistants
- **Text Summarization**: News, documents, research papers
- **Information Extraction**: Named entities, relationships
- **Question Answering**: Search engines, knowledge bases

### 1.2 Text Preprocessing 🟢

#### Common Preprocessing Steps:

**Tokenization:**
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# NLTK tokenization
text = "Hello world! How are you today?"
words = word_tokenize(text)
sentences = sent_tokenize(text)

# spaCy tokenization
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
tokens = [token.text for token in doc]
```

**Lowercasing:**
```python
def lowercase_text(text):
    return text.lower()

text = "Hello World!"
print(lowercase_text(text))  # "hello world!"
```

**Removing Punctuation:**
```python
import string
import re

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punctuation_regex(text):
    return re.sub(r'[^\w\s]', '', text)

text = "Hello, world! How are you?"
print(remove_punctuation(text))  # "Hello world How are you"
```

**Stop Words Removal:**
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text, language='english'):
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

text = "This is a sample sentence with stop words"
print(remove_stopwords(text))
```

**Stemming and Lemmatization:**
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Stemming
stemmer = PorterStemmer()
def stem_text(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stems)

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmas)

text = "running runs ran"
print(stem_text(text))      # "run run ran"
print(lemmatize_text(text)) # "running run ran"
```

#### Complete Preprocessing Pipeline:
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        # Full preprocessing pipeline
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

# Usage
preprocessor = TextPreprocessor()
text = "The running dogs are quickly chasing the cats!"
processed = preprocessor.preprocess(text)
print(processed)  # ['running', 'dog', 'quickly', 'chasing', 'cat']
```

---

## 2. Feature Extraction and Representation

### 2.1 Bag of Words (BoW) 🟢

**Concept**: Represent text as frequency count of words, ignoring grammar and word order.

#### Implementation:
```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Sample documents
documents = [
    "I love machine learning",
    "Machine learning is great",
    "I love programming",
    "Programming is fun"
]

# Create BoW model
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

# Get feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()
print("Vocabulary:", feature_names)

# Convert to dense array for viewing
bow_array = bow_matrix.toarray()
print("BoW Matrix:")
print(bow_array)

# Create DataFrame for better visualization
import pandas as pd
bow_df = pd.DataFrame(bow_array, columns=feature_names)
print(bow_df)
```

#### From Scratch Implementation:
```python
class BagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.word_to_index = {}
    
    def fit(self, documents):
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        self.vocabulary = sorted(list(all_words))
        self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}
    
    def transform(self, documents):
        # Create feature matrix
        features = []
        for doc in documents:
            doc_vector = [0] * len(self.vocabulary)
            words = doc.lower().split()
            
            for word in words:
                if word in self.word_to_index:
                    doc_vector[self.word_to_index[word]] += 1
            
            features.append(doc_vector)
        
        return np.array(features)
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
```

### 2.2 TF-IDF (Term Frequency-Inverse Document Frequency) 🟢

**Concept**: Weight words by their importance - frequent in document but rare in corpus.

#### Mathematical Foundation:
```
TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in d)
IDF(t,D) = log(|D| / |{d ∈ D : t ∈ d}|)
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

#### Implementation:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample documents
documents = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "Cats and dogs are pets",
    "I love my pet cat"
]

# Create TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print("TF-IDF Matrix:")
print(tfidf_df)

# Find most important words for each document
for i, doc in enumerate(documents):
    doc_tfidf = tfidf_matrix[i].toarray().flatten()
    top_indices = doc_tfidf.argsort()[-3:][::-1]  # Top 3 words
    top_words = [(feature_names[idx], doc_tfidf[idx]) for idx in top_indices if doc_tfidf[idx] > 0]
    print(f"Document {i+1} top words: {top_words}")
```

#### From Scratch Implementation:
```python
import math
from collections import Counter

class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary = []
        self.idf_values = {}
    
    def fit(self, documents):
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        self.vocabulary = sorted(list(all_words))
        
        # Calculate IDF values
        N = len(documents)
        for word in self.vocabulary:
            containing_docs = sum(1 for doc in documents if word in doc.lower().split())
            self.idf_values[word] = math.log(N / containing_docs)
    
    def transform(self, documents):
        features = []
        
        for doc in documents:
            words = doc.lower().split()
            word_counts = Counter(words)
            doc_length = len(words)
            
            doc_vector = []
            for word in self.vocabulary:
                tf = word_counts.get(word, 0) / doc_length
                tfidf = tf * self.idf_values[word]
                doc_vector.append(tfidf)
            
            features.append(doc_vector)
        
        return np.array(features)
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
```

### 2.3 N-grams 🟡

**Concept**: Capture sequence information by considering consecutive words.

#### Implementation:
```python
from sklearn.feature_extraction.text import CountVectorizer

# Unigrams (1-gram)
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))

# Bigrams (2-gram)
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))

# Trigrams (3-gram)
trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))

# Combined n-grams
combined_vectorizer = CountVectorizer(ngram_range=(1, 3))

documents = ["I love machine learning", "Machine learning is great"]

# Extract n-grams
unigrams = unigram_vectorizer.fit_transform(documents)
bigrams = bigram_vectorizer.fit_transform(documents)

print("Unigram features:", unigram_vectorizer.get_feature_names_out())
print("Bigram features:", bigram_vectorizer.get_feature_names_out())
```

#### Character N-grams:
```python
# Character-level n-grams
char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
char_features = char_vectorizer.fit_transform(documents)
print("Character n-grams:", char_vectorizer.get_feature_names_out()[:20])
```

---

## 3. Word Embeddings

### 3.1 Word2Vec 🟡

**Concept**: Dense vector representations that capture semantic relationships.

#### Two Architectures:
- **CBOW (Continuous Bag of Words)**: Predict target word from context
- **Skip-gram**: Predict context words from target word

#### Using Pre-trained Word2Vec:
```python
import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np

# Load pre-trained Google News vectors
# word2vec_model = api.load("word2vec-google-news-300")

# For demo, let's train our own
sentences = [
    ["I", "love", "machine", "learning"],
    ["Machine", "learning", "is", "great"],
    ["Deep", "learning", "is", "powerful"],
    ["I", "enjoy", "programming"],
    ["Programming", "is", "fun"]
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector
vector = model.wv['machine']
print(f"Vector for 'machine': {vector[:10]}")  # Show first 10 dimensions

# Find similar words
similar_words = model.wv.most_similar('machine', topn=3)
print(f"Words similar to 'machine': {similar_words}")

# Word analogy: king - man + woman = queen
# result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
```

#### Training Word2Vec from Scratch:
```python
import numpy as np
from collections import defaultdict, Counter

class Word2VecSkipGram:
    def __init__(self, vector_size=100, window=5, min_count=1, learning_rate=0.025):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.vocab = {}
        self.word_to_index = {}
        self.index_to_word = {}
        
    def build_vocab(self, sentences):
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # Filter by min_count
        vocab_words = [word for word, count in word_counts.items() if count >= self.min_count]
        
        self.word_to_index = {word: i for i, word in enumerate(vocab_words)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.vocab_size = len(vocab_words)
        
        # Initialize weight matrices
        self.W1 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, 
                                   (self.vocab_size, self.vector_size))
        self.W2 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, 
                                   (self.vector_size, self.vocab_size))
    
    def generate_training_data(self, sentences):
        training_data = []
        for sentence in sentences:
            for i, center_word in enumerate(sentence):
                if center_word not in self.word_to_index:
                    continue
                    
                center_idx = self.word_to_index[center_word]
                
                # Get context words
                for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1)):
                    if i != j and sentence[j] in self.word_to_index:
                        context_idx = self.word_to_index[sentence[j]]
                        training_data.append((center_idx, context_idx))
        
        return training_data
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def train(self, sentences, epochs=1000):
        self.build_vocab(sentences)
        training_data = self.generate_training_data(sentences)
        
        for epoch in range(epochs):
            loss = 0
            for center_idx, context_idx in training_data:
                # Forward pass
                h = self.W1[center_idx]  # Hidden layer
                u = np.dot(h, self.W2)   # Output layer
                y_pred = self.softmax(u)
                
                # Calculate loss
                loss += -np.log(y_pred[context_idx])
                
                # Backward pass
                e = y_pred.copy()
                e[context_idx] -= 1
                
                # Update weights
                self.W2 -= self.learning_rate * np.outer(h, e)
                self.W1[center_idx] -= self.learning_rate * np.dot(self.W2, e)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss/len(training_data):.4f}")
    
    def get_word_vector(self, word):
        if word in self.word_to_index:
            return self.W1[self.word_to_index[word]]
        return None
```

### 3.2 GloVe (Global Vectors) 🟡

**Concept**: Combine global statistics with local context information.

#### Using Pre-trained GloVe:
```python
import numpy as np

def load_glove_vectors(file_path):
    """Load pre-trained GloVe vectors"""
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

# Usage (assuming you have GloVe file)
# glove_vectors = load_glove_vectors('glove.6B.100d.txt')
# vector = glove_vectors['machine']
```

### 3.3 FastText 🟡

**Concept**: Extension of Word2Vec that considers subword information.

#### Using FastText:
```python
from gensim.models import FastText

sentences = [
    ["I", "love", "machine", "learning"],
    ["Machine", "learning", "is", "great"],
    ["Deep", "learning", "is", "powerful"]
]

# Train FastText model
fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector (can handle out-of-vocabulary words)
vector = fasttext_model.wv['machine']
print(f"FastText vector for 'machine': {vector[:10]}")

# Handle out-of-vocabulary words
oov_vector = fasttext_model.wv['machinelearning']  # Compound word
print(f"OOV vector shape: {oov_vector.shape}")
```

---

## 4. Language Models

### 4.1 N-gram Language Models 🟢

**Concept**: Predict next word based on previous n-1 words.

#### Bigram Model Implementation:
```python
from collections import defaultdict, Counter
import random

class BigramLanguageModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocabulary = set()
    
    def train(self, sentences):
        for sentence in sentences:
            # Add start and end tokens
            words = ['<START>'] + sentence + ['<END>']
            
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocabulary.update([w1, w2])
    
    def probability(self, w1, w2):
        """Calculate P(w2|w1)"""
        if w1 not in self.bigram_counts:
            return 0.0
        return self.bigram_counts[w1][w2] / self.unigram_counts[w1]
    
    def generate_sentence(self, max_length=20):
        """Generate sentence using the model"""
        sentence = ['<START>']
        
        for _ in range(max_length):
            current_word = sentence[-1]
            if current_word == '<END>' or current_word not in self.bigram_counts:
                break
            
            # Sample next word based on probabilities
            next_words = list(self.bigram_counts[current_word].keys())
            weights = list(self.bigram_counts[current_word].values())
            
            next_word = random.choices(next_words, weights=weights)[0]
            sentence.append(next_word)
        
        return sentence[1:-1]  # Remove start and end tokens
    
    def perplexity(self, test_sentences):
        """Calculate perplexity on test data"""
        total_log_prob = 0
        total_words = 0
        
        for sentence in test_sentences:
            words = ['<START>'] + sentence + ['<END>']
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                prob = self.probability(w1, w2)
                if prob > 0:
                    total_log_prob += np.log(prob)
                else:
                    total_log_prob += np.log(1e-10)  # Smoothing
                total_words += 1
        
        return np.exp(-total_log_prob / total_words)

# Example usage
sentences = [
    ["I", "love", "machine", "learning"],
    ["Machine", "learning", "is", "great"],
    ["I", "love", "programming"],
    ["Programming", "is", "fun"]
]

model = BigramLanguageModel()
model.train(sentences)

# Generate sentences
for _ in range(3):
    generated = model.generate_sentence()
    print("Generated:", ' '.join(generated))
```

#### Trigram Model with Smoothing:
```python
class TrigramLanguageModel:
    def __init__(self, smoothing='laplace', alpha=1.0):
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocabulary = set()
        self.smoothing = smoothing
        self.alpha = alpha
    
    def train(self, sentences):
        for sentence in sentences:
            words = ['<START>', '<START>'] + sentence + ['<END>']
            
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i + 1], words[i + 2]
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocabulary.update([w1, w2, w3])
    
    def probability(self, w1, w2, w3):
        """Calculate P(w3|w1,w2) with smoothing"""
        if self.smoothing == 'laplace':
            numerator = self.trigram_counts[w1][w2][w3] + self.alpha
            denominator = self.bigram_counts[w1][w2] + self.alpha * len(self.vocabulary)
            return numerator / denominator
        else:
            if (w1, w2) not in [(k1, k2) for k1 in self.trigram_counts for k2 in self.trigram_counts[k1]]:
                return 0.0
            return self.trigram_counts[w1][w2][w3] / self.bigram_counts[w1][w2]
```

### 4.2 Neural Language Models 🟡

**Concept**: Use neural networks to model language, capturing long-range dependencies.

#### Simple Feed-Forward Language Model:
```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(NeuralLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        embeds = self.embedding(inputs).view(inputs.size(0), -1)
        hidden = self.relu(self.linear1(embeds))
        output = self.linear2(hidden)
        return output

# Prepare data
def create_training_data(sentences, word_to_idx, context_size=3):
    data = []
    for sentence in sentences:
        indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sentence]
        for i in range(context_size, len(indices)):
            context = indices[i-context_size:i]
            target = indices[i]
            data.append((context, target))
    return data

# Training function
def train_neural_lm(model, training_data, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in training_data:
            context_tensor = torch.LongTensor([context])
            target_tensor = torch.LongTensor([target])
            
            optimizer.zero_grad()
            output = model(context_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(training_data):.4f}")
```

---

## 5. Sentiment Analysis

### 5.1 Rule-Based Sentiment Analysis 🟢

**Concept**: Use predefined dictionaries and rules to determine sentiment.

#### Simple Lexicon-Based Approach:
```python
class LexiconSentimentAnalyzer:
    def __init__(self):
        # Simple sentiment lexicon
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'sad', 'angry', 'disappointed', 'frustrated', 'annoyed'
        }
        
        self.negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere'}
    
    def preprocess(self, text):
        return text.lower().split()
    
    def analyze_sentiment(self, text):
        words = self.preprocess(text)
        positive_score = 0
        negative_score = 0
        negate = False
        
        for word in words:
            if word in self.negation_words:
                negate = True
                continue
            
            if word in self.positive_words:
                if negate:
                    negative_score += 1
                else:
                    positive_score += 1
                negate = False
            elif word in self.negative_words:
                if negate:
                    positive_score += 1
                else:
                    negative_score += 1
                negate = False
            else:
                negate = False
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def get_sentiment_score(self, text):
        words = self.preprocess(text)
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0
        
        return (positive_score - negative_score) / total_words

# Usage
analyzer = LexiconSentimentAnalyzer()
texts = [
    "I love this movie, it's amazing!",
    "This product is terrible and awful.",
    "The weather is okay today.",
    "I don't like this at all."
]

for text in texts:
    sentiment = analyzer.analyze_sentiment(text)
    score = analyzer.get_sentiment_score(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}, Score: {score:.2f}\n")
```

#### Using VADER Sentiment:
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

texts = [
    "I love this movie!",
    "This is terrible.",
    "The movie was okay.",
    "I REALLY hate this!!!",
    "Not bad at all."
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Positive: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}, "
          f"Neutral: {scores['neu']:.2f}, Compound: {scores['compound']:.2f}\n")
```

### 5.2 Machine Learning Sentiment Analysis 🟡

#### Naive Bayes Classifier:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class MLSentimentAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
    
    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)
    
    def predict(self, texts):
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

# Example with sample data
sample_data = [
    ("I love this product", "positive"),
    ("This is amazing", "positive"),
    ("Great quality", "positive"),
    ("I hate this", "negative"),
    ("Terrible experience", "negative"),
    ("Worst product ever", "negative"),
    ("It's okay", "neutral"),
    ("Average quality", "neutral")
]

texts, labels = zip(*sample_data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Train model
ml_analyzer = MLSentimentAnalyzer()
ml_analyzer.train(X_train, y_train)

# Make predictions
predictions = ml_analyzer.predict(X_test)
probabilities = ml_analyzer.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

#### Deep Learning Sentiment Analysis:
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        words = text.lower().split()
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return torch.LongTensor(indices), torch.LongTensor([label])

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2):
        super(LSTMSentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state
        final_hidden = hidden[-1]
        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)
        
        return output

# Training function
def train_sentiment_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels.squeeze())
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels.squeeze()).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {100*correct/total:.2f}%")
```

---

## 6. Named Entity Recognition (NER)

### 6.1 Rule-Based NER 🟢

**Concept**: Use patterns and dictionaries to identify named entities.

#### Simple Rule-Based NER:
```python
import re
from collections import defaultdict

class RuleBasedNER:
    def __init__(self):
        # Simple entity patterns
        self.patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # FirstName LastName
                r'\b(?:Mr|Mrs|Ms|Dr)\.? [A-Z][a-z]+\b'  # Title Name
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'PHONE': [
                r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
                r'\(\d{3}\) \d{3}-\d{4}\b'  # (123) 456-7890
            ],
            'DATE': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b'  # YYYY-MM-DD
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'  # $1,000.00
            ]
        }
        
        # Entity dictionaries
        self.entity_dicts = {
            'PERSON': {'John Smith', 'Jane Doe', 'Alice Johnson'},
            'ORGANIZATION': {'Google', 'Microsoft', 'Apple', 'Facebook'},
            'LOCATION': {'New York', 'California', 'London', 'Paris'}
        }
    
    def extract_entities(self, text):
        entities = defaultdict(list)
        
        # Pattern-based extraction
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities[entity_type].append({
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Dictionary-based extraction
        for entity_type, entity_dict in self.entity_dicts.items():
            for entity in entity_dict:
                if entity in text:
                    start = text.find(entity)
                    entities[entity_type].append({
                        'text': entity,
                        'start': start,
                        'end': start + len(entity)
                    })
        
        return dict(entities)

# Usage
ner = RuleBasedNER()
text = "John Smith works at Google and can be reached at john@gmail.com or 123-456-7890. The meeting is on 2023-12-25."

entities = ner.extract_entities(text)
for entity_type, entity_list in entities.items():
    print(f"{entity_type}: {entity_list}")
```

### 6.2 Machine Learning NER 🟡

#### Using spaCy for NER:
```python
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random

# Load pre-trained model
nlp = spacy.load("en_core_web_sm")

def extract_entities_spacy(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'description': spacy.explain(ent.label_)
        })
    
    return entities

# Example usage
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = extract_entities_spacy(text)

for entity in entities:
    print(f"Entity: {entity['text']}, Label: {entity['label']}, "
          f"Description: {entity['description']}")
```

#### Training Custom NER Model:
```python
def train_custom_ner(training_data, iterations=100):
    # Create blank English model
    nlp = spacy.blank("en")
    
    # Add NER pipeline component
    ner = nlp.add_pipe("ner")
    
    # Add labels to NER
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Disable other pipelines during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for iteration in range(iterations):
            random.shuffle(training_data)
            losses = {}
            
            # Batch training data
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                nlp.update(examples, drop=0.5, losses=losses)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Losses: {losses}")
    
    return nlp

# Example training data format
training_data = [
    ("Apple is a technology company.", {"entities": [(0, 5, "ORG")]}),
    ("Steve Jobs founded Apple.", {"entities": [(0, 10, "PERSON"), (18, 23, "ORG")]}),
    ("The meeting is in New York.", {"entities": [(18, 26, "GPE")]})
]

# Train model
# custom_nlp = train_custom_ner(training_data)
```

---

## 7. Modern NLP with Transformers

### 7.1 BERT (Bidirectional Encoder Representations from Transformers) 🔴

**Concept**: Pre-trained transformer model that understands context bidirectionally.

#### Using BERT with Transformers Library:
```python
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import pipeline
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_bert_embeddings(text):
    # Tokenize text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden states
    last_hidden_states = outputs.last_hidden_state
    
    # Average pooling
    embeddings = torch.mean(last_hidden_states, dim=1)
    
    return embeddings

# Example usage
text = "The quick brown fox jumps over the lazy dog."
embeddings = get_bert_embeddings(text)
print(f"BERT embeddings shape: {embeddings.shape}")

# Use BERT for classification
classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
result = classifier("I love this movie!")
print(f"Sentiment analysis result: {result}")
```

#### Fine-tuning BERT for Classification:
```python
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def prepare_data(self, texts, labels=None, max_length=128):
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        dataset_inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
        
        if labels is not None:
            dataset_inputs['labels'] = torch.tensor(labels)
        
        return TensorDataset(*dataset_inputs.values())
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              epochs=3, batch_size=16, learning_rate=2e-5):
        
        train_dataset = self.prepare_data(train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def predict(self, texts):
        dataset = self.prepare_data(texts)
        loader = DataLoader(dataset, batch_size=16)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)
                predictions.extend(predicted_labels.cpu().numpy())
        
        return predictions

# Example usage
# classifier = BERTClassifier(num_labels=3)  # positive, negative, neutral
# classifier.train(train_texts, train_labels, epochs=3)
# predictions = classifier.predict(test_texts)
```

### 7.2 GPT (Generative Pre-trained Transformer) 🔴

**Concept**: Autoregressive language model for text generation.

#### Using GPT for Text Generation:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load pre-trained GPT-2 model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7, num_return_sequences=1):
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decode generated text
    generated_texts = []
    for sequence in output:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

# Example usage
prompt = "The future of artificial intelligence is"
generated = generate_text(prompt, max_length=50, num_return_sequences=3)

for i, text in enumerate(generated):
    print(f"Generation {i+1}: {text}\n")

# Using pipeline for easier text generation
generator = pipeline('text-generation', model='gpt2')
result = generator("Once upon a time", max_length=50, num_return_sequences=2)

for story in result:
    print(story['generated_text'])
```

### 7.3 T5 (Text-to-Text Transfer Transformer) 🔴

**Concept**: Treats all NLP tasks as text-to-text problems.

#### Using T5 for Various Tasks:
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Model:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def generate_text(self, input_text, max_length=100):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def summarize(self, text):
        input_text = f"summarize: {text}"
        return self.generate_text(input_text)
    
    def translate(self, text, target_language='German'):
        input_text = f"translate English to {target_language}: {text}"
        return self.generate_text(input_text)
    
    def answer_question(self, question, context):
        input_text = f"question: {question} context: {context}"
        return self.generate_text(input_text)

# Example usage
t5_model = T5Model()

# Summarization
long_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

summary = t5_model.summarize(long_text)
print(f"Summary: {summary}")

# Question answering
context = "The capital of France is Paris. It is known for the Eiffel Tower."
question = "What is the capital of France?"
answer = t5_model.answer_question(question, context)
print(f"Answer: {answer}")
```

---

## 🎯 Key Takeaways

### Task-Based NLP Guide:

#### For Text Classification:
- **Simple tasks**: TF-IDF + Logistic Regression/SVM
- **Complex tasks**: BERT, RoBERTa fine-tuning
- **Large scale**: DistilBERT for efficiency

#### For Text Generation:
- **Simple completion**: GPT-2, GPT-3
- **Conditional generation**: T5, BART
- **Dialogue**: DialoGPT, BlenderBot

#### For Information Extraction:
- **Named entities**: spaCy NER, BERT-NER
- **Relations**: Dependency parsing + rules
- **Events**: Transformer-based models

#### For Language Understanding:
- **Similarity**: Sentence transformers
- **Question answering**: BERT, RoBERTa, T5
- **Language modeling**: GPT family, BERT family

### Best Practices:
1. **Start with pre-trained models**: Transfer learning is powerful
2. **Data quality matters**: Clean, relevant data beats complex models
3. **Evaluation is crucial**: Use appropriate metrics for your task
4. **Consider computational constraints**: Balance performance vs. efficiency
5. **Domain adaptation**: Fine-tune on domain-specific data
6. **Handle multiple languages**: Use multilingual models when needed

---

## 📚 Next Steps

Continue your NLP journey with:
- **[Computer Vision](09_Computer_Vision.md)** - Learn about image processing and analysis
- **[Time Series Analysis](10_Time_Series_Analysis.md)** - Temporal data analysis techniques

---

## 🛠️ Practical Exercises

### Exercise 1: Build Text Classifier
1. Collect text data from social media/reviews
2. Implement both traditional ML and BERT approaches
3. Compare performance and analyze errors
4. Deploy as web API

### Exercise 2: Create Chatbot
1. Design conversation flow
2. Implement intent classification
3. Add entity extraction
4. Use pre-trained language model for responses

### Exercise 3: Text Summarization System
1. Implement extractive summarization
2. Try abstractive summarization with T5
3. Evaluate using ROUGE metrics
4. Create web interface

---

*Next: [Computer Vision →](09_Computer_Vision.md)*
