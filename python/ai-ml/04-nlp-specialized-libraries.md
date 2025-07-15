# Natural Language Processing and Specialized Libraries

## Table of Contents
1. [NLTK - Natural Language Toolkit](#nltk---natural-language-toolkit)
2. [spaCy - Industrial NLP](#spacy---industrial-nlp)
3. [Transformers - Hugging Face](#transformers---hugging-face)
4. [Gensim - Topic Modeling](#gensim---topic-modeling)
5. [OpenCV - Computer Vision](#opencv---computer-vision)
6. [Plotly - Interactive Visualization](#plotly---interactive-visualization)

## NLTK - Natural Language Toolkit

NLTK is a comprehensive natural language processing library with tools for classification, tokenization, stemming, tagging, parsing, and more.

### 1. Installation and Setup
```python
# Installation
pip install nltk

import nltk
import string
import re
from collections import Counter
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

print(f"NLTK version: {nltk.__version__}")
```

### 2. Text Preprocessing
```python
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Sample text
text = """
Natural Language Processing (NLP) is a fascinating field that combines 
computer science, artificial intelligence, and linguistics. It enables 
computers to understand, interpret, and generate human language in a 
valuable way. Applications include machine translation, sentiment analysis, 
and chatbots.
"""

# Tokenization
print("=== Tokenization ===")
sentences = sent_tokenize(text)
words = word_tokenize(text)

print(f"Number of sentences: {len(sentences)}")
print(f"Number of words: {len(words)}")
print(f"First sentence: {sentences[0]}")
print(f"First 10 words: {words[:10]}")

# Cleaning text
def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

cleaned_text = clean_text(text)
print(f"\nCleaned text: {cleaned_text[:100]}...")

# Remove stopwords
stop_words = set(stopwords.words('english'))
words_no_stop = [word for word in word_tokenize(cleaned_text) if word not in stop_words]

print(f"\nWords without stopwords: {words_no_stop[:15]}")

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words_no_stop]

print(f"\nStemmed words: {stemmed_words[:15]}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_no_stop]

print(f"\nLemmatized words: {lemmatized_words[:15]}")

# Word frequency analysis
word_freq = Counter(words_no_stop)
print(f"\nMost common words: {word_freq.most_common(10)}")
```

### 3. Part-of-Speech Tagging and Named Entity Recognition
```python
# Part-of-speech tagging
sample_sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sample_sentence)
pos_tags = pos_tag(tokens)

print("=== Part-of-Speech Tagging ===")
for word, pos in pos_tags:
    print(f"{word}: {pos}")

# Named Entity Recognition
print("\n=== Named Entity Recognition ===")
ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
ner_tokens = word_tokenize(ner_text)
ner_pos = pos_tag(ner_tokens)
ner_chunks = ne_chunk(ner_pos)

print(ner_chunks)

# Extract named entities
def extract_entities(chunked):
    """Extract named entities from chunked text"""
    entities = []
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entity = ' '.join([token for token, pos in chunk.leaves()])
            entities.append((entity, chunk.label()))
    return entities

entities = extract_entities(ner_chunks)
print(f"\nExtracted entities: {entities}")
```

### 4. Sentiment Analysis
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    scores = analyzer.polarity_scores(text)
    
    # Determine overall sentiment
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'sentiment': sentiment,
        'scores': scores
    }

# Test sentiment analysis
test_sentences = [
    "I love this product! It's amazing!",
    "This is the worst experience ever.",
    "The weather is okay today.",
    "The movie was fantastic and entertaining!",
    "I hate waiting in long lines."
]

print("=== Sentiment Analysis ===")
for sentence in test_sentences:
    result = analyze_sentiment(sentence)
    print(f"Text: {sentence}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Scores: {result['scores']}")
    print("-" * 50)
```

### 5. Text Classification
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np

# Sample dataset (normally you'd load from a file)
documents = [
    ("This movie is excellent! Great acting and plot.", "positive"),
    ("I loved the storyline and characters.", "positive"),
    ("Amazing cinematography and direction.", "positive"),
    ("Best film I've seen this year!", "positive"),
    ("Brilliant performance by the lead actor.", "positive"),
    ("Terrible movie with poor acting.", "negative"),
    ("Worst film ever made.", "negative"),
    ("Boring and predictable plot.", "negative"),
    ("Complete waste of time.", "negative"),
    ("Awful script and direction.", "negative"),
    ("The movie was okay, nothing special.", "neutral"),
    ("Average film with decent acting.", "neutral"),
    ("It was watchable but forgettable.", "neutral"),
    ("Not bad, not great either.", "neutral"),
    ("The film has its moments.", "neutral"),
]

# Prepare data
texts = [doc[0] for doc in documents]
labels = [doc[1] for doc in documents]

# Text preprocessing function
def preprocess_text(text):
    """Preprocess text for classification"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Preprocess all texts
processed_texts = [preprocess_text(text) for text in texts]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate
print("=== Text Classification Results ===")
print(classification_report(y_test, y_pred))

# Predict new text
def predict_sentiment(text):
    """Predict sentiment of new text"""
    processed = preprocess_text(text)
    tfidf = vectorizer.transform([processed])
    prediction = classifier.predict(tfidf)[0]
    probability = max(classifier.predict_proba(tfidf)[0])
    
    return prediction, probability

# Test with new examples
new_texts = [
    "This is an outstanding movie with great performances!",
    "Completely disappointed with this film.",
    "The movie was just average."
]

for text in new_texts:
    pred, prob = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted: {pred} (confidence: {prob:.3f})")
    print()
```

## spaCy - Industrial NLP

spaCy is a fast and efficient NLP library designed for production use.

### 1. Installation and Setup
```python
# Installation
pip install spacy
python -m spacy download en_core_web_sm

import spacy
import matplotlib.pyplot as plt
from collections import Counter

# Load English model
nlp = spacy.load("en_core_web_sm")

print(f"spaCy version: {spacy.__version__}")
print(f"Model: {nlp.meta['name']} v{nlp.meta['version']}")
```

### 2. Text Processing with spaCy
```python
# Process text
text = """
Apple Inc. is planning to build a new headquarters in Austin, Texas. 
The company's CEO, Tim Cook, announced this during a press conference 
on January 15th, 2024. The project will cost approximately $1 billion 
and create 5,000 new jobs.
"""

doc = nlp(text)

# Token analysis
print("=== Token Analysis ===")
for token in doc:
    print(f"Text: {token.text:<15} "
          f"Lemma: {token.lemma_:<15} "
          f"POS: {token.pos_:<8} "
          f"Tag: {token.tag_:<8} "
          f"Shape: {token.shape_:<10} "
          f"Alpha: {token.is_alpha}")

# Named Entity Recognition
print("\n=== Named Entities ===")
for ent in doc.ents:
    print(f"Entity: {ent.text:<20} Label: {ent.label_:<15} Description: {spacy.explain(ent.label_)}")

# Dependency parsing
print("\n=== Dependency Parsing ===")
for token in doc:
    print(f"Text: {token.text:<15} "
          f"Dependency: {token.dep_:<15} "
          f"Head: {token.head.text:<15} "
          f"Children: {[child.text for child in token.children]}")

# Sentence segmentation
print("\n=== Sentences ===")
for i, sent in enumerate(doc.sents):
    print(f"Sentence {i+1}: {sent.text.strip()}")
```

### 3. Advanced NLP with spaCy
```python
# Custom entity ruler
from spacy.pipeline import EntityRuler

# Add custom entities
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "COMPANY", "pattern": "OpenAI"},
    {"label": "COMPANY", "pattern": "Microsoft"},
    {"label": "PRODUCT", "pattern": "ChatGPT"},
    {"label": "PRODUCT", "pattern": "GPT-4"},
]
ruler.add_patterns(patterns)

# Test custom entities
custom_text = "OpenAI developed ChatGPT using GPT-4 technology in partnership with Microsoft."
custom_doc = nlp(custom_text)

print("=== Custom Named Entities ===")
for ent in custom_doc.ents:
    print(f"Entity: {ent.text:<15} Label: {ent.label_}")

# Text similarity
text1 = "I love machine learning and artificial intelligence."
text2 = "Machine learning and AI are fascinating fields."
text3 = "I enjoy cooking and gardening."

doc1 = nlp(text1)
doc2 = nlp(text2)
doc3 = nlp(text3)

print("\n=== Text Similarity ===")
print(f"Text 1 vs Text 2: {doc1.similarity(doc2):.3f}")
print(f"Text 1 vs Text 3: {doc1.similarity(doc3):.3f}")
print(f"Text 2 vs Text 3: {doc2.similarity(doc3):.3f}")

# Word vectors
print("\n=== Word Vectors ===")
words = ["king", "queen", "man", "woman", "computer", "technology"]
for word in words:
    token = nlp(word)[0]
    if token.has_vector:
        print(f"Word: {word:<10} Vector shape: {token.vector.shape}")

# Find similar words
def find_similar_words(word, n=5):
    """Find similar words using word vectors"""
    word_doc = nlp(word)
    if not word_doc[0].has_vector:
        return []
    
    # This is a simplified approach - in practice, you'd use a larger vocabulary
    test_words = ["king", "queen", "man", "woman", "prince", "princess", 
                  "boy", "girl", "computer", "technology", "science", "data"]
    
    similarities = []
    for test_word in test_words:
        if test_word != word:
            test_doc = nlp(test_word)
            if test_doc[0].has_vector:
                similarity = word_doc.similarity(test_doc)
                similarities.append((test_word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]

similar_to_king = find_similar_words("king")
print(f"\nWords similar to 'king': {similar_to_king}")
```

### 4. Information Extraction
```python
# Extract key information from text
def extract_information(text):
    """Extract key information from text"""
    doc = nlp(text)
    
    info = {
        'people': [],
        'organizations': [],
        'locations': [],
        'money': [],
        'dates': [],
        'quantities': []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            info['people'].append(ent.text)
        elif ent.label_ in ["ORG", "COMPANY"]:
            info['organizations'].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            info['locations'].append(ent.text)
        elif ent.label_ == "MONEY":
            info['money'].append(ent.text)
        elif ent.label_ == "DATE":
            info['dates'].append(ent.text)
        elif ent.label_ in ["QUANTITY", "CARDINAL"]:
            info['quantities'].append(ent.text)
    
    return info

# Test information extraction
news_text = """
Elon Musk's company Tesla announced record quarterly profits of $2.3 billion 
for Q3 2023. The electric vehicle manufacturer, based in Austin, Texas, 
delivered 435,000 vehicles in the quarter. SpaceX, another company founded 
by Musk, launched 25 rockets in the same period.
"""

extracted_info = extract_information(news_text)
print("=== Extracted Information ===")
for key, values in extracted_info.items():
    if values:
        print(f"{key.capitalize()}: {list(set(values))}")  # Remove duplicates

# Text classification with spaCy
from spacy.training import Example
import random

# Create a simple text classifier
@spacy.Language.component("sentiment_classifier")
def sentiment_classifier(doc):
    """Simple rule-based sentiment classifier"""
    positive_words = {"good", "great", "excellent", "amazing", "love", "best", "fantastic"}
    negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting"}
    
    positive_count = sum(1 for token in doc if token.lemma_.lower() in positive_words)
    negative_count = sum(1 for token in doc if token.lemma_.lower() in negative_words)
    
    if positive_count > negative_count:
        doc._.sentiment = "positive"
    elif negative_count > positive_count:
        doc._.sentiment = "negative"
    else:
        doc._.sentiment = "neutral"
    
    return doc

# Register custom attribute
spacy.tokens.Doc.set_extension("sentiment", default=None, force=True)

# Add component to pipeline
nlp.add_pipe("sentiment_classifier")

# Test sentiment classification
test_sentences = [
    "This is an excellent product!",
    "I hate this terrible service.",
    "The weather is okay today."
]

for sentence in test_sentences:
    doc = nlp(sentence)
    print(f"Text: {sentence}")
    print(f"Sentiment: {doc._.sentiment}")
    print()
```

## Transformers - Hugging Face

Hugging Face Transformers provides state-of-the-art pre-trained models for NLP tasks.

### 1. Installation and Basic Usage
```python
# Installation
pip install transformers torch

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
)
import torch
import numpy as np

print("=== Transformers Library ===")

# Using pipelines (easiest way)
# Sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
]

print("=== Sentiment Analysis Pipeline ===")
for text in texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    print()
```

### 2. Named Entity Recognition with Transformers
```python
# NER pipeline
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

ner_text = """
Elon Musk is the CEO of Tesla and SpaceX. He was born in South Africa 
and later moved to the United States. Tesla is headquartered in Austin, Texas.
"""

print("=== Named Entity Recognition ===")
ner_results = ner_pipeline(ner_text)
for entity in ner_results:
    print(f"Entity: {entity['word']:<15} "
          f"Label: {entity['entity_group']:<10} "
          f"Score: {entity['score']:.3f}")
```

### 3. Text Generation
```python
# Text generation pipeline
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"

print("=== Text Generation ===")
generated_texts = generator(
    prompt,
    max_length=100,
    num_return_sequences=2,
    temperature=0.7,
    do_sample=True
)

for i, text in enumerate(generated_texts):
    print(f"Generated text {i+1}:")
    print(text['generated_text'])
    print()
```

### 4. Question Answering
```python
# Question answering pipeline
qa_pipeline = pipeline("question-answering")

context = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, 
in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": 
any device that perceives its environment and takes actions that maximize 
its chance of successfully achieving its goals. Machine learning is a subset 
of AI that enables machines to learn from data without being explicitly programmed.
"""

questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "How do intelligent agents work?"
]

print("=== Question Answering ===")
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.3f}")
    print()
```

### 5. Custom Model Usage
```python
# Load pre-trained BERT model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_bert_embeddings(text):
    """Get BERT embeddings for text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings

# Get embeddings for sample texts
sample_texts = [
    "I love machine learning",
    "Machine learning is fascinating",
    "I enjoy cooking"
]

print("=== BERT Embeddings ===")
embeddings = []
for text in sample_texts:
    embedding = get_bert_embeddings(text)
    embeddings.append(embedding[0])
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")

# Calculate similarity between embeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.array(embeddings)
similarity_matrix = cosine_similarity(embeddings)

print("\n=== Similarity Matrix ===")
for i, text1 in enumerate(sample_texts):
    for j, text2 in enumerate(sample_texts):
        if i != j:
            print(f"{text1} vs {text2}: {similarity_matrix[i][j]:.3f}")
```

## Gensim - Topic Modeling

Gensim is a library for topic modeling, document similarity analysis, and other NLP tasks.

### 1. Installation and Basic Setup
```python
# Installation
pip install gensim

import gensim
from gensim import corpora, models
from gensim.models import LdaModel, Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from collections import defaultdict

print(f"Gensim version: {gensim.__version__}")
```

### 2. Topic Modeling with LDA
```python
# Sample documents
documents = [
    "Machine learning algorithms can learn from data without explicit programming",
    "Deep learning is a subset of machine learning that uses neural networks",
    "Natural language processing helps computers understand human language",
    "Computer vision enables machines to interpret visual information",
    "Data science combines statistics programming and domain expertise",
    "Artificial intelligence aims to create intelligent machines",
    "Python is a popular programming language for data science",
    "Statistics is fundamental to data analysis and machine learning",
    "Big data requires specialized tools and techniques for processing",
    "Cloud computing provides scalable infrastructure for AI applications"
]

# Preprocess documents
def preprocess_documents(documents):
    """Preprocess documents for topic modeling"""
    processed_docs = []
    stopwords = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    for doc in documents:
        # Convert to lowercase and split
        tokens = doc.lower().split()
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
        processed_docs.append(tokens)
    
    return processed_docs

processed_docs = preprocess_documents(documents)

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

print("=== Topic Modeling with LDA ===")
print(f"Dictionary size: {len(dictionary)}")
print(f"Corpus size: {len(corpus)}")

# Train LDA model
num_topics = 3
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# Print topics
print("\n=== Discovered Topics ===")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx}: {topic}")

# Get topic distribution for documents
print("\n=== Document Topic Distributions ===")
for i, doc in enumerate(processed_docs):
    doc_bow = dictionary.doc2bow(doc)
    doc_topics = lda_model[doc_bow]
    print(f"Document {i}: {' '.join(doc[:5])}...")
    for topic_id, prob in doc_topics:
        print(f"  Topic {topic_id}: {prob:.3f}")
    print()
```

### 3. Word2Vec Word Embeddings
```python
# Prepare sentences for Word2Vec
sentences = [doc for doc in processed_docs]

# Train Word2Vec model
print("=== Word2Vec Training ===")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # Embedding dimension
    window=5,           # Context window size
    min_count=1,        # Minimum word frequency
    workers=4,          # Number of CPU cores
    sg=1               # Skip-gram (1) or CBOW (0)
)

# Get word vectors
print("\n=== Word Vectors ===")
vocab_words = list(w2v_model.wv.key_to_index.keys())
print(f"Vocabulary size: {len(vocab_words)}")
print(f"Sample words: {vocab_words[:10]}")

# Find similar words
if 'machine' in w2v_model.wv:
    similar_words = w2v_model.wv.most_similar('machine', topn=5)
    print(f"\nWords similar to 'machine': {similar_words}")

if 'data' in w2v_model.wv:
    similar_words = w2v_model.wv.most_similar('data', topn=5)
    print(f"Words similar to 'data': {similar_words}")

# Word analogies (if enough data)
try:
    # king - man + woman = queen (classic example)
    if all(word in w2v_model.wv for word in ['learning', 'machine', 'data']):
        analogy = w2v_model.wv.most_similar(
            positive=['learning', 'data'], 
            negative=['machine'], 
            topn=3
        )
        print(f"\nAnalogy (learning + data - machine): {analogy}")
except:
    print("Not enough data for meaningful analogies")
```

### 4. Doc2Vec Document Embeddings
```python
# Prepare tagged documents for Doc2Vec
tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(processed_docs)]

# Train Doc2Vec model
print("=== Doc2Vec Training ===")
d2v_model = Doc2Vec(
    documents=tagged_docs,
    vector_size=50,
    window=2,
    min_count=1,
    workers=4,
    epochs=20
)

# Get document vectors
print("\n=== Document Vectors ===")
for i in range(len(documents)):
    doc_vector = d2v_model.dv[str(i)]
    print(f"Document {i} vector shape: {doc_vector.shape}")

# Find similar documents
print("\n=== Similar Documents ===")
for i in [0, 1, 2]:  # Check similarity for first 3 documents
    similar_docs = d2v_model.dv.most_similar([str(i)], topn=3)
    print(f"Document {i}: {documents[i][:50]}...")
    print(f"Similar documents: {similar_docs}")
    print()

# Infer vector for new document
new_doc = "artificial intelligence and machine learning are transforming technology"
new_doc_tokens = new_doc.lower().split()
new_vector = d2v_model.infer_vector(new_doc_tokens)
print(f"New document vector shape: {new_vector.shape}")

# Find similar documents to new document
similar_to_new = d2v_model.dv.most_similar([new_vector], topn=3)
print(f"Documents similar to new document: {similar_to_new}")
```

## 7. Advanced NLP Techniques

### Custom Named Entity Recognition with spaCy

```python
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import random

# Create custom NER model
def create_custom_ner_model():
    """Create custom NER model for specific domains"""
    
    # Training data format: (text, {"entities": [(start, end, label)]})
    TRAIN_DATA = [
        ("Apple Inc. is developing new AI technologies", {"entities": [(0, 9, "COMPANY"), (25, 27, "TECH")]}),
        ("Google released TensorFlow for machine learning", {"entities": [(0, 6, "COMPANY"), (16, 26, "TECH")]}),
        ("Microsoft Azure provides cloud computing services", {"entities": [(0, 9, "COMPANY"), (10, 15, "TECH")]}),
        ("OpenAI created GPT-3 for natural language processing", {"entities": [(0, 6, "COMPANY"), (15, 20, "TECH")]}),
        ("Facebook developed PyTorch deep learning framework", {"entities": [(0, 8, "COMPANY"), (19, 26, "TECH")]}),
    ]
    
    # Create blank model
    nlp = spacy.blank("en")
    
    # Add NER pipe
    ner = nlp.add_pipe("ner")
    
    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Training
    nlp.begin_training()
    
    for iteration in range(30):
        random.shuffle(TRAIN_DATA)
        losses = {}
        
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], losses=losses, drop=0.5)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Losses: {losses}")
    
    return nlp

# Train custom NER model
print("=== Training Custom NER Model ===")
custom_nlp = create_custom_ner_model()

# Test custom model
test_text = "Amazon Web Services and IBM Watson are leading AI platforms"
doc = custom_nlp(test_text)

print(f"\n=== Custom NER Results ===")
print(f"Text: {test_text}")
for ent in doc.ents:
    print(f"Entity: '{ent.text}' -> Label: {ent.label_}")
```

### Advanced Text Classification with Transformers

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class TextClassificationDataset(Dataset):
    """Custom dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_bert_classifier():
    """Fine-tune BERT for custom text classification"""
    
    # Sample data (sentiment analysis)
    texts = [
        "I love this product, it's amazing!",
        "This is the worst service ever.",
        "The movie was okay, not great but not bad.",
        "Excellent quality and fast delivery!",
        "I hate waiting in long lines.",
        "The weather is nice today.",
        "Outstanding customer support!",
        "This doesn't work as expected."
    ]
    
    labels = [1, 0, 1, 1, 0, 1, 1, 0]  # 1: positive, 0: negative
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    
    # Create datasets
    train_dataset = TextClassificationDataset(texts, labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # In practice, use separate eval set
        tokenizer=tokenizer
    )
    
    # Train model
    print("=== Fine-tuning BERT ===")
    trainer.train()
    
    return model, tokenizer

# Fine-tune BERT (commented out for demo)
# bert_model, bert_tokenizer = fine_tune_bert_classifier()

# Use pre-trained sentiment analysis pipeline
print("=== Pre-trained Sentiment Analysis ===")
sentiment_pipeline = pipeline("sentiment-analysis", 
                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")

test_texts = [
    "I love this new technology!",
    "This product is terrible.",
    "The service was okay."
]

for text in test_texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']} (Score: {result[0]['score']:.3f})")
    print()
```

### Advanced Topic Modeling with LDA and BERTopic

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import pandas as pd

def advanced_topic_modeling():
    """Compare LDA and BERTopic for topic modeling"""
    
    # Sample documents
    documents = [
        "Machine learning algorithms are transforming healthcare",
        "Deep neural networks excel at image recognition tasks",
        "Natural language processing enables chatbots and translation",
        "Computer vision applications include autonomous vehicles",
        "Reinforcement learning is used in game playing AI",
        "Big data analytics helps businesses make decisions",
        "Cloud computing provides scalable infrastructure",
        "Blockchain technology ensures secure transactions",
        "Internet of Things connects smart devices",
        "Cybersecurity protects against digital threats",
        "Artificial intelligence automates repetitive tasks",
        "Data science extracts insights from large datasets",
        "Software engineering builds reliable applications",
        "Web development creates interactive user interfaces",
        "Mobile apps provide convenient user experiences"
    ]
    
    print("=== Traditional LDA Topic Modeling ===")
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # LDA Model
    lda = LatentDirichletAllocation(
        n_components=3,
        random_state=42,
        max_iter=10
    )
    
    lda.fit(doc_term_matrix)
    
    # Display topics
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    print("\n=== BERTopic (Advanced) ===")
    
    # BERTopic with custom settings
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit model
    topics, probs = topic_model.fit_transform(documents)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print("Topic Information:")
    print(topic_info)
    
    # Get representative documents
    print("\nRepresentative Documents per Topic:")
    for topic_id in topic_info['Topic'][:3]:  # Show first 3 topics
        if topic_id != -1:  # Skip outlier topic
            topic_docs = topic_model.get_representative_docs(topic_id)
            print(f"\nTopic {topic_id}:")
            for doc in topic_docs[:2]:  # Show top 2 docs
                print(f"  - {doc}")
    
    return topic_model

# Run advanced topic modeling
topic_model = advanced_topic_modeling()
```

### Text Summarization and Question Answering

```python
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

def advanced_text_summarization():
    """Advanced text summarization techniques"""
    
    # Sample long text
    long_text = """
    Artificial Intelligence (AI) has revolutionized numerous industries and aspects of human life. 
    Machine learning, a subset of AI, enables computers to learn and improve from experience without 
    being explicitly programmed. Deep learning, which uses neural networks with multiple layers, 
    has achieved remarkable success in image recognition, natural language processing, and speech 
    recognition. Natural Language Processing (NLP) allows computers to understand, interpret, and 
    generate human language. This technology powers chatbots, language translation services, and 
    sentiment analysis tools. Computer vision enables machines to interpret and analyze visual 
    information from the world, leading to applications in autonomous vehicles, medical imaging, 
    and security systems. Reinforcement learning is another important branch of machine learning 
    where agents learn to make decisions through trial and error, receiving rewards or penalties 
    based on their actions. This approach has been successfully applied to game playing, robotics, 
    and optimization problems. The ethical implications of AI development include concerns about 
    job displacement, privacy, bias in algorithms, and the need for transparent and explainable 
    AI systems. As AI continues to advance, it's crucial to ensure that these technologies are 
    developed and deployed responsibly for the benefit of humanity.
    """
    
    print("=== Text Summarization ===")
    
    # Extractive summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(long_text, max_length=100, min_length=30, do_sample=False)
    print("Extractive Summary:")
    print(summary[0]['summary_text'])
    
    # Abstractive summarization
    abstractive_summarizer = pipeline("summarization", model="t5-base")
    
    # Prepare text for T5 (requires "summarize:" prefix)
    t5_input = "summarize: " + long_text
    abstract_summary = abstractive_summarizer(t5_input, max_length=100, min_length=30)
    print("\nAbstractive Summary:")
    print(abstract_summary[0]['summary_text'])

def question_answering_system():
    """Advanced question answering system"""
    
    # Context and questions
    context = """
    Machine Learning is a subset of artificial intelligence that enables computers to learn 
    and improve from experience without being explicitly programmed. There are three main 
    types of machine learning: supervised learning, unsupervised learning, and reinforcement 
    learning. Supervised learning uses labeled training data to learn a mapping from inputs 
    to outputs. Unsupervised learning finds hidden patterns in data without labeled examples. 
    Reinforcement learning learns through interaction with an environment, receiving rewards 
    or penalties for actions.
    """
    
    questions = [
        "What is machine learning?",
        "How many types of machine learning are there?",
        "What does supervised learning use?",
        "How does reinforcement learning work?"
    ]
    
    print("\n=== Question Answering System ===")
    
    # Load QA model
    qa_pipeline = pipeline("question-answering", 
                          model="distilbert-base-cased-distilled-squad")
    
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (Score: {result['score']:.3f})")
        print()

# Run advanced NLP tasks
advanced_text_summarization()
question_answering_system()
```

### Advanced Language Model Fine-tuning

```python
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TextDataset, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)

class CustomTextGenerator:
    """Custom text generator with fine-tuned language model"""
    
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt, max_length=100, num_return_sequences=1, 
                     temperature=0.7, do_sample=True):
        """Generate text based on prompt"""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def fine_tune_on_domain_data(self, texts, output_dir="./fine_tuned_model"):
        """Fine-tune model on domain-specific data"""
        
        # Prepare training data
        training_data = "\n\n".join(texts)
        
        # Create dataset
        def create_dataset(texts, tokenizer, block_size=128):
            """Create dataset for language modeling"""
            examples = []
            for text in texts:
                tokenized = tokenizer(text, truncation=True, padding=True, 
                                    max_length=block_size)
                examples.append(tokenized)
            return examples
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            warmup_steps=100,
            logging_steps=100
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2 is not a masked language model
        )
        
        print("Fine-tuning would be performed here...")
        print(f"Training on {len(texts)} texts")

# Example usage
print("\n=== Custom Text Generator ===")
generator = CustomTextGenerator()

# Generate text
prompts = [
    "The future of artificial intelligence",
    "Machine learning applications in",
    "Deep learning revolutionizes"
]

for prompt in prompts:
    generated = generator.generate_text(prompt, max_length=80, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated[0]}")
    print()

# Domain-specific fine-tuning example
domain_texts = [
    "Machine learning models require careful preprocessing of data.",
    "Neural networks learn complex patterns through backpropagation.",
    "Deep learning has transformed computer vision and NLP.",
    "Gradient descent optimizes model parameters iteratively.",
    "Overfitting occurs when models memorize training data."
]

generator.fine_tune_on_domain_data(domain_texts)
```

This completes the comprehensive NLP and specialized libraries documentation with state-of-the-art techniques including custom NER, transformer fine-tuning, advanced topic modeling, text summarization, question answering, and language model customization.
