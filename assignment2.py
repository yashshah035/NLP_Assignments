from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample data
corpus = ["The cat is jumping over the fence.",
          "She loves eating apples in the evening.",
          "I can't believe it's already Friday!"]

# Bag-of-Words (Count Occurrence)
vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(corpus)
count_matrix = X_count.toarray()

# Bag-of-Words (Normalized Count Occurrence)
normalized_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)

# TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(corpus)
tfidf_matrix = X_tfidf.toarray()

# Word2Vec
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
word_embeddings = {word: word2vec_model.wv[word] for word in word2vec_model.wv.index_to_key}

# Print results
print("Count Occurrence Matrix:")
print(count_matrix)

print("\nNormalized Count Occurrence Matrix:")
print(normalized_matrix)

print("\nTF-IDF Matrix:")
print(tfidf_matrix)

print("\nWord Embeddings:")
print(word_embeddings)
