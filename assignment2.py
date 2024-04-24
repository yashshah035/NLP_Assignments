from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample data
corpus = ["The cat is jumping over the fence",
          "She loves eating apples in the evening",
          "I can not believe its already Friday!"]

# Bag-of-Words (Count Occurrence)
vectorizer_count = CountVectorizer()
count = vectorizer_count.fit_transform(corpus)
count_matrix = count.toarray()

# Bag-of-Words (Normalized Count Occurrence)
normalized_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)

# TF-IDF
vectorizer_tfidf = TfidfVectorizer() 
X_tfidf = vectorizer_tfidf.fit_transform(corpus)
tfidf_matrix = X_tfidf.toarray()

# Word2Vec
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_sizeign=5, window=5, min_count=1)
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


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extract word vectors and their corresponding words
words = list(word_embeddings.keys())
vectors = [word_embeddings[word] for word in words]

# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(vectors)

# Plot the word embeddings
plt.figure(figsize=(10, 8))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], marker='o', s=30)

# Annotate each point with its corresponding word
for i, word in enumerate(words):
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Word Embeddings Visualization (PCA)')
plt.grid(True)
plt.show()

