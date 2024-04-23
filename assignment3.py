import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample text
input_text = '''Tokenization is a crucial step #in natural language processing. Different )tokenization methods include whitespace, punctuation-based, Treebank, Tweet, and MWE.
        NLTK library provides useful tools for text processing tasks. Stemming reduces words to their base form, e.g., running becomes run. Porter stemmer and snowball stemmer are popular stemming algorithms.,
       Lemmatization, on the other hand, considers the context and meaning of words. #NLProc is an awesome #NLP conference!, Looking forward to #ACL2023 this summer!,
       @JohnDoe will present on # Transformers!'''

# Cleaned text
cleaned_text = re.sub(r'[^a-zA-Z\s]', '', input_text) 
cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
print("\nCleaned text:") 
print(cleaned_text)

# Lemmatization  
lemmatizer = WordNetLemmatizer()
lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text.split()]
lemmatized_text = " ".join(lemmatized_text)
print("\nLemmatized text:")
print(lemmatized_text)  

# Remove stop words
stop_words = set(stopwords.words('english'))  
filtered_text = [word for word in lemmatized_text.split() if word not in stop_words]
filtered_text = " ".join(filtered_text) 
print("\nText after removing stop words:")
print(filtered_text)

# Label encoding
labels = word_tokenize(input_text)
encoder = LabelEncoder()
encoder.fit(labels) 
encoded_value = encoder.transform(labels)
print("\nLabel encoded value:")
print(encoded_value)


# TF-IDF vectorization 
vectorizer = TfidfVectorizer()
tfidf_vector = vectorizer.fit_transform([filtered_text])
print("\nTF-IDF vector:")
print(tfidf_vector)    

# Save outputs 
out = {'cleaned_text': cleaned_text, 
       'lemma_text': lemmatizer.lemmatize(lemmatized_text),
       'stopword_free_text': filtered_text,
       'tf_idf_vector': tfidf_vector.toarray()[0]}  
df = pd.DataFrame(out) 
print("\nDataframe:")
print(df)
df.to_csv('output.csv', index=False)

print("\nSaved outputs to output.csv")