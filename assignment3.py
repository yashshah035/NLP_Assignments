import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

text = '''Tokenization is a crucial step #in natural language processing. Different )tokenization methods include whitespace, punctuation-based, Treebank, Tweet, and MWE.
        NLTK library provides useful tools for text processing tasks. Stemming reduces words to their base form, e.g., running becomes run. Porter stemmer and snowball stemmer are popular stemming algorithms.,
       Lemmatization, on the other hand, considers the context and meaning of words. #NLProc is an awesome #NLP conference!, Looking forward to #ACL2023 this summer!,
       @JohnDoe will present on # Transformers!'''

# Text cleaning
text = re.sub(r'[^a-zA-Z\s]', '', text) 
text = re.sub(r'\s+', ' ', text)
print("\nCleaned text:") 
print(text)

# Lemmatization  
lemmatizer = WordNetLemmatizer()
text = [lemmatizer.lemmatize(word) for word in text.split()]
text = " ".join(text)
print("\nLemmatized text:")
print(text)  

# Remove stop words
stop_words = set(stopwords.words('english'))  
text = [word for word in text.split() if word not in stop_words]
text = " ".join(text) 
print("\nText after removing stop words:")
print(text)

# Label encoding
labels = ['positive', 'negative']
encoder = LabelEncoder()
encoder.fit(labels) 
print("\nLabel mapping:")  
for cl in labels:
    print(cl, '->', encoder.transform([cl]))
    
# TF-IDF vectorization 
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform([text])
print("\nTF-IDF vector:")
print(vector)  

# Save outputs 
out = {'cleaned_text': text, 
       'lemma_text': lemmatizer.lemmatize(text),
       'stopword_free_text': text,
       'tf_idf_vector': vector.toarray()[0]}  
df = pd.DataFrame(out) 
print("\nDataframe:")
print(df)
df.to_csv('output.csv', index=False)

print("\nSaved outputs to output.csv")