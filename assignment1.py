from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

text = """Tokenization is a crucial step in natural language processing. Different tokenization methods include whitespace, punctuation-based, Treebank, Tweet, and MWE.
        NLTK library provides useful tools for text processing tasks. Stemming reduces words to it's base form, e.g., running becomes run. Porter stemmer and snowball stemmer are popular stemming algorithms.,
       Lemmatization, on the other hand, considers the context and meaning of words. #NLProc is an awesome #NLP conference!, Looking forward to #ACL2023 this summer!,
       @JohnDoe will present on # Transformers!"""

# Whitespace tokenization
print("\nWhitespace tokenizer:")
tokenizer = WhitespaceTokenizer()
print(tokenizer.tokenize(text))

# Punctuation tokenization
print("\nPunctuation tokenizer:")
tokenizer = WordPunctTokenizer()
print(tokenizer.tokenize(text))

# Treebank tokenization
print("\nTreebank tokenizer:")
tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(text))

# Tweet tokenization
print("\nTweet tokenizer:")
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(text))

# Multi-word expression tokenization
print("\nMWE tokenizer:")
tokenizer = MWETokenizer()
print(tokenizer.tokenize(text))

# Stemming
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
words = ["writing", "writes", "generously", "movies", "filmy", "lengthy", "honestly"]
for word in words:
    print(word + ":" + porter_stemmer.stem(word))
print("\nSnowball stemming:")
for word in words:
    print(word + ":" + snowball_stemmer.stem(word))

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("\nLemmatization:")
for word in words:
    print(word + ":" + lemmatizer.lemmatize(word))
