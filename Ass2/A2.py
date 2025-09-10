# bagofword2.py

import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Sample paragraph
paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

# Sentence tokenization
sentences = nltk.sent_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()
corpus = []

# Text cleaning & lemmatization
for sentence in sentences:
    sent = re.sub('[^a-zA-Z]', ' ', sentence)  # Keep only letters
    sent = sent.lower()
    sent = sent.split()
    sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    corpus.append(sent)

print("Cleaned Corpus:")
print(corpus)

# Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
independentFeatures = cv.fit_transform(corpus).toarray()

df_bow = pd.DataFrame(independentFeatures, columns=cv.get_feature_names_out())
print("\nBag of Words Table:")
print(df_bow)

# TF-IDF Model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()

df_tfidf = pd.DataFrame(independentFeatures_tfIDF, columns=tfidf.get_feature_names_out())
print("\nTF-IDF Table:")
print(df_tfidf.round(3))  # Rounded for cleaner output
