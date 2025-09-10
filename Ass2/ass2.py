# bagofword2_spacy.py

import re
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load English model
nlp = spacy.load("en_core_web_sm")

# Sample paragraph
paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

# Process paragraph with spaCy
doc = nlp(paragraph)

corpus = []

# Tokenization, stopword removal, lemmatization
for sent in doc.sents:
    tokens = [
        token.lemma_.lower()
        for token in sent
        if token.is_alpha and not token.is_stop
    ]
    corpus.append(" ".join(tokens))

print("Cleaned Corpus:")
print(corpus)

# Bag of Words
cv = CountVectorizer()
independentFeatures = cv.fit_transform(corpus).toarray()
df_bow = pd.DataFrame(independentFeatures, columns=cv.get_feature_names_out())
print("\nBag of Words Table:")
print(df_bow)

# TF-IDF
tfidf = TfidfVectorizer()
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()
df_tfidf = pd.DataFrame(independentFeatures_tfIDF, columns=tfidf.get_feature_names_out())
print("\nTF-IDF Table:")
print(df_tfidf.round(3))
