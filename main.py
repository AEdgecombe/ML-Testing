import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re

data = pd.read_csv("big_sentiment_data.csv")

data['text'] = data['text'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))  # Remove non-ASCII characters
data['text'] = data['text'].apply(lambda x: x.lower())  # Convert text to lowercase

X = data["text"]
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def custom_tokenizer(text):
    return text.split() 

vectorizer = CountVectorizer(min_df=2, tokenizer=custom_tokenizer)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

y_pred = classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))
