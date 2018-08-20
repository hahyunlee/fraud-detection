import pandas as pd
import numpy as np
from main import pipeline
from sklearn.model_selection import train_test_split
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import os
import random
from wordcloud import WordCloud, STOPWORDS
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_wordcloud(df, col_target, col_description,target=1):
    words = ' '.join((list(df[df[col_target]==target][col_description])))
    wc = WordCloud(width = 512, height = 512, stopwords=STOPWORDS).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'k')
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def text_to_features(df,col_desc):
    # Uses PorterStemmer and TF-IDF
    descriptions = df[col_desc].tolist()
    lowered = [content.lower() for content in descriptions]
    sw = set(stopwords.words('english'))
    words = [word for word in lowered if word not in sw]
    tokens = [word_tokenize(content) for content in words]
    porter = PorterStemmer()
    docs_porter = [[porter.stem(w) for w in words] for words in tokens]
    joined_docs_porter = [' '.join(x) for x in docs_porter]
    # Returns sparse matrix of featurized text
    vc = TfidfVectorizer(analyzer='word', stop_words='english')
    feature = vc.fit_transform(joined_docs_porter)
    return feature

if __name__ == '__main__':
    # Limiting model to just description
    df0 = pd.read_json('data/data.json')
    df = df0.copy()
    df = pipeline(df)
    df = df.filter(items=['fraud','description'])
    create_wordcloud(df,'fraud','description',1)
    create_wordcloud(df,'fraud','description',0)

    X = df.description
    y = df.fraud






