import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

df = pd.read_csv('dataSet/fake reviews dataset.csv')

plt.figure(figsize=(15,8))
labels = df['rating'].value_counts().keys()
values = df['rating'].value_counts().values
plt.pie(values, labels = labels, shadow=False, autopct='%1.1f%%')
plt.title('Proportion of each rating', fontweight='bold', fontsize=25, pad=20, color = 'crimson')
plt.show()

def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


#print(df['text_'].head().apply(clean_text))

df['text_'] = df['text_'].astype(str)

def preprocess(text):
     return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

preprocess(df['text_'][4])

df['text_'][:10000] = df['text_'][:10000].apply(preprocess)
df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)
df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)
df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)
df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)
df['text_'] = df['text_'].str.lower()

stemmer = PorterStemmer()

def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

df['text_'] = df['text_'].apply(lambda x: stem_words(x))

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

print(stem_words(lemmatize_words(clean_text(("It was gradually decreasing.")))))

df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))
df['text_'].head()


df.to_csv('dataSet/Preprocessed Fake Reviews Detection Dataset.csv')
