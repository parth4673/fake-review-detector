import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tkinter import * 
import pickle
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


df = pd.read_csv('dataSet/Preprocessed Fake Reviews Detection Dataset.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)

df.dropna(inplace=True)
df['length'] = df['text_'].apply(len)



stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
     return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])


def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


# plt.hist(df['length'],bins=50)
# plt.show()

# df.hist(column='length',by='label',bins=50,color='blue',figsize=(12,5))
# plt.show()

df[df['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer=text_process)

# bow_transformer.fit(df['text_'])

# print("Total Vocabulary:",len(bow_transformer.vocabulary_))

review4 = df['text_'][3]

# bow_msg4 = bow_transformer.transform([review4])
# print(bow_msg4)
# print(bow_msg4.shape)

# print(bow_transformer.get_feature_names()[15841])
# print(bow_transformer.get_feature_names()[23848])

# bow_reviews = bow_transformer.transform(df['text_'])
# print("Shape of Bag of Words Transformer for the entire reviews corpus:",bow_reviews.shape)
# print("Amount of non zero values in the bag of words model:",bow_reviews.nnz)

# tfidf_transformer = TfidfTransformer().fit(bow_reviews)
# tfidf_rev4 = tfidf_transformer.transform(bow_msg4)
# print(tfidf_rev4)

# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['mango']])
# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['book']])

# tfidf_reviews = tfidf_transformer.transform(bow_reviews)
# print(tfidf_reviews)
# print("Shape:",tfidf_reviews.shape)
# print("No. of Dimensions:",tfidf_reviews.ndim)

review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)

# pipeline = Pipeline([
#     ('bow',CountVectorizer(analyzer=text_process)),
#     ('tfidf',TfidfTransformer()),
#     ('classifier',SVC())
# ])

# pipeline.fit(review_train,label_train)

# svc_pred = pipeline.predict(review_test)
# #print(svc_pred)

# print('Classification Report:',classification_report(label_test,svc_pred))
# print('Confusion Matrix:',confusion_matrix(label_test,svc_pred))
# print('Accuracy Score:',accuracy_score(label_test,svc_pred))
# print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')

# pipeline = Pipeline([
#     ('bow',CountVectorizer(analyzer=text_process)),
#     ('tfidf',TfidfTransformer()),
#     ('classifier',LogisticRegression())
# ])
# pipeline.fit(review_train,label_train)

# lr_pred = pipeline.predict(review_test)
# #print(lr_pred)

# # print('Classification Report:',classification_report(label_test,lr_pred))
# # print('Confusion Matrix:',confusion_matrix(label_test,lr_pred))
# # print('Accuracy Score:',accuracy_score(label_test,lr_pred))
# # print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')

# #test = pipeline.predict("Nice taste")


# print(pipeline.predict(df12))

# joblib.dump(pipeline,'Predictor.joblib')

# with open('model.pkl','wb') as file:
#     pickle.dump(pipeline,file)

with open("model.pkl","rb") as file:
    newModel = pickle.load(file)



###### Tkinter GUI #########

root = Tk()
root.state('zoomed')
root.configure(background="#E8EFCF")

root.title("Review Predictor")


heading = Label(root,text="Enter Review", font=('Arial',42), bg="#E8EFCF")
heading.place(relx = 0.52,rely=0.3, anchor='center')


e = Entry(font=('Arial',18))
e.place(relx = 0.5, rely = 0.6, anchor = 'center')

global myLabel

myLabel = None

def clearPrev():
    global myLabel
    if myLabel:
        myLabel.pack_forget()
        myLabel = None

def predictReview():
    clearPrev()
    reviews = []
    review = e.get()
    review = preprocess(review)
    review = stem_words(review)
    review = lemmatize_words(review)
    reviews.append({"text_":review})
    df12 = pd.DataFrame(reviews)
    global myLabel

    if newModel.predict(df12["text_"])[0] == 'OR':
        myLabel = Label(root, text="The review is original", font=('Arial', 20), bg="#E8EFCF")
    else:
        myLabel = Label(root, text="The review is computer generated", font=('Arial', 20), bg="#E8EFCF")
    myLabel.place(relx=0.4, rely = 0.7)


myButton = Button(root, text="Submit", command=predictReview)
myButton.place(relx = 0.65, rely = 0.6, anchor='center')

root.mainloop()




