import csv
from nltk.corpus import stopwords as stopwords

from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import sentiwordnet as swn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
from sklearn import svm

from sklearn.metrics import classification_report
import statistics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
train_x=[]#sentiment_text
train_y=[]#sentiment label
def loading_data_preProcessing(path):
    with open(path, encoding='latin-1') as dataSet:
        reader = csv.DictReader(dataSet, delimiter=',')
        i=0
        for row in reader:
            print(i)
            i=i+1
            # to_fix_row=row['SentimentText'].split('@')
            to_fix_row=row['SentimentText']
            if to_fix_row[1:]== '@':
                to_fix_row=to_fix_row[1:]
                to_fix_row=to_fix_row[0].split(' ')

            # to_fix_row=to_fix_row[1:]
            # to_fix_row=to_fix_row[0].split(' ')
            to_fix_row = to_fix_row.split(' ')
            new_row=""
            for word in to_fix_row:
                if (word not in stop_words and not word.isdigit() and not ''):
                    word2= word.lower()
                    word3=PorterStemmer().stem(word2)
                    new_row=new_row+" "+word3

            train_x.append(new_row)
            train_y.append(row['Sentiment'])

        return train_x, train_y
def loading_data_preProcessing_test(path):
    with open(path, encoding='latin-1') as dataSet:
        reader = csv.DictReader(dataSet, delimiter=',')
        i=0
        for row in reader:
            print(i)
            i=i+1
            # to_fix_row=row['SentimentText'].split('@')
            to_fix_row=row['SentimentText']
            if to_fix_row[1:]== '@':
                to_fix_row=to_fix_row[1:]
                to_fix_row=to_fix_row[0].split(' ')

            # to_fix_row=to_fix_row[1:]
            # to_fix_row=to_fix_row[0].split(' ')
            to_fix_row = to_fix_row.split(' ')
            new_row=""
            for word in to_fix_row:
                if (word not in stop_words and not word.isdigit() and not ''):
                    word2= word.lower()
                    word3=PorterStemmer().stem(word2)
                    new_row=new_row+" "+word3

            train_x.append(new_row)
            # train_y.append(row['Sentiment'])

        return train_x
# def split_train_test
#feature extraction using tfidf
def feature_extraction_train_test(train,test):
    vectorize=TfidfVectorizer()
    features_train=vectorize.fit_transform(train)
    features_test=vectorize.transform(test)
    return features_train,features_test


def classifierNaiveBayse(features_train,test,features_test,test_test):
    model=MultinomialNB(alpha=0.01)
    model.fit(features_train,test)
    pred = model.predict(features_test)
    # score = accuracy_score(test_test, pred)
    # print("accuracy:", score)

def classifierSVM(train,test):
    vectorize = TfidfVectorizer()
    features_train = vectorize.fit_transform(train)
    model=svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(model, features_train,test, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

def classifierLogisticREG(features_train,test):
    pass

print('='*20+"starting preprocessing"+'='*20)
train,test=loading_data_preProcessing("C:\\Users\\Chen\\Desktop\\python_course\\Train.csv")
train_test=loading_data_preProcessing_test("C:\\Users\\Chen\\Desktop\\python_course\\Test.csv")


print('='*20+"starting features extraction"+'='*20)
features_train,features_test=feature_extraction_train_test(train,train_test)


print('='*20+"starting training the model naive bayes"+'='*20)
classifierNaiveBayse(features_train,test,features_test)


print('='*20+"starting training the model svm"+'='*20)
classifierSVM(train,test)
