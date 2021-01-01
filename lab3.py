import csv
from nltk.corpus import stopwords as stopwords
from nltk.tokenize import TweetTokenizer

from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import sentiwordnet as swn
import nltk
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
nltk.download('sentiwordnet')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import classification_report
import statistics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
stop_words.update(["Ð","¨","°","Ñ","Ð","Ø","¹","Ù","","²","§",":",".","!","...",",",";","?","..","~","@","'","^","#","=]","/","¼","Ã","-","¿","©",")","(","\"","*","$","::","xx","&","¶",":|","&","_","","]","["],">","<","´",'"\"',"___i",". . .","|")
train_x=[]#sentiment_text
train_y=[]#sentiment label
def loading_data_preProcessing(path, is_test):
    with open(path, encoding='latin-1') as dataSet:
        reader = csv.DictReader(dataSet, delimiter=',')
        i=0
        for row in reader:

            # to_fix_row=row['SentimentText'].split('@')

            to_fix_row=row['SentimentText']
            if(i==80000):
                print(to_fix_row)
            # if to_fix_row[1:]== '@':
            #     to_fix_row=to_fix_row[1:]
            #     to_fix_row=to_fix_row[0].split(' ')
            tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
            to_fix_row=tknzr.tokenize(to_fix_row)
            # to_fix_row=to_fix_row[1:]
            # to_fix_row=to_fix_row[0].split(' ')
            # to_fix_row = to_fix_row.split(' ')
            new_row=""
            for word in to_fix_row:
                #add . , : !
                if (word not in stop_words and not word.isdigit() and not len(word)==1):
                    word2= word.lower()
                    # word3=PorterStemmer().stem(word2)
                    new_row=new_row+" "+word2
            print(i)
            i=i+1

            train_x.append(new_row)
            if not is_test:
                train_y.append(row['Sentiment'])
        if is_test:
            return train_x

        return train_x, train_y

#  #feature extraction using tfidf
def feature_extraction_train_test(train,test):
    vectorize=TfidfVectorizer(ngram_range=(1,2))
    vectorize = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True,ngram_range=(1,2))
    # vectorize = TfidfVectorizer()
    features_train=vectorize.fit_transform(train)
    features_test=vectorize.transform(test)
    return features_train,features_test


def classifierNaiveBayse(features_train,test,features_test):
    model=MultinomialNB(alpha=0.01)
    params={}
    # skf = StratifiedKFold(n_splits=10)
    # model = GridSearchCV(MultinomialNB(),cv=skf,params=params, n_jobs=1)

    model.fit(features_train,test)
    pred = model.predict(features_test)
    return pred



def write_csv(pred,model_name):
    if(model_name=='Naive Bayes'):
        with open('sample_'+model_name+'.csv', 'w', newline='') as csvfile:
            fieldnames = ['ID', 'Sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            i=0
            for label in pred:
                writer.writerow({'ID': i, 'Sentiment': label})
                i=i+1
    if(model_name=='Logistic Regression'):
        with open('sample_'+model_name+'.csv', 'w', newline='') as csvfile:
            fieldnames = ['ID', 'Sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            i=0
            for label in pred:
                writer.writerow({'ID': i, 'Sentiment': label})
                i=i+1
    if(model_name=='SVM'):
        with open('sample_'+model_name+'.csv', 'w', newline='') as csvfile:
            fieldnames = ['ID', 'Sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            i=0
            for label in pred:
                writer.writerow({'ID': i, 'Sentiment': label})
                i=i+1



def classifierSVM(features_train,test,features_test):
    # vectorize = TfidfVectorizer()
    # features_train = vectorize.fit_transform(train)
    model=svm.SVC(kernel='linear', C=2,random_state=42)
    model.fit(features_train,test)
    pred=model.predict(features_test)
    # scores = cross_val_score(model, features_train,test, cv=10)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    return pred

def classifierLogisticREG(features_train,test):
    model=LogisticRegression(random_state=42,C=2)
    model.fit(features_train,test)
    pred=model.predict(features_test)
    return pred


print('='*20+"starting preprocessing"+'='*20)
train,test=loading_data_preProcessing("Train.csv",False)
train_x=[]
train_y=[]
train_test=loading_data_preProcessing("Test.csv",True)


print('='*20+"starting features extraction"+'='*20)
features_train,features_test=feature_extraction_train_test(train,train_test)


print('='*20+"starting training the model naive bayes"+'='*20)
pred=classifierNaiveBayse(features_train,test,features_test)
write_csv(pred,'Naive Bayes')

print('='*20+"starting training the model svm"+'='*20)
pred=classifierSVM(features_train,test,features_test)
write_csv(pred,'SVM')

print('='*20+"starting training the model logistic regression"+'='*20)
pred=classifierLogisticREG(features_train,test)
write_csv(pred,'Logistic Regression')

