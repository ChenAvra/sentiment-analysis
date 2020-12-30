import csv
from nltk.corpus import stopwords as stopwords
from nltk.tokenize import TweetTokenizer

from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import sentiwordnet as swn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import classification_report
import statistics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
stop_words.update(["111111111111","222222222"])
# stop_words.add()
print(stop_words)