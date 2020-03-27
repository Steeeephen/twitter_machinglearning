import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
import re
import math
import string
import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
from keras import layers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from spellchecker import SpellChecker
from sklearn.model_selection import RandomizedSearchCV

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print("Train, test sets imported")

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    url = re.compile(r'<.*?>')
    return url.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))        

def tokenization(text):
    text = re.split('\W+', text)
    return text

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

spell = SpellChecker()
def spelling(text):
    text = [spell.correction(x) for x in text]
    return(text)
print("Processing data")
k = 60
for datas in [train_df,test_df]:
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : remove_html(x))
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : remove_url(x))
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : remove_emoji(x))
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : remove_punc(x))
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : tokenization(x.lower()))
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : remove_stopwords(x))
    print(train_df['text'][k])
    datas['text'] = datas['text'].apply(lambda x : ' '.join(x))
    print(train_df['text'][k])


vocab_size = 10000
print("Encoding words")
encoded_docs = [one_hot(d, vocab_size) for d in train_df['text']]
max_length = 25#
print("Padding vectors")
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

count_vec = feature_extraction.text.CountVectorizer()
train_vec = count_vec.fit_transform(train_df['text'])
test_vec = count_vec.transform(test_df['text'])

test_vec = train_vec[6000:]
test_label = train_df['target'][6000:]
train_vec = train_vec[:6000]
train_label = train_df['target'][:6000]


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model = RandomForestClassifier()
print("Beginning tuning")
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
model_random.fit(train_vec, train_label)
#model.fit(train_vec,train_label)
#preds = pd.DataFrame(model.predict(test_vec)).round()

#accuracy_score(model.predict(test_vec),test_label)
print(model_random.best_params_)