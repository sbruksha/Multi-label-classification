import time
import pandas as pd
import numpy as np
import pickle
import itertools

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

# Read the csv file into dataframe df - with 100% accuracy
df = pd.read_csv("multilabeltest100.csv")

# Separate the comment field data and outcome labels
comment = df['comment_text']
print(comment.head())
comment = comment.as_matrix()

label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
label = label.as_matrix()

comments = []
labels = []

for ix in range(comment.shape[0]):
    if len(comment[ix])<=400:
        comments.append(comment[ix])
        labels.append(label[ix])


labels = np.asarray(labels)
print(labels)

# Cleaning data
# - Removing Punctuations and other special characters
# - Splitting the comments into individual words
# - Removing Stop Words
# - Stemming and Lemmatising
# - Applying Count Vectoriser
# - Splitting dataset into Training and Testing

# ## Preparing a string containing all punctuations to be removed
import string
print(string.punctuation)
punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
print (punctuation_edit)
outtab = "                                         "
trantab = str.maketrans(punctuation_edit, outtab)


# Updating the list of stop words
from stop_words import get_stop_words
stop_words = get_stop_words('english')
stop_words.append('')

for x in range(ord('b'), ord('z')+1):
    stop_words.append(chr(x))

# Stemming and Lemmatizing
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer


# create objects for stemmer and lemmatizer
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()
# download words from wordnet library
nltk.download('wordnet')


# We can now, loop once through all the comments applying :
# - punctuation removal
# - splitting the words by space
# - applying stemmer and lemmatizer
# - recombining the words again for further processing

for i in range(len(comments)):
    comments[i] = comments[i].lower().translate(trantab)
    l = []
    for word in comments[i].split():
        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
    comments[i] = " ".join(l)


# Applying Count Vectorizer
open_file = open("pickled_algos/CountVectorizer.pickle", "rb")
count_vector = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/CountVectorizerComments.pickle","rb")
cmnts = pickle.load(open_file)
open_file.close()

tf1 = count_vector.fit_transform(cmnts).toarray()
# call transform() for that CountVectorizer object, and it will give you the counts of the tokens found in the training data that are also found in the new (test) data. This gives you data of the same shape to pass on to your classifier
tf = count_vector.transform(comments).toarray()
print(tf.shape)

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def evaluate_score(Y_test,predict):
    loss = hamming_loss(Y_test,predict)
    print("Hamming_loss : {}".format(loss*100))
    accuracy = accuracy_score(Y_test,predict)
    print("Accuracy : {}".format(accuracy*100))
    try :
        loss = log_loss(Y_test,predict)
    except :
        loss = log_loss(Y_test,predict.toarray())
    print("Log_loss : {}".format(loss))


# 1
open_file = open("pickled_algos/MultilabelBinaryRelevanceWithMultinomialNB.pickle", "rb")
MultilabelBinaryRelevanceWithMultinomialNB = pickle.load(open_file)
open_file.close()

# 2
open_file = open("pickled_algos/MultilabelBinaryRelevanceWithSVM.pickle", "rb")
MultilabelBinaryRelevanceWithSVM = pickle.load(open_file)
open_file.close()

# 3
open_file = open("pickled_algos/MultilabelBinaryRelevanceWithMultinomial.pickle", "rb")
MultilabelBinaryRelevanceWithMultinomial = pickle.load(open_file)
open_file.close()

# 4
open_file = open("pickled_algos/MultilabelClassifierchainWithMultinomialNB.pickle", "rb")
MultilabelClassifierchainWithMultinomialNB = pickle.load(open_file)
open_file.close()

# 5
open_file = open("pickled_algos/MultilabelPowersetWithMultinomialNB.pickle", "rb")
LabelPowersetMultinomialNB_classifier = pickle.load(open_file)
open_file.close()


def using_tocoo_izip(x):
    cx = x.tocoo()
    for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
        print("i",i,"j",j,"v",v)

def using_tocoo(x):
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        print("ReviewIndex", i, "LabelIndex", j, "v", v)


def classify(X_test):
    # 1 predict list contains the predictions, it is transposed later to get the proper shape
    predict = []
    for ix in range(6):
        predict.append(MultilabelBinaryRelevanceWithMultinomialNB[ix].predict(X_test))

    predict = np.asarray(np.transpose(predict))
    print(predict)
    # calculate results
    #evaluate_score(labels, predict)

    # 2
    predictions = MultilabelBinaryRelevanceWithSVM.predict(X_test)
    using_tocoo(predictions)
    print("2", predictions)
    #evaluate_score(labels, predictions)

    # 3 MultilabelBinaryRelevanceWithMultinomial
    predictions = MultilabelBinaryRelevanceWithMultinomial.predict(X_test)
    using_tocoo(predictions)
    print("3", predictions)

    # 5
    predictions = MultilabelClassifierchainWithMultinomialNB.predict(X_test)
    using_tocoo(predictions)
    print("5", predictions)

    # 6
    predictions = LabelPowersetMultinomialNB_classifier.predict(X_test)
    using_tocoo(predictions)
    print("6", predictions)


classify(tf)