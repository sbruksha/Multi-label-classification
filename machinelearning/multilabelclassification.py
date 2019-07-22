# Multi-label text classification
import time
import pandas as pd
import numpy as np
import string
import pickle
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score

start_time = time.time()
print("Start--- %s seconds ---" % (time.time() - start_time))

# Read the csv file into dataframe using pandas
df = pd.read_csv("multilabeldata.csv")
print("Initial shape:", df.shape)

# Sample data, without that it will run for hours
df = df.sample(frac = 0.1, replace = False, random_state=42)
print("Sample shape:", df.shape)

# Shuffling of indices
df = df.reindex(np.random.permutation(df.index))

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Cleaning data:
# - Lower text
# - Tokenize text and remove punctuations
# - Remove words that contain numbers
# - Remove Stop Words
# - Remove empty tokens
# - Lemmatise text
# - Remove words with only one letter
def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove punctuations
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text: rooms - > room, went - > go
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return (text)


# clean text data
df["comment_clean"] = df.apply(lambda row: clean_text(row["comment_text"]), axis=1)
print("After clean shape:",df.shape,"--- %s seconds ---" % (time.time() - start_time))

# Separate the comment field data and outcome labels
comment = df['comment_clean']
comment = comment.as_matrix()

label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
label = label.as_matrix()

# Remove excessive length comments
comments = []
labels = []

# Removing comments longer than 500
for ix in range(comment.shape[0]):
    if len(comment[ix])<=500:
        comments.append(comment[ix])
        labels.append(label[ix])

labels = np.asarray(labels)

# Updating the list of stop words
from stop_words import get_stop_words
stop_words = get_stop_words('english')
stop_words.append('')
for x in range(ord('b'), ord('z')+1):
    stop_words.append(chr(x))

# Applying Count Vectorizer
# Here we can finally convert our comments into a matrix of token counts, which signifies the number of times it occurs.
from sklearn.feature_extraction.text import CountVectorizer

# Create object supplying our custom stop words
count_vector = CountVectorizer(stop_words=stop_words)
# Create and save with pickle
save_mydocuments = open("pickled_algos/CountVectorizer.pickle","wb")
pickle.dump(count_vector, save_mydocuments)
save_mydocuments.close()
# Fitting it to converts comments into bag of words format
tf = count_vector.fit_transform(comments).toarray()
# Create and save CountVectorizerComments with pickle
save_mydocuments = open("pickled_algos/CountVectorizerComments.pickle","wb")
pickle.dump(comments, save_mydocuments)
save_mydocuments.close()

# Splitting dataset into training and testing
def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = shuffle(tf, labels,3)

print("X_test shape:",X_test.shape,"--- %s seconds ---" % (time.time() - start_time))

print("X_train shape:",X_train.shape)

# Starting with the First Model
# 1. Binary Relevance (BR) Method with MultinomialNB classifiers (from scratch)
from sklearn.naive_bayes import MultinomialNB
# clf will be the list of the classifiers for all the 6 labels
# each classifier is fit with the training data and corresponding classifier
clf = []
for ix in range(6):
    clf.append(MultinomialNB())
    clf[ix].fit(X_train,Y_train[:,ix])


# predict list contains the predictions, it is transposed later to get the proper shape
predict = []
for ix in range(6):
    predict.append(clf[ix].predict(X_test))

predict = np.asarray(np.transpose(predict))
print("Predict.shape:",predict.shape)

# Accuracy
print("Accuracy : {}".format(accuracy_score(Y_test,predict)*100))

# Create and save with pickle
save_mydocuments = open("pickled_algos/MultilabelBinaryRelevanceWithMultinomialNB.pickle","wb")
pickle.dump(clf, save_mydocuments)
save_mydocuments.close()
print("Binary Relevance (BR) Method with MultinomialNB is done, time--- %s seconds ---" % (time.time() - start_time))

# 2. BR Method with SVM classifier (from scikit-multilearn)
# create and fit classifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
BinaryRelevance_classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
BinaryRelevance_classifier.fit(X_train, Y_train)

#predictions
predictions = BinaryRelevance_classifier.predict(X_test)

# Accuracy
print("Accuracy : {}".format(accuracy_score(Y_test,predict)*100))
# Create and save with pickle
save_mydocuments = open("pickled_algos/MultilabelBinaryRelevanceWithSVM.pickle","wb")
pickle.dump(BinaryRelevance_classifier, save_mydocuments)
save_mydocuments.close()
print("BinaryRelevance_classifier is done, time--- %s seconds ---" % (time.time() - start_time))

# 3. BR Method with Multinomial classifier (from scikit-multilearn)
# create and fit classifier
classifier = BinaryRelevance(classifier = MultinomialNB(), require_dense = [False, True])
classifier.fit(X_train, Y_train)

# Predictions
predictions = classifier.predict(X_test)

# Accuracy
print("Accuracy : {}".format(accuracy_score(Y_test,predict)*100))
# Create and save with pickle
save_mydocuments = open("pickled_algos/MultilabelBinaryRelevanceWithMultinomial.pickle","wb")
pickle.dump(classifier, save_mydocuments)
save_mydocuments.close()
print("BR Method with Multinomial classifier is done, time--- %s seconds ---" % (time.time() - start_time))

# Very slow for full set of data
# 4. BR Method with GausseanNB classifier (from scratch)
# from sklearn.naive_bayes import GaussianNB
# #create and fit classifiers
# clf = []
# for ix in range(6):
#     clf.append(GaussianNB())
#     clf[ix].fit(X_train,Y_train[:,ix])
#
#
# #predictions
# predict = []
# for ix in range(6):
#     predict.append(clf[ix].predict(X_test))
#
#
# #calculate scores
# predict = np.asarray(np.transpose(predict))
# # Accuracy
# print("Accuracy : {}".format(accuracy_score(Y_test,predict)*100))
# # Create and save with pickle
# save_mydocuments = open("pickled_algos/MultilabelBinaryRelevanceWithGausseanNB.pickle","wb")
# pickle.dump(clf, save_mydocuments)
# save_mydocuments.close()
# print("BR Method with GausseanNB classifier is done, time--- %s seconds ---" % (time.time() - start_time))

# 5. Classifier chain with MultinomialNB classifier (from scikit-multilearn)
# create and fit classifier
from skmultilearn.problem_transform import ClassifierChain
ClassifierChainMultinomialNB_classifier = ClassifierChain(MultinomialNB())
ClassifierChainMultinomialNB_classifier.fit(X_train, Y_train)

# Predictions
predictions = ClassifierChainMultinomialNB_classifier.predict(X_test)

# Accuracy
print("Accuracy : {}".format(accuracy_score(Y_test,predict)*100))
# Create and save with pickle
save_mydocuments = open("pickled_algos/MultilabelClassifierchainWithMultinomialNB.pickle","wb")
pickle.dump(ClassifierChainMultinomialNB_classifier, save_mydocuments)
save_mydocuments.close()
print("Classifier chain with MultinomialNB classifier is done, time--- %s seconds ---" % (time.time() - start_time))

# 6. Label Powerset with MultinomialNB classifier (from scikit-multilearn)
# create and fit classifier
from skmultilearn.problem_transform import LabelPowerset
LabelPowersetMultinomialNB_classifier = LabelPowerset(MultinomialNB())
LabelPowersetMultinomialNB_classifier.fit(X_train, Y_train)

# Predictions
predictions = LabelPowersetMultinomialNB_classifier.predict(X_test)

# Accuracy
print("Accuracy : {}".format(accuracy_score(Y_test,predict)*100))
# Create and save with pickle
save_mydocuments = open("pickled_algos/MultilabelPowersetWithMultinomialNB.pickle","wb")
pickle.dump(LabelPowersetMultinomialNB_classifier, save_mydocuments)
save_mydocuments.close()
print("LabelPowersetMultinomialNB_classifier is done, time--- %s seconds ---" % (time.time() - start_time))

print("Done")