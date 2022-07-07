import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# # # # GETTING THE DATA # # # #
DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('Project_Data.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
print(df.sample(5))

data = df[['text', 'target']]
data['target'] = data['target'].replace(4, 1)
print(data['target'].unique())

# Separating positive and negative tweets
data_pos_df = data[data['target'] == 1]
data_neg_df = data[data['target'] == 0]

data_pos = data_pos_df.iloc[:int(20000)]
data_neg = data_neg_df.iloc[:int(20000)]

# Combining positive and negative tweets
dataset = pd.concat([data_pos, data_neg])

# Making statement text in lower case
print("Converting to Lower case")
dataset['text'] = dataset['text'].str.lower()


# Cleaning and removing the above stop words list from the tweet text
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords.words()])


print("Removing stopwords")
dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))

# Removing special characters
print("Removing special characters")
english_punctuations = string.punctuation
punctuations_list = english_punctuations


def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_punctuations(x))


# Cleaning and removing repeating characters
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)


print("Removing repeating characters")
dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))


# Cleaning and removing URL’s
def cleaning_URLs(data1):
    return re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', data1)


print("Removing URL'S")
dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))


# Cleaning and removing Numeric numbers
def cleaning_numbers(data1):
    return re.sub('[0-9]+', '', data1)


print("Removing numbers")
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))

# Getting tokenization of tweet text
print("Tokenization")
dataset['text'] = dataset['text'].apply(lambda x: word_tokenize(x))

# Applying Stemming
print("Stemming")
st = nltk.PorterStemmer()


def stemming_on_text(data1):
    text = [st.stem(word) for word in data1]
    return data1


dataset['text'] = dataset['text'].apply(lambda x: stemming_on_text(x))

# Applying Lemmatization
print("Lemmatization")
lm = nltk.WordNetLemmatizer()


def lemmatizer_on_text(data1):
    text = [lm.lemmatize(word) for word in data1]
    return text


dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))

# Separating input feature and label
X = data.text
y = data.target

# Splitting our data into Train and Test Subset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26105)

# Transforming Dataset using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


#  Model Evaluation
def model_Evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()


# Model Building

LRmodel = LogisticRegression()
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)
print("Accuracy Score = ", accuracy_score(y_test, y_pred3))
