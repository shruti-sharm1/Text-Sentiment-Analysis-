# Text-Sentiment-Analysis
~ Machine Learning Project

~ Segregating given dataset of tweets into positive, negative and neutral sentiment using sentiment analysis.

# Introduction

Given the availability of a large volume of online data, sentiment analysis becomes increasingly important. 

In this project, a sentiment classifier is built which evaluates the sentiment of a piece of text being either positive negative or neutral.

# Getting the Dataset
The "SentimentAnalysis.txt" shall be used for this project.The tweets are divided into the training, validation and test set.

# Data Preprocessing
The csv file has six columns,"target", “ids”, "date", “flag”, "user" and "text".

The column “text” contains tweets and the column “target” consists of sentiment labels, 4 for positive , 0 for negative . 

Cleaning the tweets involves:

    1. Remove the usernames
    2. Remove all the numbers
    3. Convert the tweet in lowercase
    4. Remove all the links
    5. Remove special characters
    6. Remove stopwords 
    7. Tokenizing the data
    8. Stemming the data

# Environment
Language : Python

# Libraries 
Scikit, Pandas, bs4, nltk, re, Numpy, Matplotlib, WordCloud, Spacy, SkLearn

# Result
Here, 1 is given for positive labels, -1 is for negative labels and 0 for neutral labels.
