# Text-Sentiment-Analysis
~ Machine Learning Project

~ Segregating given dataset of tweets into positive, negative and neutral sentiment using sentiment analysis.

# Introduction

Given the availability of a large volume of online data, sentiment analysis becomes increasingly important. 

In this project, a sentiment classifier is built which evaluates the sentiment of a piece of text being either positive negative or neutral.

# Getting the Dataset
The "Text Emotion" shall be used for this project.The 40,000 tweets are divided evenly into the training, validation and test set.

# Data Preprocessing
The csv file has three columns,"tweet_id", “content”, "author" and “sentiment”.

The column “content” contains tweets and the column “sentiment” consists of sentiment labels, 1 for positive , -1 for negative and 0 for neutral. 

Cleaning the tweets involves:

    1. Remove the usernames
    2. Remove all the numbers
    3. Convert the tweet in lowercase
    4. Remove all the links
    5. Remove special characters
    6. Remove stopwords 
    7. Tokenizing the data
    8. Stemming the data

# Algorithmic Overview
Functions used in the preprocessing_data.ipynb file:

    1. get_subject_phrase : Function to find the subject of all the tweets in the dataset 
    2. get_object_phrase  : Function to find the object of all the tweets in the dataset 
    3. pre_processing     : Function to pre-process each tweet
# Environment
Language : Python

# Libraries 
Scikit, Pandas, bs4, nltk, re, Numpy, Matplotlib, WordCloud, Spacy, SkLearn

# Result
Here, 1 is given for positive labels, -1 is for negative labels and 0 for neutral labels.
