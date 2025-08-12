import numpy as np
import pandas as pd

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os

# Load the processed training and testing datasets, dropping rows with missing 'content'
train_data = pd.read_csv("data/processed/train.csv").dropna(subset=['content'])
test_data = pd.read_csv("data/processed/test.csv").dropna(subset=['content'])

# Extract the 'content' column as features (X) and 'label' column as targets (y) for training and testing
x_train = train_data['content'].values
y_train = train_data['sentiment'].values

x_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Initialize the CountVectorizer for Bag of Words feature extraction
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it into a sparse matrix
X_train_bow = vectorizer.fit_transform(x_train)

# Transform the test data using the same vectorizer (to ensure consistency)
X_test_bow = vectorizer.transform(x_test)

# Convert the sparse matrix of training data into a DataFrame
train_df = pd.DataFrame(X_train_bow.toarray())

# Add the target labels (y_train) as a new column in the training DataFrame
train_df['sentiment'] = y_train

# Convert the sparse matrix of test data into a DataFrame
test_df = pd.DataFrame(X_test_bow.toarray())

# Add the target labels (y_test) as a new column in the test DataFrame
test_df['sentiment'] = y_test

# Save the processed training data with Bag of Words features to a CSV file
train_df.to_csv("data/interim/train_bow.csv", index=False)

# Save the processed test data with Bag of Words features to a CSV file
test_df.to_csv("data/interim/test_bow.csv", index=False)