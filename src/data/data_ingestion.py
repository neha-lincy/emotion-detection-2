import numpy as np
import pandas as pd
import os 

from sklearn.model_selection import train_test_split

# Load the dataset from a remote URL
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Drop the 'tweet_id' column as it is not needed for analysis
df.drop(columns=['tweet_id'], inplace=True)

# Filter the dataset to include only rows where sentiment is 'happiness' or 'sadness'
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Replace sentiment labels with numerical values: 'happiness' -> 1, 'sadness' -> 0
final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# Create the directory structure to store raw data if it doesn't already exist
os.makedirs("data/raw", exist_ok=True)

# Save the training data to a CSV file
train_data.to_csv("data/raw/train.csv", index=False)

# Save the testing data to a CSV file
test_data.to_csv("data/raw/test.csv", index=False)