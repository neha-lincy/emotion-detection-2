import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the processed training data
train_data = pd.read_csv("data/interim/train_bow.csv")

# Separate features (X) and target labels (y) from the training data
x_train = train_data.drop(columns=['sentiment']).values  # Drop the 'label' column to get feature values
y_train = train_data['sentiment'].values  # Extract the 'label' column as target values

# Initialize a Random Forest Classifier with 100 estimators and a fixed random state for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to a file using pickle for later use
pickle.dump(model, open("models/random_forest_model.pkl", "wb"))

with open ("models/random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)