from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pickle
import pandas as pd
import json

# Load the trained Random Forest model from the pickle file
with open("models/random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the processed test data
test_data = pd.read_csv("data/interim/test_bow.csv")

# Separate features (X) and target labels (y) from the test data
x_test = test_data.drop(columns=['sentiment']).values  # Drop the 'label' column to get feature values
y_test = test_data['sentiment'].values  # Extract the 'label' column as target values

# Use the trained model to make predictions on the test data
y_pred = model.predict(x_test)

# Calculate evaluation metrics for the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model
precision = precision_score(y_test, y_pred, average='weighted')  # Calculate precision (weighted average)
recall = recall_score(y_test, y_pred, average='weighted')  # Calculate recall (weighted average)

# Store the calculated metrics in a dictionary
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall
}

# Save the evaluation metrics to a JSON file for future reference
with open("reports/evaluation_metrics.json", "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)