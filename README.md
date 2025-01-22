Downtime Prediction API
This Flask-based API allows users to predict machine downtime using a logistic regression model trained on a dataset. The script provides endpoints to load a dataset, train the model, and predict downtime based on input features.

Features
Load Dataset: Load data from a predefined CSV file (MachineDowntime.csv).
Train Model: Train a logistic regression model using the loaded dataset.
Predict Downtime: Predict whether a machine will face downtime based on provided input.

Setup Instructions
Prerequisites
Python 3.8 or higher
Required Python packages (Flask, pandas, scikit-learn, joblib)
Installation Steps
Clone or download the project files to your local system.
Navigate to the project directory and install the required dependencies (pip install -r requirements.txt)

Ensure the MachineDowntime.csv file is in the project directory or update the CSV_FILE_PATH in the script with its correct location.

Usage Instructions
Start the Flask Server
Run the script using: python downtime.py
By default, the server will start at http://127.0.0.1:5000/.

Endpoints
1.Load Dataset
URL: /load_data
Method: GET
Description: Loads the dataset from the specified path.
Response: Success message or error if the file is missing.
2.Train Model
URL: /train
Method: POST
Description: Trains a logistic regression model on the loaded data.
Response: Success message with the model's accuracy or error if data isn't loaded.
3.Predict Downtime
URL: /predict
Method: POST
Description: Predicts machine downtime based on input features.
Request Body: JSON object containing the input features.
Response: Downtime prediction (Yes/No) and confidence score.

File Descriptions
downtime.py: The main script containing the API logic.
MachineDowntime.csv: Example dataset (should be placed in the project directory).
requirements.txt: List of required Python packages.
feature_names.pkl: Stores feature names for prediction (generated during training).
model.pkl: Trained model file (created after training).
scaler.pkl: Feature scaler file (created after training).

Notes
Ensure the MachineDowntime.csv file includes a Downtime column, which must be labeled as Machine_Failure for failures and other values otherwise.
Missing values are handled by imputing numerical columns with their mean and filling non-numerical columns with "Unknown".
The model and scaler files are automatically saved after training.

Troubleshooting
CSV File Not Found: Ensure the file path in CSV_FILE_PATH is correct.
Model Not Trained: Use /train after loading the dataset.
Feature Mismatch: Ensure input features for /predict match those used during training.


