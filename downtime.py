
from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import joblib

app = Flask(__name__)

# Initialize variables
model = None
scaler = None
uploaded_data = None
feature_names = None

# Define the path to your CSV file
CSV_FILE_PATH = "MachineDowntime.csv"  # Replace with your actual path

# Endpoint: Load Dataset from a fixed path
@app.route('/load_data', methods=['GET'])
def load_data():
    global uploaded_data

    # Check if the CSV file exists at the specified path
    if not os.path.exists(CSV_FILE_PATH):
        return jsonify({"error": "CSV file not found at the specified path."}), 404

    # Read the CSV file into a DataFrame
    uploaded_data = pd.read_csv(CSV_FILE_PATH)
    return jsonify({"message": "File loaded successfully"}), 200

# Endpoint: Train Model
@app.route('/train', methods=['POST'])
def train():
    global model, scaler, uploaded_data, feature_names

    if uploaded_data is None:
        return jsonify({"error": "No data loaded. Use /load_data endpoint first."}), 400

    try:
        # Preprocess the data
        data = uploaded_data.copy()
        if 'Downtime' in data.columns:
            data['Downtime'] = data['Downtime'].apply(lambda x: 1 if x == "Machine_Failure" else 0)
        else:
            return jsonify({"error": "Column 'Downtime' not found in the dataset."}), 400

        # Handle missing values
        numeric_cols = data.select_dtypes(include=["number"]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

        non_numeric_cols = data.select_dtypes(exclude=["number"]).columns
        data[non_numeric_cols] = data[non_numeric_cols].fillna("Unknown")

        # Encode categorical variables
        data = pd.get_dummies(data, drop_first=True)

        # Separate features and target variable
        target_column = "Downtime"
        if target_column not in data.columns:
            return jsonify({"error": f"Target column '{target_column}' is missing after preprocessing."}), 400

        X = data.drop(columns=["Date", "Machine_ID", "Assembly_Line_No", target_column], errors="ignore")
        y = data[target_column]

        # Save feature names
        feature_names = X.columns.tolist()
        with open("feature_names.pkl", "wb") as f:
            joblib.dump(feature_names, f)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate the model (optional)
        accuracy = model.score(X_test, y_test)

        # Save the model and scaler
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        return jsonify({"message": "Model trained successfully", "accuracy": accuracy}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint: Predict Downtime
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, feature_names

    # Check if model and scaler are available
    if model is None or scaler is None:
        return jsonify({"error": "Model not trained. Use /train endpoint first."}), 400

    # Load feature names if not already loaded
    if feature_names is None:
        try:
            with open("feature_names.pkl", "rb") as f:
                feature_names = joblib.load(f)
        except FileNotFoundError:
            return jsonify({"error": "Feature names file not found. Train the model first."}), 500

    # Get JSON input
    input_data = request.json
    if not input_data:
        return jsonify({"error": "Invalid input. Provide JSON data."}), 400

    try:
        # Convert JSON input to DataFrame
        sample_data = pd.DataFrame([input_data])

        # Align sample_data to match feature names
        for col in feature_names:
            if col not in sample_data.columns:
                sample_data[col] = 0
        sample_data = sample_data[feature_names]

        # Standardize features
        sample_data_scaled = scaler.transform(sample_data)

        # Predict downtime and confidence
        downtime_prediction = model.predict(sample_data_scaled)[0]
        confidence_score = model.predict_proba(sample_data_scaled)[0][1]

        # Format the output
        output = {
            "Downtime": "Yes" if downtime_prediction == 1 else "No",
            "Confidence": round(confidence_score, 2)
        }

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

