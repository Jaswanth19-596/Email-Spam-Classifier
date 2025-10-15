"""
Flask Application for Spam Email Detection
Integrated with MLflow for model management
"""

from flask import Flask, render_template, request, jsonify
import mlflow
import mlflow.pyfunc
import joblib
from utils.text_cleaner import TextCleaner
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
text_cleaner = None


os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/madhajaswanth@gmail.com/TempExperiment")


def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, text_cleaner
    vectorizer = joblib.load("models/text_vectorizer.pkl")
    text_cleaner = TextCleaner()

    model = mlflow.pyfunc.load_model(
        "models:/development.models.spam_classifier@staging"
    )


@app.route("/")
def home():
    """Render the main page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict if email is spam or not"""
    try:
        # Get email text from request
        data = request.get_json()
        email_text = data.get("email", "")

        if not email_text:
            return jsonify({"error": "Please enter an email text"}), 400

        cleaned_email = text_cleaner.transform(pd.Series([email_text]))

        # Transform text using vectorizer
        email_vectorized = vectorizer.transform(cleaned_email).toarray()
        num_features = email_vectorized.shape[1]
        column_names = [str(i) for i in range(num_features)]

        email_vectorized_df = pd.DataFrame(email_vectorized, columns=column_names)

        # Make prediction
        prediction = model.predict(email_vectorized_df)[0]

        # Prepare response
        result = {
            "prediction": "Spam" if prediction == 1 else "Not Spam (Ham)",
            "is_spam": bool(prediction == 1),
            # 'confidence': f"{confidence:.2f}%",
            # 'spam_probability': f"{probability[1] * 100:.2f}%",
            # 'ham_probability': f"{probability[0] * 100:.2f}%"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500


if __name__ == "__main__":
    # Load or train model on startup
    print("Initializing spam detection model...")
    load_model()
    print("Model ready!")

    # Run the Flask app
    app.run(debug=False, host="0.0.0.0")
