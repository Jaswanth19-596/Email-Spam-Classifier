"""
Flask Application for Spam Email Detection
Integrated with MLflow for model management
"""

from flask import Flask, render_template, request, jsonify
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
text_cleaner = None


def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, text_cleaner
    vectorizer = joblib.load('models/text_vectorizer.pkl')
    text_cleaner = joblib.load('models/text_cleaner.pkl')

    model = mlflow.sklearn.load_model('models:/spam_classifier/Staging')

    print(model)




@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if email is spam or not"""
    try:
        # Get email text from request
        data = request.get_json()
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({
                'error': 'Please enter an email text'
            }), 400
        
        # Transform text using vectorizer
        email_vectorized = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(email_vectorized)[0]
        probability = model.predict_proba(email_vectorized)[0]
        
        # Get confidence
        confidence = max(probability) * 100
        
        # Prepare response
        result = {
            'prediction': 'Spam' if prediction == 1 else 'Not Spam (Ham)',
            'is_spam': bool(prediction == 1),
            'confidence': f"{confidence:.2f}%",
            'spam_probability': f"{probability[1] * 100:.2f}%",
            'ham_probability': f"{probability[0] * 100:.2f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Error making prediction: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load or train model on startup
    print("Initializing spam detection model...")
    load_model()
    print("Model ready!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)