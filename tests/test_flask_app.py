"""
Basic tests for Flask spam detection app
"""
import pytest
import json
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import sys
import os

# Add flask_app to path so we can import it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../flask_app')))


@pytest.fixture
def app():
    """Create Flask app for testing"""
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture(autouse=True)
def mock_model():
    """Mock the model so we don't load it from MLflow"""
    mock = Mock()
    mock.predict.return_value = np.array([0])  # Default: not spam
    
    with patch('app.model', mock), \
         patch('app.vectorizer') as mock_vec, \
         patch('app.text_cleaner') as mock_clean:
        
        # Setup mocks
        mock_vec.transform.return_value.toarray.return_value = np.zeros((1, 10000))
        mock_clean.transform.return_value = pd.Series(["cleaned"])
        
        yield mock


class TestBasicFunctionality:
    """Basic tests for the Flask app"""
    
    def test_home_page_works(self, client):
        """Home page should load"""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_predict_returns_result(self, client):
        """Predict endpoint should return a result"""
        response = client.post('/predict',
            data=json.dumps({'email': 'Hello world'}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'is_spam' in data
    
    def test_predict_empty_email_fails(self, client):
        """Empty email should return error"""
        response = client.post('/predict',
            data=json.dumps({'email': ''}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_spam_detection(self, client, mock_model):
        """Should detect spam when model predicts 1"""
        mock_model.predict.return_value = np.array([1])
        
        response = client.post('/predict',
            data=json.dumps({'email': 'WIN FREE MONEY'}),
            content_type='application/json'
        )
        
        data = response.get_json()
        assert data['prediction'] == 'Spam'
        assert data['is_spam'] is True
    
    def test_predict_ham_detection(self, client, mock_model):
        """Should detect ham when model predicts 0"""
        mock_model.predict.return_value = np.array([0])
        
        response = client.post('/predict',
            data=json.dumps({'email': 'Hello friend'}),
            content_type='application/json'
        )
        
        data = response.get_json()
        assert data['prediction'] == 'Not Spam (Ham)'
        assert data['is_spam'] is False