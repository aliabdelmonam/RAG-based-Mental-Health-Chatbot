import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import only LanguageDetector since TextPreprocessor was removed
from rag.Language_Detection_module.Language_detector import LanguageDetector


class TestLanguageDetector:
    """Tests for the LanguageDetector class."""

    @pytest.fixture
    def mock_model(self):
        """
        Creates a mock model object to simulate joblib.load behavior.
        Updated to match the new list of 20 languages.
        """
        model = MagicMock()
        
        # Updated list of 20 languages
        model.classes_ = np.array(['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'nl', 'pl', 'pt', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'])
        
        # Default behavior: return array shape (1, 20) matching classes_ size
        # Defaults to high confidence for 'ar' (index 0)
        probs = np.zeros((1, 20))
        probs[0, 0] = 0.95
        model.predict_proba.return_value = probs
        
        return model

    def test_init_loads_model(self, mock_model):
        """Test that the initializer calls joblib.load."""
        with patch('joblib.load', return_value=mock_model) as mock_load:
            detector = LanguageDetector(model_path="dummy_path.pkl")
            
            mock_load.assert_called_once_with("dummy_path.pkl")
            assert detector.model is not None

    def test_init_failure_raises_error(self):
        """Test that a loading error raises RuntimeError."""
        with patch('joblib.load', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                LanguageDetector(model_path="bad_path.pkl")

    def test_predict_high_confidence(self, mock_model):
        """Test successful prediction when confidence > threshold."""
        # Setup mock to return high confidence for 'ar' (index 0)
        # Create an array of 20 elements, set index 0 to 0.80
        probs = np.zeros((1, 20))
        probs[0, 0] = 0.80 
        
        mock_model.predict_proba.return_value = probs
        
        with patch('joblib.load', return_value=mock_model):
            detector = LanguageDetector(model_path="dummy.pkl", threshold=0.60)
            result = detector.predict("some text")

        assert result['language'] == 'ar'
        assert result['confidence'] == 0.80
        assert result['reliable'] is True

    def test_predict_low_confidence(self, mock_model):
        """Test 'uncertain' result when confidence < threshold."""
        # Setup mock to return low max confidence (e.g., 0.40)
        # Since we have 20 classes, we distribute probabilities such that max is low
        probs = np.full((1, 20), 0.03) # 0.03 * 20 = 0.60 total (approx)
        probs[0, 0] = 0.40 # Set highest probability
        
        mock_model.predict_proba.return_value = probs
        
        with patch('joblib.load', return_value=mock_model):
            detector = LanguageDetector(model_path="dummy.pkl", threshold=0.60)
            result = detector.predict("ambiguous text")

        assert result['language'] == "uncertain"
        assert result['confidence'] == 0.40
        assert result['reliable'] is False

    def test_predict_direct_text_input(self, mock_model):
        """
        Verify that predict passes text to the model.
        """
        with patch('joblib.load', return_value=mock_model):
            detector = LanguageDetector(model_path="dummy.pkl")
            detector.predict("RAW TEXT")
        
            mock_model.predict_proba.assert_called_once_with(["raw text"])