import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, call

# Mock external downloads to avoid network usage during tests
# These decorators apply the mock before the module is imported
@patch('nltk.download')
@patch('nltk.data.find')
def test_imports(mock_nltk_find, mock_nltk_download):
    # We import inside the function to ensure the mocks above are applied first
    # This prevents the actual module from trying to download files on import
    from rag.Emotion_Classifier_module import EmotionClassifier, TextPreprocessor
    return EmotionClassifier, TextPreprocessor

# Retrieve the classes from the import test
EmotionClassifier, TextPreprocessor = test_imports()

class TestTextPreprocessor:
    """Tests for the text cleaning logic."""
    
    def test_lowercase_conversion(self):
        assert TextPreprocessor.preprocess("I AM HAPPY") == "happy"

    def test_stopword_removal(self):
        # "is", "a" should be removed. "great" stays.
        result = TextPreprocessor.preprocess("This is a great day")
        assert "great" in result
        assert "is" not in result
        assert "a" not in result

    def test_emoji_replacement(self):
        # We mock demoji inside this test to control the output
        with patch('Emotion_Classifier_module.demoji.replace_with_desc', return_value="smiling face"):
            result = TextPreprocessor.preprocess("I feel 😊")
            assert "smiling face" in result
            assert "😊" not in result

    def test_empty_string_handling(self):
        assert TextPreprocessor.preprocess("") == ""
        assert TextPreprocessor.preprocess(None) == ""


class TestEmotionClassifier:
    """Tests for the Emotion Classifier logic."""

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture to mock all heavy dependencies (Torch, Transformers, Logger)."""
        with patch('Emotion_Classifier_module.AutoTokenizer') as mock_tokenizer_cls, \
             patch('Emotion_Classifier_module.AutoModelForSequenceClassification') as mock_model_cls, \
             patch('Emotion_Classifier_module.get_logger') as mock_logger, \
             patch('Emotion_Classifier_module.torch') as mock_torch:
             
            # 1. Mock Device Detection
            mock_torch.cuda.is_available.return_value = False
            mock_torch.return_value = torch.tensor([[0.1, 0.9]]) # Dummy tensor for return

            # 2. Mock Tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            # 3. Mock Model
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            
            # Mock config for labels
            mock_model.config.id2label = {0: "sadness", 1: "joy"}

            # 4. Mock Model Forward Pass Output
            # We use actual torch tensors for math operations (softmax/argmax) to work
            mock_outputs = MagicMock()
            mock_outputs.logits = torch.tensor([[0.1, 5.0]]) # Index 1 (joy) has higher score
            mock_model.return_value = mock_outputs

            yield {
                "tokenizer_cls": mock_tokenizer_cls,
                "model_cls": mock_model_cls,
                "tokenizer": mock_tokenizer,
                "model": mock_model,
                "torch": mock_torch,
                "logger": mock_logger
            }

    def test_init_cpu_mode(self, mock_dependencies):
        """Test that init sets device to CPU if CUDA is unavailable."""
        mock_dependencies['torch'].cuda.is_available.return_value = False
        
        classifier = EmotionClassifier(model_path="dummy_path")
        
        assert classifier.device == "cpu"
        mock_dependencies['model'].to.assert_called_once_with("cpu")

    def test_init_gpu_mode(self, mock_dependencies):
        """Test that init sets device to CUDA if available."""
        mock_dependencies['torch'].cuda.is_available.return_value = True
        
        classifier = EmotionClassifier(model_path="dummy_path")
        
        assert classifier.device == "cuda"
        mock_dependencies['model'].to.assert_called_once_with("cuda")

    def test_predict_success(self, mock_dependencies):
        """Test the full prediction pipeline."""
        classifier = EmotionClassifier(model_path="dummy_path")
        
        # Input text
        result = classifier.predict("I am very happy")
        
        # Assertions
        assert result['emotion'] == "joy"
        # Softmax of [0.1, 5.0] is approx [0.008, 0.991]
        assert result['confidence'] > 0.99 
        assert result['reliable'] is True # Assuming reliable is calculated, else ignore
        
        # Verify tokenizer was called
        mock_dependencies['tokenizer'].assert_called_once()
        
        # Verify model was called in eval mode
        mock_dependencies['model'].eval.assert_called_once()

    def test_predict_empty_text(self, mock_dependencies):
        """Test handling of empty input."""
        classifier = EmotionClassifier(model_path="dummy_path")
        
        result = classifier.predict("")
        
        assert result['emotion'] is None
        assert result['confidence'] == 0.0
        
        # Verify warning was logged
        mock_dependencies['logger'].return_value.warning.assert_called()

    def test_preprocessor_is_called(self, mock_dependencies):
        """Verify that predict calls the TextPreprocessor."""
        with patch.object(TextPreprocessor, 'preprocess', return_value="clean text") as mock_preprocess:
            classifier = EmotionClassifier(model_path="dummy_path")
            classifier.predict("RAW TEXT")
            
            # Assert preprocess was called with the raw text
            mock_preprocess.assert_called_once_with("RAW TEXT")