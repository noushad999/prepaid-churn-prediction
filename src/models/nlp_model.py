import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class NLPSentimentAnalyzer(nn.Module):
    """
    NLP model for analyzing customer sentiment from support tickets and complaints.
    Uses transformer-based architecture (DistilBERT) for efficiency.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 128)
        self.dropout = config.get('dropout', 0.3)
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Get transformer hidden size
        self.hidden_size = self.transformer.config.hidden_size
        
        # Sentiment classification head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 3)  # Negative, Neutral, Positive
        )
        
        # Feature extractor for churn prediction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Sentiment prediction
        sentiment_logits = self.sentiment_classifier(cls_output)
        
        # Feature extraction for churn model
        nlp_features = self.feature_extractor(cls_output)
        
        return sentiment_logits, nlp_features
    
    def tokenize_texts(self, texts: List[str]) -> Dict:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def extract_sentiment_features(self, texts: List[str]) -> torch.Tensor:
        """Extract sentiment features from customer texts."""
        self.eval()
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenize_texts(texts)
            
            # Forward pass
            _, features = self(encoded['input_ids'], encoded['attention_mask'])
            
            return features
