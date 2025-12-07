import torch
import torch.nn as nn

class BiLSTM_NER(nn.Module):
    """
    Baseline BiLSTM model for NER task
    
    Architecture:
    - Embedding layer
    - Bidirectional LSTM
    - Linear classification layer
    
    Parameters:
    - vocab_size: Size of vocabulary
    - emb_dim: Embedding dimension (default: 128)
    - hidden_dim: LSTM hidden dimension (default: 256)
    - num_labels: Number of output classes (default: 11)
    - dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_labels=11, dropout=0.1):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, num_labels]
        """
        # Embedding layer
        x = self.embedding(input_ids)  # [batch_size, seq_len, emb_dim]
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Classification layer
        out = self.fc(lstm_out)  # [batch_size, seq_len, num_labels]
        
        return out


def create_model(vocab_size, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs):
    """
    Create and initialize model
    
    Args:
        vocab_size: Size of vocabulary
        device: Device to place model on
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    model = BiLSTM_NER(vocab_size, **kwargs)
    return model.to(device)


if __name__ == "__main__":
    # Test model
    vocab_size = 30522  # BERT base uncased
    model = create_model(vocab_size)
    
    # Create dummy input
    batch_size, seq_len = 4, 128
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")