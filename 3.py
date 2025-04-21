import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from torch.utils.data import Dataset, DataLoader
import time
import pickle
from tqdm import tqdm


class BasketBERT4RecConfig:
    """Configuration class for BasketBERT4Rec model."""
    
    def __init__(
        self,
        vocab_size=50000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=50,
        max_basket_size=20,
        basket_embedding_type='mean',  # Options: 'mean', 'max', 'attention'
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        mask_token_id=1,
    ):
        """Initialize the BasketBERT4RecConfig."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_basket_size = max_basket_size
        self.basket_embedding_type = basket_embedding_type
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
    
    @classmethod
    def from_json_file(cls, json_file):
        """Load config from a JSON file."""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save_to_json_file(self, json_file):
        """Save config to a JSON file."""
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class BasketEmbedding(nn.Module):
    """
    Module to create embeddings for baskets of items.
    Converts a basket of item IDs to a single embedding vector.
    """
    
    def __init__(self, config):
        super(BasketEmbedding, self).__init__()
        self.config = config
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # If using attention for basket embedding
        if config.basket_embedding_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )
        
        # Normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_baskets, basket_masks=None):
        """
        Args:
            input_baskets: Tensor of shape [batch_size, seq_length, basket_size]
                containing item IDs for each basket.
            basket_masks: Tensor of shape [batch_size, seq_length, basket_size]
                with 1s for valid items and 0s for padding.
        
        Returns:
            basket_embeddings: Tensor of shape [batch_size, seq_length, hidden_size]
                containing embeddings for each basket.
        """
        batch_size, seq_length, basket_size = input_baskets.size()
        
        # Reshape to embed all items
        flat_items = input_baskets.view(-1, basket_size)  # [batch_size*seq_length, basket_size]
        flat_masks = basket_masks.view(-1, basket_size) if basket_masks is not None else None
        
        # Get item embeddings [batch_size*seq_length, basket_size, hidden_size]
        item_embeddings = self.item_embeddings(flat_items)
        
        # Aggregate item embeddings to basket embeddings
        if self.config.basket_embedding_type == 'mean':
            # Mean pooling with mask
            if flat_masks is not None:
                # Expand mask for proper broadcasting
                expanded_mask = flat_masks.unsqueeze(-1).expand_as(item_embeddings)
                # Mask out padding items
                masked_embeddings = item_embeddings * expanded_mask
                # Sum and divide by count of valid items
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                # Avoid division by zero
                item_counts = torch.sum(flat_masks, dim=1, keepdim=True).clamp(min=1)
                basket_embeddings = sum_embeddings / item_counts
            else:
                # Simple mean if no mask
                basket_embeddings = torch.mean(item_embeddings, dim=1)
        
        elif self.config.basket_embedding_type == 'max':
            # Max pooling with mask
            if flat_masks is not None:
                # Create a large negative value mask for padding items
                neg_inf = torch.ones_like(item_embeddings) * -1e9
                neg_inf = neg_inf * (1 - flat_masks.unsqueeze(-1))
                # Set padding positions to large negative values
                masked_embeddings = item_embeddings + neg_inf
                # Max pooling
                basket_embeddings, _ = torch.max(masked_embeddings, dim=1)
            else:
                # Simple max if no mask
                basket_embeddings, _ = torch.max(item_embeddings, dim=1)
        
        elif self.config.basket_embedding_type == 'attention':
            # Attention-based pooling
            attention_scores = self.attention(item_embeddings).squeeze(-1)  # [batch*seq, basket_size]
            
            if flat_masks is not None:
                # Mask attention scores for padding items
                attention_scores = attention_scores.masked_fill(flat_masks == 0, -1e9)
            
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch*seq, basket_size]
            
            # Apply attention weights to get weighted sum
            basket_embeddings = torch.bmm(
                attention_weights.unsqueeze(1),
                item_embeddings
            ).squeeze(1)  # [batch*seq, hidden_size]
        
        # Reshape back to [batch_size, seq_length, hidden_size]
        basket_embeddings = basket_embeddings.view(batch_size, seq_length, -1)
        
        # Apply layer normalization and dropout
        basket_embeddings = self.LayerNorm(basket_embeddings)
        basket_embeddings = self.dropout(basket_embeddings)
        
        return basket_embeddings


class PositionalEmbedding(nn.Module):
    """
    Positional embeddings for transformer model.
    """
    
    def __init__(self, config):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, basket_embeddings, sequence_mask=None):
        """
        Args:
            basket_embeddings: Tensor of shape [batch_size, seq_length, hidden_size]
            sequence_mask: Tensor of shape [batch_size, seq_length] with 1s for valid
                positions and 0s for padding.
        
        Returns:
            position_embeddings: Tensor of shape [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length, _ = basket_embeddings.size()
        
        # Create position IDs tensor
        position_ids = torch.arange(seq_length, dtype=torch.long, device=basket_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        # If sequence mask is provided, set position IDs for padding to 0
        if sequence_mask is not None:
            position_ids = position_ids * sequence_mask
        
        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add position embeddings to basket embeddings
        embeddings = basket_embeddings + position_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention computation."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_length, head_size)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Tensor of shape [batch_size, 1, seq_length, seq_length]
                with 1s for valid positions and 0s for masked positions.
        
        Returns:
            attention_output: Tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Project query, key, value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        attention_output = self.output(context_layer)
        
        return attention_output


class FeedForward(nn.Module):
    """Feed-forward network for transformer."""
    
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
        
        Returns:
            ffn_output: Tensor of shape [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Tensor for attention masking
        
        Returns:
            layer_output: Tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout1(attention_output)
        hidden_states = self.layernorm1(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        ffn_output = self.feed_forward(hidden_states)
        ffn_output = self.dropout2(ffn_output)
        layer_output = self.layernorm2(hidden_states + ffn_output)
        
        return layer_output


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Tensor for attention masking
        
        Returns:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states


class BasketBERT4Rec(nn.Module):
    """
    BERT4Rec model adapted for basket-level recommendations.
    """
    
    def __init__(self, config):
        super(BasketBERT4Rec, self).__init__()
        self.config = config
        
        # Embeddings
        self.basket_embedding = BasketEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(config)
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_baskets, basket_masks=None, sequence_mask=None, masked_positions=None):
        """
        Args:
            input_baskets: Tensor of shape [batch_size, seq_length, basket_size]
                containing item IDs for each basket.
            basket_masks: Tensor of shape [batch_size, seq_length, basket_size]
                with 1s for valid items and 0s for padding.
            sequence_mask: Tensor of shape [batch_size, seq_length]
                with 1s for valid baskets and 0s for padding.
            masked_positions: Tensor of shape [batch_size, num_masked]
                containing indices of positions to predict.
        
        Returns:
            prediction_scores: Tensor of shape [batch_size, num_masked, vocab_size]
                containing prediction scores for masked positions.
        """
        device = input_baskets.device
        batch_size, seq_length = input_baskets.size()[:2]
        
        # Get basket embeddings
        basket_embeddings = self.basket_embedding(input_baskets, basket_masks)
        
        # Add positional embeddings
        embeddings = self.position_embedding(basket_embeddings, sequence_mask)
        
        # Create attention mask for padding
        if sequence_mask is not None:
            # Convert sequence mask to attention mask
            # Shape: [batch_size, 1, 1, seq_length]
            attention_mask = sequence_mask.unsqueeze(1).unsqueeze(2)
            
            # Create mask that allows each position to attend to all valid positions
            # Shape: [batch_size, 1, seq_length, seq_length]
            attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
            
            # Convert 0s to large negative value for softmax
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None
        
        # Apply transformer encoder
        sequence_output = self.encoder(embeddings, attention_mask)
        
        # Get predictions for masked positions or all positions
        if masked_positions is not None:
            # Gather embeddings at masked positions
            # Reshape masked_positions for gathering: [batch_size*num_masked, 1]
            batch_size, num_masked = masked_positions.size()
            gathered_output = torch.gather(
                sequence_output,
                dim=1,
                index=masked_positions.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            )
            
            # Get prediction scores
            prediction_scores = self.output_layer(gathered_output)
        else:
            # Get prediction scores for all positions
            prediction_scores = self.output_layer(sequence_output)
        
        return prediction_scores
    
    def get_item_embeddings(self):
        """Return the item embeddings weight matrix."""
        return self.basket_embedding.item_embeddings.weight.detach()


class BasketBERT4RecDataset(Dataset):
    """Dataset for training BasketBERT4Rec model."""
    
    def __init__(self, sequences, sequence_masks, basket_masks, 
                 masked_positions=None, masked_labels=None, 
                 targets_padded=None, targets_multihot=None,
                 mode='mlm', max_mask_len=10):
        """
        Initialize the dataset.
        
        Args:
            sequences: Tensor of shape [num_seqs, seq_length, basket_size]
                containing item IDs for each basket.
            sequence_masks: Tensor of shape [num_seqs, seq_length]
                with 1s for valid baskets and 0s for padding.
            basket_masks: Tensor of shape [num_seqs, seq_length, basket_size]
                with 1s for valid items and 0s for padding.
            masked_positions: List of lists containing indices of masked positions.
            masked_labels: List of lists containing original basket values at masked positions.
            targets_padded: Tensor of shape [num_seqs, basket_size]
                containing item IDs for the target basket (next basket prediction).
            targets_multihot: Tensor of shape [num_seqs, vocab_size]
                containing multi-hot encoding of the target basket.
            mode: Training mode - 'mlm' for masked language modeling, 
                'next_basket' for next basket prediction.
            max_mask_len: Maximum number of masked positions to handle in a batch
        """
        self.sequences = sequences
        self.sequence_masks = sequence_masks
        self.basket_masks = basket_masks
        self.masked_positions = masked_positions
        self.masked_labels = masked_labels
        self.targets_padded = targets_padded
        self.targets_multihot = targets_multihot
        self.mode = mode
        self.max_mask_len = max_mask_len
        
        # Determine basket size from the data
        self.basket_size = sequences.shape[2] if hasattr(sequences, 'shape') else sequences[0].shape[1]
        
        # Convert to tensors if not already
        if isinstance(self.sequences, np.ndarray):
            self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        if isinstance(self.sequence_masks, np.ndarray):
            self.sequence_masks = torch.tensor(self.sequence_masks, dtype=torch.long)
        if isinstance(self.basket_masks, np.ndarray):
            self.basket_masks = torch.tensor(self.basket_masks, dtype=torch.long)
        if isinstance(self.targets_padded, np.ndarray):
            self.targets_padded = torch.tensor(self.targets_padded, dtype=torch.long)
        if isinstance(self.targets_multihot, np.ndarray):
            self.targets_multihot = torch.tensor(self.targets_multihot, dtype=torch.float)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.mode == 'mlm':
            # Get masked positions for this example
            if self.masked_positions is not None and idx < len(self.masked_positions):
                positions = self.masked_positions[idx]
                
                # If no masked positions, create a dummy mask position at position 0
                if not positions:
                    masked_positions = torch.zeros(1, dtype=torch.long)
                    masked_labels = torch.zeros((1, self.basket_size), dtype=torch.long)
                else:
                    # Convert positions to tensor
                    masked_positions = torch.tensor(positions, dtype=torch.long)
                    
                    # Get corresponding labels
                    labels = [torch.tensor(self.masked_labels[idx][i], dtype=torch.long) 
                             for i in range(len(positions))]
                    
                    # Pad labels if needed
                    padded_labels = []
                    for label in labels:
                        if len(label) < self.basket_size:
                            padding = torch.zeros(self.basket_size - len(label), dtype=torch.long)
                            label = torch.cat([label, padding])
                        padded_labels.append(label)
                    
                    # Stack labels
                    masked_labels = torch.stack(padded_labels)
            else:
                # No masked positions for this example
                masked_positions = torch.zeros(1, dtype=torch.long)
                masked_labels = torch.zeros((1, self.basket_size), dtype=torch.long)
            
            return {
                'sequences': self.sequences[idx],
                'sequence_masks': self.sequence_masks[idx],
                'basket_masks': self.basket_masks[idx],
                'masked_positions': masked_positions,
                'masked_labels': masked_labels,
                'num_masks': len(masked_positions)  # Store actual number of masks
            }
        
        elif self.mode == 'next_basket':
            return {
                'sequences': self.sequences[idx],
                'sequence_masks': self.sequence_masks[idx],
                'basket_masks': self.basket_masks[idx],
                'target_padded': self.targets_padded[idx],
                'target_multihot': self.targets_multihot[idx]
            }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length masked positions.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Batched tensors with consistent sizes
    """
    # Determine max number of masked positions in this batch
    max_masks = max([item['num_masks'] for item in batch]) if 'num_masks' in batch[0] else 0
    
    # Initialize batch dictionary
    batch_dict = {}
    
    # Process each key in the batch
    for key in batch[0].keys():
        if key == 'masked_positions':
            # Pad masked positions to the same length
            padded_positions = []
            for item in batch:
                pos = item['masked_positions']
                if len(pos) < max_masks:
                    # Pad with zeros
                    padding = torch.zeros(max_masks - len(pos), dtype=pos.dtype)
                    pos = torch.cat([pos, padding])
                elif len(pos) > max_masks:
                    # Truncate if too long (shouldn't happen with our num_masks tracking)
                    pos = pos[:max_masks]
                padded_positions.append(pos)
            
            batch_dict[key] = torch.stack(padded_positions)
            
        elif key == 'masked_labels':
            # Pad masked labels to the same length
            padded_labels = []
            for item in batch:
                labels = item['masked_labels']
                if len(labels) < max_masks:
                    # Create padding: [max_masks - len(labels), basket_size]
                    padding = torch.zeros(
                        (max_masks - len(labels), labels.size(1)), 
                        dtype=labels.dtype
                    )
                    labels = torch.cat([labels, padding], dim=0)
                elif len(labels) > max_masks:
                    # Truncate if too long
                    labels = labels[:max_masks]
                padded_labels.append(labels)
            
            batch_dict[key] = torch.stack(padded_labels)
            
        elif key != 'num_masks':  # Skip num_masks as it was just for processing
            # Standard handling for other keys
            batch_dict[key] = torch.stack([item[key] for item in batch])
    
    return batch_dict


class BasketBERT4RecTrainer:
    """Trainer for BasketBERT4Rec model."""
    
    def __init__(self, model, train_dataset, val_dataset=None, 
                 batch_size=32, lr=0.001, weight_decay=0.01,
                 num_epochs=10, device='cuda', output_dir='./model_output/',
                 mlm_weight=1.0, nbr_weight=1.0):
        """
        Initialize the trainer.
        
        Args:
            model: BasketBERT4Rec model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Training batch size
            lr: Learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            device: Device to train on ('cuda' or 'cpu')
            output_dir: Directory to save model outputs
            mlm_weight: Weight for masked language modeling loss
            nbr_weight: Weight for next basket recommendation loss
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        self.mlm_weight = mlm_weight
        self.nbr_weight = nbr_weight
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create data loaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=2,
            pin_memory=True if device == 'cuda' else False
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=2,
                pin_memory=True if device == 'cuda' else False
            )
        else:
            self.val_loader = None
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize training state
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        self.early_stop_patience = 5
    
    def train(self):
        """Train the model for the specified number of epochs."""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(epoch)
            
            # Validation phase
            if self.val_loader:
                val_loss, val_metrics = self._validate_epoch(epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.no_improve_epochs = 0
                    # Save best model
                    self._save_model('best_model.pt')
                else:
                    self.no_improve_epochs += 1
                    if self.no_improve_epochs >= self.early_stop_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Always save the latest model
            self._save_model(f'model_epoch_{epoch+1}.pt')
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, ", end="")
            if self.val_loader:
                print(f"Val Loss: {val_loss:.4f}")
            else:
                print("")
        
        print("Training completed!")
        
        # Return metrics for evaluation
        return {
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss if self.val_loader else None,
            'val_metrics': val_metrics if self.val_loader else None
        }
    
    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_nbr_loss = 0
        num_batches = len(self.train_loader)
        
        # Track correct predictions for MLM
        total_mlm_items = 0
        correct_mlm_items = 0
        
        # Track metrics for next basket prediction
        total_nbr_items = 0
        total_recall = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss, metrics = self._compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mlm_loss += metrics.get('mlm_loss', 0)
            total_nbr_loss += metrics.get('nbr_loss', 0)
            
            if 'mlm_correct' in metrics and 'mlm_total' in metrics:
                total_mlm_items += metrics['mlm_total']
                correct_mlm_items += metrics['mlm_correct']
            
            if 'recall' in metrics and 'nbr_total' in metrics:
                total_nbr_items += metrics['nbr_total']
                total_recall += metrics['recall'] * metrics['nbr_total']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mlm_loss': f"{metrics.get('mlm_loss', 0):.4f}",
                'nbr_loss': f"{metrics.get('nbr_loss', 0):.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches
        avg_nbr_loss = total_nbr_loss / num_batches
        
        metrics = {
            'avg_mlm_loss': avg_mlm_loss,
            'avg_nbr_loss': avg_nbr_loss
        }
        
        if total_mlm_items > 0:
            metrics['mlm_accuracy'] = correct_mlm_items / total_mlm_items
        
        if total_nbr_items > 0:
            metrics['recall'] = total_recall / total_nbr_items
        
        return avg_loss, metrics
    
    def _validate_epoch(self, epoch):
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        total_mlm_loss = 0
        total_nbr_loss = 0
        num_batches = len(self.val_loader)
        
        # Track correct predictions for MLM
        total_mlm_items = 0
        correct_mlm_items = 0
        
        # Track metrics for next basket prediction
        total_nbr_items = 0
        total_recall = 0
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass and loss calculation
                loss, metrics = self._compute_loss(batch)
                
                # Update metrics
                total_loss += loss.item()
                total_mlm_loss += metrics.get('mlm_loss', 0)
                total_nbr_loss += metrics.get('nbr_loss', 0)
                
                if 'mlm_correct' in metrics and 'mlm_total' in metrics:
                    total_mlm_items += metrics['mlm_total']
                    correct_mlm_items += metrics['mlm_correct']
                
                if 'recall' in metrics and 'nbr_total' in metrics:
                    total_nbr_items += metrics['nbr_total']
                    total_recall += metrics['recall'] * metrics['nbr_total']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mlm_loss': f"{metrics.get('mlm_loss', 0):.4f}",
                    'nbr_loss': f"{metrics.get('nbr_loss', 0):.4f}"
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches
        avg_nbr_loss = total_nbr_loss / num_batches
        
        metrics = {
            'avg_mlm_loss': avg_mlm_loss,
            'avg_nbr_loss': avg_nbr_loss
        }
        
        if total_mlm_items > 0:
            metrics['mlm_accuracy'] = correct_mlm_items / total_mlm_items
        
        if total_nbr_items > 0:
            metrics['recall'] = total_recall / total_nbr_items
        
        return avg_loss, metrics
    
    def _compute_loss(self, batch):
        """
        Compute loss for current batch.
        
        Args:
            batch: Dictionary of tensors for the current batch.
        
        Returns:
            loss: Combined loss value
            metrics: Dictionary of metric values
        """
        metrics = {}
        
        # Extract common inputs
        sequences = batch['sequences']
        sequence_masks = batch['sequence_masks']
        basket_masks = batch['basket_masks']
        
        # MLM loss calculation
        mlm_loss = torch.tensor(0.0, device=self.device)
        if 'masked_positions' in batch and 'masked_labels' in batch:
            masked_positions = batch['masked_positions']
            masked_labels = batch['masked_labels']
            
            # Skip if no masked positions
            if masked_positions.size(1) > 0:
                # Forward pass for MLM
                prediction_scores = self.model(
                    sequences, basket_masks, sequence_masks, masked_positions
                )
                
                # Compute MLM loss
                mlm_loss, mlm_metrics = self._compute_mlm_loss(
                    prediction_scores, masked_labels, masked_positions
                )
                
                metrics.update(mlm_metrics)
                metrics['mlm_loss'] = mlm_loss.item()
        
        # Next basket prediction loss calculation
        nbr_loss = torch.tensor(0.0, device=self.device)
        if 'target_padded' in batch or 'target_multihot' in batch:
            # Get the last valid basket embedding
            # (we'll use this to predict the next basket)
            last_positions = sequence_masks.sum(dim=1) - 1
            last_positions = last_positions.unsqueeze(1)  # [batch_size, 1]
            
            # Forward pass to get sequence output
            all_outputs = self.model(sequences, basket_masks, sequence_masks)
            
            # Gather last position outputs
            batch_size = sequences.size(0)
            hidden_size = all_outputs.size(-1)
            
            # Get embeddings of the last position
            last_outputs = torch.gather(
                all_outputs,
                dim=1,
                index=last_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            ).squeeze(1)  # [batch_size, hidden_size]
            
            # Predict next basket
            next_basket_logits = self.model.output_layer(last_outputs)  # [batch_size, vocab_size]
            
            # Compute NBR loss
            if 'target_multihot' in batch:
                nbr_loss, nbr_metrics = self._compute_nbr_loss(
                    next_basket_logits, batch['target_multihot']
                )
                
                metrics.update(nbr_metrics)
                metrics['nbr_loss'] = nbr_loss.item()
            elif 'target_padded' in batch:
                nbr_loss, nbr_metrics = self._compute_nbr_loss_from_padded(
                    next_basket_logits, batch['target_padded']
                )
                
                metrics.update(nbr_metrics)
                metrics['nbr_loss'] = nbr_loss.item()
        
        # Combine losses with weights
        total_loss = self.mlm_weight * mlm_loss + self.nbr_weight * nbr_loss
        
        return total_loss, metrics
    
    def _compute_mlm_loss(self, prediction_scores, masked_labels, masked_positions):
        """
        Compute MLM loss and metrics.
        
        Args:
            prediction_scores: Tensor of shape [batch_size, num_masked, vocab_size]
            masked_labels: Tensor of shape [batch_size, num_masked, basket_size]
            masked_positions: Tensor of shape [batch_size, num_masked]
        
        Returns:
            loss: MLM loss value
            metrics: Dictionary of MLM metrics
        """
        batch_size, num_masked, vocab_size = prediction_scores.size()
        
        # Initialize loss and metrics
        loss = 0
        total_items = 0
        correct_items = 0
        
        # Create a mask for valid masked positions (non-zero)
        valid_mask = (masked_positions > 0).float()  # [batch_size, num_masked]
        
        # Loop through each masked position
        for i in range(num_masked):
            # Skip invalid masked positions (zeros)
            position_valid = valid_mask[:, i]  # [batch_size]
            if position_valid.sum() == 0:
                continue
            
            # Get predictions and targets for this position
            pos_preds = prediction_scores[:, i]  # [batch_size, vocab_size]
            pos_targets = masked_labels[:, i]    # [batch_size, basket_size]
            
            # Convert targets to multi-hot encoding
            multi_hot_targets = torch.zeros(
                batch_size, vocab_size, device=prediction_scores.device
            )
            
            for b in range(batch_size):
                if position_valid[b] > 0:
                    # Get non-padding items in the basket
                    valid_items = pos_targets[b][pos_targets[b] > 0]
                    if len(valid_items) > 0:
                        multi_hot_targets[b, valid_items] = 1.0
            
            # Compute binary cross-entropy loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pos_preds, multi_hot_targets, reduction='none'
            )
            
            # Apply mask for valid positions and sum across vocabulary
            masked_loss = bce_loss.mean(dim=1) * position_valid
            
            # Add to total loss
            loss += masked_loss.sum()
            
            # Update metrics
            for b in range(batch_size):
                if position_valid[b] > 0:
                    # Get non-padding items in the basket
                    valid_items = pos_targets[b][pos_targets[b] > 0].tolist()
                    
                    if valid_items:
                        # Get top-k predictions
                        _, top_preds = torch.topk(pos_preds[b], k=len(valid_items))
                        top_preds = top_preds.tolist()
                        
                        # Count correct predictions
                        for item in valid_items:
                            if item in top_preds:
                                correct_items += 1
                        
                        total_items += len(valid_items)
        
        # Calculate average loss
        if total_items > 0:
            loss = loss / total_items
        
        # Calculate accuracy
        accuracy = correct_items / total_items if total_items > 0 else 0
        
        metrics = {
            'mlm_correct': correct_items,
            'mlm_total': total_items,
            'mlm_accuracy': accuracy
        }
        
        return loss, metrics
    
    def _compute_nbr_loss(self, logits, target_multihot):
        """
        Compute next basket recommendation loss and metrics.
        
        Args:
            logits: Tensor of shape [batch_size, vocab_size]
            target_multihot: Tensor of shape [batch_size, vocab_size]
        
        Returns:
            loss: NBR loss value
            metrics: Dictionary of NBR metrics
        """
        batch_size, vocab_size = logits.size()
        
        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target_multihot, reduction='mean'
        )
        
        # Compute recall@k metrics
        k_values = [5, 10, 20]
        recall_metrics = {}
        
        # Get top-k predictions
        _, top_preds = torch.topk(logits, k=max(k_values))
        
        # Compute recall for each k
        total_recall = 0
        for k in k_values:
            recall_sum = 0
            
            for b in range(batch_size):
                # Get ground truth items
                true_items = target_multihot[b].nonzero().squeeze(-1)
                
                if len(true_items) > 0:
                    # Get top-k predictions for this example
                    top_k = top_preds[b, :k]
                    
                    # Count correctly predicted items
                    correct = 0
                    for item in true_items:
                        if item in top_k:
                            correct += 1
                    
                    # Calculate recall
                    recall = correct / len(true_items)
                    recall_sum += recall
            
            # Average recall for this k
            recall_k = recall_sum / batch_size
            recall_metrics[f'recall@{k}'] = recall_k
            
            # Use recall@10 as the main metric
            if k == 10:
                total_recall = recall_k
        
        metrics = {
            'recall': total_recall,
            'nbr_total': batch_size,
            **recall_metrics
        }
        
        return bce_loss, metrics
    
    def _compute_nbr_loss_from_padded(self, logits, target_padded):
        """
        Compute next basket recommendation loss from padded target.
        
        Args:
            logits: Tensor of shape [batch_size, vocab_size]
            target_padded: Tensor of shape [batch_size, basket_size]
        
        Returns:
            loss: NBR loss value
            metrics: Dictionary of NBR metrics
        """
        batch_size, vocab_size = logits.size()
        _, basket_size = target_padded.size()
        
        # Convert padded targets to multi-hot encoding
        target_multihot = torch.zeros(
            batch_size, vocab_size, device=logits.device
        )
        
        for b in range(batch_size):
            # Get non-padding items in the basket
            valid_items = target_padded[b][target_padded[b] > 0]
            if len(valid_items) > 0:
                target_multihot[b, valid_items] = 1.0
        
        # Use the same loss function as with multi-hot targets
        return self._compute_nbr_loss(logits, target_multihot)
    
    def _save_model(self, filename):
        """Save model, configuration, and optimizer state."""
        model_path = os.path.join(self.output_dir, filename)
        
        # Save model state dict, config, and optimizer
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config.__dict__
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dataset: Test dataset
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        self.model.eval()
        total_loss = 0
        total_items = 0
        recall_sums = {k: 0 for k in [5, 10, 20]}
        ndcg_sums = {k: 0 for k in [5, 10, 20]}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get inputs
                sequences = batch['sequences']
                sequence_masks = batch['sequence_masks']
                basket_masks = batch['basket_masks']
                
                # For next basket prediction, we need the targets
                if 'target_multihot' in batch:
                    target_multihot = batch['target_multihot']
                else:
                    # Convert padded targets to multi-hot
                    target_padded = batch['target_padded']
                    batch_size, vocab_size = len(target_padded), self.model.config.vocab_size
                    
                    target_multihot = torch.zeros(
                        batch_size, vocab_size, device=self.device
                    )
                    
                    for b in range(batch_size):
                        valid_items = target_padded[b][target_padded[b] > 0]
                        if len(valid_items) > 0:
                            target_multihot[b, valid_items] = 1.0
                
                # Forward pass to get sequence output
                all_outputs = self.model(sequences, basket_masks, sequence_masks)
                
                # Get the last valid basket embedding
                last_positions = sequence_masks.sum(dim=1) - 1
                last_positions = last_positions.unsqueeze(1)  # [batch_size, 1]
                
                # Gather last position outputs
                batch_size = sequences.size(0)
                hidden_size = all_outputs.size(-1)
                
                last_outputs = torch.gather(
                    all_outputs,
                    dim=1,
                    index=last_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
                ).squeeze(1)  # [batch_size, hidden_size]
                
                # Predict next basket
                next_basket_logits = self.model.output_layer(last_outputs)  # [batch_size, vocab_size]
                
                # Compute loss
                bce_loss = F.binary_cross_entropy_with_logits(
                    next_basket_logits, target_multihot, reduction='mean'
                )
                
                total_loss += bce_loss.item() * batch_size
                total_items += batch_size
                
                # Compute metrics for each example
                for b in range(batch_size):
                    # Get ground truth items
                    true_items = target_multihot[b].nonzero().squeeze(-1)
                    
                    if len(true_items) > 0:
                        # Get relevance scores for all items
                        relevance = torch.zeros(
                            self.model.config.vocab_size, device=self.device
                        )
                        relevance[true_items] = 1.0
                        
                        # Get predictions (scores)
                        pred_scores = next_basket_logits[b]
                        
                        # Compute metrics for different k values
                        for k in [5, 10, 20]:
                            # Get top-k predictions
                            _, top_indices = torch.topk(pred_scores, k=k)
                            
                            # Calculate recall@k
                            top_relevance = relevance[top_indices]
                            recall = top_relevance.sum() / true_items.size(0)
                            recall_sums[k] += recall.item()
                            
                            # Calculate NDCG@k
                            dcg = torch.sum(top_relevance / torch.log2(torch.arange(k, device=self.device) + 2))
                            # Ideal DCG: all true items at the top
                            idcg = torch.sum(torch.ones(min(k, true_items.size(0)), device=self.device) / 
                                            torch.log2(torch.arange(min(k, true_items.size(0)), device=self.device) + 2))
                            
                            ndcg = dcg / idcg if idcg > 0 else 0
                            ndcg_sums[k] += ndcg.item()
        
        # Calculate average metrics
        avg_loss = total_loss / total_items
        avg_recall = {k: recall_sums[k] / total_items for k in recall_sums}
        avg_ndcg = {k: ndcg_sums[k] / total_items for k in ndcg_sums}
        
        metrics = {
            'loss': avg_loss,
            **{f'recall@{k}': avg_recall[k] for k in avg_recall},
            **{f'ndcg@{k}': avg_ndcg[k] for k in avg_ndcg}
        }
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics


def load_data_for_training(data_dir):
    """
    Load preprocessed data for training.
    
    Args:
        data_dir: Directory containing preprocessed data files.
    
    Returns:
        config: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    print(f"Loading data from {data_dir}...")
    
    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load vocabulary
    try:
        with open(os.path.join(data_dir, 'vocabulary.json'), 'r') as f:
            vocab = json.load(f)
    except:
        # If vocabulary.json doesn't exist, use vocab info from metadata
        vocab = {
            'vocab_size': metadata['vocab_size'],
            'pad_token': metadata['pad_token'],
            'mask_token': metadata['mask_token']
        }
    
    # Create model config
    config = BasketBERT4RecConfig(
        vocab_size=vocab['vocab_size'],
        max_position_embeddings=metadata['max_seq_length'],
        max_basket_size=metadata['max_basket_size'],
        pad_token_id=vocab['pad_token'],
        mask_token_id=vocab['mask_token'],
        # Model architecture hyperparameters
        hidden_size=128,  # Reduced from 256 to handle smaller datasets/machines
        num_hidden_layers=2,  # Reduced from 4 for faster training
        num_attention_heads=4,
        intermediate_size=512,  # Reduced from 1024
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        basket_embedding_type='mean'  # Options: 'mean', 'max', 'attention'
    )
    
    # Load training data
    train_sequences = np.load(os.path.join(data_dir, 'train/sequences.npy'))
    train_sequence_masks = np.load(os.path.join(data_dir, 'train/sequence_masks.npy'))
    
    # Try to load basket masks, if available
    try:
        train_basket_masks = np.load(os.path.join(data_dir, 'train/basket_masks.npy'))
    except:
        # If basket masks aren't available, create them (1 for items, 0 for padding)
        train_basket_masks = np.ones_like(train_sequences)
        train_basket_masks[train_sequences == config.pad_token_id] = 0
    
    train_targets_padded = np.load(os.path.join(data_dir, 'train/targets_padded.npy'))
    
    # Try to load multihot targets or create them on the fly during training
    try:
        train_targets_multihot = np.load(os.path.join(data_dir, 'train/targets_multihot.npy'))
    except:
        # We'll handle this in the dataset
        train_targets_multihot = None
    
    # Load masked positions and labels
    try:
        with open(os.path.join(data_dir, 'train/masked_positions.pkl'), 'rb') as f:
            train_masked_positions = pickle.load(f)
        
        with open(os.path.join(data_dir, 'train/masked_labels.pkl'), 'rb') as f:
            train_masked_labels = pickle.load(f)
    except:
        # If masked data isn't available, we'll create it dynamically
        print("Masked positions/labels not found. Will create masks dynamically during training.")
        train_masked_positions = None
        train_masked_labels = None
    
    # Create training dataset
    train_dataset = BasketBERT4RecDataset(
        sequences=train_sequences,
        sequence_masks=train_sequence_masks,
        basket_masks=train_basket_masks,
        masked_positions=train_masked_positions,
        masked_labels=train_masked_labels,
        targets_padded=train_targets_padded,
        targets_multihot=train_targets_multihot,
        mode='mlm'  # Use MLM for pre-training
    )
    
    # Load validation data if available
    if os.path.exists(os.path.join(data_dir, 'val')):
        val_sequences = np.load(os.path.join(data_dir, 'val/sequences.npy'))
        val_sequence_masks = np.load(os.path.join(data_dir, 'val/sequence_masks.npy'))
        
        # Try to load basket masks, if available
        try:
            val_basket_masks = np.load(os.path.join(data_dir, 'val/basket_masks.npy'))
        except:
            # If basket masks aren't available, create them
            val_basket_masks = np.ones_like(val_sequences)
            val_basket_masks[val_sequences == config.pad_token_id] = 0
        
        val_targets_padded = np.load(os.path.join(data_dir, 'val/targets_padded.npy'))
        
        # Try to load multihot targets
        try:
            val_targets_multihot = np.load(os.path.join(data_dir, 'val/targets_multihot.npy'))
        except:
            val_targets_multihot = None
        
        # Load masked positions and labels
        try:
            with open(os.path.join(data_dir, 'val/masked_positions.pkl'), 'rb') as f:
                val_masked_positions = pickle.load(f)
            
            with open(os.path.join(data_dir, 'val/masked_labels.pkl'), 'rb') as f:
                val_masked_labels = pickle.load(f)
        except:
            val_masked_positions = None
            val_masked_labels = None
        
        # Create validation dataset
        val_dataset = BasketBERT4RecDataset(
            sequences=val_sequences,
            sequence_masks=val_sequence_masks,
            basket_masks=val_basket_masks,
            masked_positions=val_masked_positions,
            masked_labels=val_masked_labels,
            targets_padded=val_targets_padded,
            targets_multihot=val_targets_multihot,
            mode='mlm'  # Use MLM for validation
        )
    else:
        val_dataset = None
    
    print(f"Data loaded: {len(train_dataset)} training examples, ", end="")
    print(f"{len(val_dataset) if val_dataset else 0} validation examples")
    
    return config, train_dataset, val_dataset


def train_model(data_dir='./bert4rec_data/', output_dir='./bert4rec_model/',
                batch_size=16, num_epochs=10, learning_rate=0.001,
                use_gpu=True):
    """
    Train a BasketBERT4Rec model on the preprocessed data.
    
    Args:
        data_dir: Directory containing preprocessed data files.
        output_dir: Directory to save model outputs.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimization.
        use_gpu: Whether to use GPU for training if available.
    
    Returns:
        model: Trained BasketBERT4Rec model
        trainer: Model trainer with training history
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load data for training
    config, train_dataset, val_dataset = load_data_for_training(data_dir)
    
    # Create model
    model = BasketBERT4Rec(config)
    
    # Create trainer
    trainer = BasketBERT4RecTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=0.01,
        num_epochs=num_epochs,
        device=device,
        output_dir=output_dir,
        mlm_weight=1.0,  # Weight for masked language modeling loss
        nbr_weight=0.5   # Weight for next basket recommendation loss
    )
    
    # Train model
    training_metrics = trainer.train()
    
    # Save item embeddings
    item_embeddings = model.get_item_embeddings().cpu().numpy()
    np.save(os.path.join(output_dir, 'item_embeddings.npy'), item_embeddings)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__
    }, os.path.join(output_dir, 'final_model.pt'))
    
    print(f"Training completed! Model saved to {output_dir}")
    
    # Return model and trainer for further analysis
    return model, trainer


if __name__ == "__main__":
    # Command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BasketBERT4Rec model')
    parser.add_argument('--data_dir', type=str, default='./bert4rec_data/', 
                        help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./bert4rec_model/', 
                        help='Directory to save model outputs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--cpu', action='store_true', 
                        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Train model
    model, trainer = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_gpu=not args.cpu
    )