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
from sklearn.model_selection import train_test_split

# Check for CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name (GPU 0): {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Define paths
data_dir = './bert4rec_data/'  # Directory containing preprocessed data from 2.py
model_output_dir = './bert4rec_model/'  # Directory for model outputs
os.makedirs(model_output_dir, exist_ok=True)


class BasketBERT4RecConfig:
    """Configuration class for BasketBERT4Rec model."""
    
    def __init__(
        self,
        vocab_size=50000,
        hidden_size=64,              
        num_hidden_layers=1,         
        num_attention_heads=2,       
        intermediate_size=256,       
        hidden_dropout_prob=0.3,     
        attention_probs_dropout_prob=0.3,  # Increased from 0.1
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
            prediction_scores or sequence_output: Depending on whether masked_positions is provided
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
            num_masked = masked_positions.size(1)
            gathered_output = torch.gather(
                sequence_output,
                dim=1,
                index=masked_positions.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            )
            
            # Get prediction scores
            prediction_scores = self.output_layer(gathered_output)
            return prediction_scores
        else:
            # Return the full sequence output for further processing
            return sequence_output
    
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
            
            # Ensure masked_labels are correctly sized even if empty
            if masked_positions.size(0) > 0 and masked_labels.size(0) == 0:  # if positions exist but labels are empty
                masked_labels = torch.zeros((masked_positions.size(0), self.basket_size), dtype=torch.long)

            return {
                'sequences': self.sequences[idx],
                'sequence_masks': self.sequence_masks[idx],
                'basket_masks': self.basket_masks[idx],
                'masked_positions': masked_positions,
                'masked_labels': masked_labels,
                'num_masks': masked_positions.size(0) if masked_positions is not None else 0
            }
        
        elif self.mode == 'next_basket':
            item_target_padded = None
            if self.targets_padded is not None:
                item_target_padded = self.targets_padded[idx]
            # else: item_target_padded remains None, or handle error if it's critical

            item_target_multihot = None
            if self.targets_multihot is not None:  # Check if the dataset attribute is not None
                item_target_multihot = self.targets_multihot[idx]  # Then access it
            
            return {
                'sequences': self.sequences[idx],
                'sequence_masks': self.sequence_masks[idx],
                'basket_masks': self.basket_masks[idx],
                'target_padded': item_target_padded,  # Use the potentially None value
                'target_multihot': item_target_multihot  # This can be None
            }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length masked positions and potential None values.
    """
    if not batch: # Handle empty batch
        return {}

    # Determine max number of masked positions in this batch
    # Ensure 'num_masks' key exists and items are not None before trying to access it.
    max_masks = 0
    # Check if the first item exists and has 'num_masks'
    if batch[0] and 'num_masks' in batch[0] and batch[0]['num_masks'] is not None:
        valid_num_masks = [item.get('num_masks', 0) for item in batch if item is not None and item.get('num_masks') is not None]
        if valid_num_masks:
            max_masks = max(valid_num_masks)
        # If all num_masks are None or items are None, max_masks remains 0

    batch_dict = {}
    
    # Assuming all items in the batch should have the same keys as the first valid item
    first_valid_item = next((item for item in batch if item is not None), None)
    if not first_valid_item: # All items in batch are None
        return {}

    for key in first_valid_item.keys():
        if key == 'num_masks': # Already processed or not needed in final batch
            continue

        all_items_for_key = [item.get(key) for item in batch if item is not None]

        if not all_items_for_key: # All items were None or key was missing for all
            batch_dict[key] = None
            continue

        if all(x is None for x in all_items_for_key):
            batch_dict[key] = None
            continue
        
        # If there's a mix of None and non-None, and it's not a special padding key
        if any(x is None for x in all_items_for_key) and key not in ['masked_positions', 'masked_labels']:
            print(f"Warning: Mixed None and Tensor values for key '{key}' in batch. Setting to None for this batch.")
            batch_dict[key] = None
            continue

        try:
            if key == 'masked_positions':
                padded_positions = []
                for item_val in all_items_for_key:
                    pos = item_val if item_val is not None else torch.zeros(0, dtype=torch.long)
                    current_len = pos.size(0) if pos.dim() > 0 else 0
                    if current_len < max_masks:
                        padding = torch.zeros(max_masks - current_len, dtype=pos.dtype, device=pos.device if pos.numel() > 0 else 'cpu')
                        pos = torch.cat([pos, padding]) if current_len > 0 else padding
                    elif current_len > max_masks:
                        pos = pos[:max_masks]
                    padded_positions.append(pos)
                batch_dict[key] = torch.stack(padded_positions) if padded_positions else torch.empty(0, dtype=torch.long)

            elif key == 'masked_labels':
                padded_labels = []
                basket_s = 1 
                first_valid_label = next((val for val in all_items_for_key if val is not None and val.ndim > 1 and val.shape[0] > 0), None)
                if first_valid_label is not None:
                    basket_s = first_valid_label.size(1)
                else:
                    first_valid_seq = first_valid_item.get('sequences')
                    if first_valid_seq is not None and first_valid_seq.ndim == 3:
                        basket_s = first_valid_seq.size(2)
                if basket_s == 0 : basket_s = 1 # Ensure basket_s is at least 1

                for item_val in all_items_for_key:
                    labels = item_val if item_val is not None else torch.zeros((0, basket_s), dtype=torch.long)
                    current_len = labels.size(0) if labels.dim() > 0 else 0
                    
                    actual_basket_s = labels.size(1) if labels.ndim > 1 and labels.size(1) > 0 else basket_s
                    if actual_basket_s == 0: actual_basket_s = basket_s # Fallback

                    if labels.ndim == 1 and current_len > 0 and actual_basket_s > 0 :
                         labels = labels.unsqueeze(1).expand(-1, actual_basket_s)
                    
                    if current_len < max_masks:
                        padding_shape = (max_masks - current_len, actual_basket_s)
                        if padding_shape[0] >= 0 and padding_shape[1] > 0: # Ensure padding dimensions are valid
                            padding = torch.zeros(padding_shape, dtype=labels.dtype, device=labels.device if labels.numel() > 0 else 'cpu')
                            labels = torch.cat([labels, padding], dim=0) if current_len > 0 else padding
                        elif current_len == 0 and max_masks > 0 and padding_shape[1] > 0:
                            labels = torch.zeros((max_masks, padding_shape[1]), dtype=torch.long, device=labels.device if labels.numel() > 0 else 'cpu')
                    elif current_len > max_masks:
                        labels = labels[:max_masks]
                    
                    # Ensure all labels have consistent second dimension before appending
                    if labels.ndim == 2 and labels.size(1) != basket_s and padded_labels and padded_labels[0].ndim == 2:
                         # Attempt to conform or warn; for now, let's assume it should be basket_s
                         if labels.size(1) == 0 and basket_s > 0: # if it became [N,0]
                             labels = torch.zeros((labels.size(0), basket_s), dtype=labels.dtype, device=labels.device)

                    padded_labels.append(labels)
                
                if padded_labels:
                    # Check for consistent shapes before stacking
                    ref_shape = None
                    all_same_shape = True
                    for lbl in padded_labels:
                        if lbl.numel() == 0 and max_masks > 0 and basket_s > 0: # Handle completely empty labels that should have shape
                            lbl_to_check = torch.zeros((max_masks, basket_s), dtype=lbl.dtype, device=lbl.device)
                        else:
                            lbl_to_check = lbl

                        if ref_shape is None and lbl_to_check.numel() > 0:
                            ref_shape = lbl_to_check.shape
                        if lbl_to_check.numel() > 0 and lbl_to_check.shape != ref_shape:
                            all_same_shape = False
                            break
                    
                    if all_same_shape or not any(lbl.numel() > 0 for lbl in padded_labels): # Stack if all same or all empty
                        # Ensure even empty labels are shaped correctly if max_masks > 0
                        processed_padded_labels = []
                        for lbl in padded_labels:
                            if lbl.numel() == 0 and max_masks > 0 and basket_s > 0:
                                processed_padded_labels.append(torch.zeros((max_masks, basket_s), dtype=torch.long, device=lbl.device if hasattr(lbl, 'device') else 'cpu'))
                            elif lbl.ndim == 2 and lbl.shape[0] == max_masks and lbl.shape[1] == basket_s:
                                 processed_padded_labels.append(lbl)
                            elif lbl.ndim == 2 and lbl.shape[0] == max_masks and basket_s > 0 : # if basket size was inferred differently but length is ok
                                 # This case is tricky, might need to pad/truncate the basket dim or error
                                 # For now, if it's going to fail stack, let it, or pad to common basket_s
                                 if lbl.size(1) < basket_s:
                                     padding_basket = torch.zeros((max_masks, basket_s - lbl.size(1)), dtype=lbl.dtype, device=lbl.device)
                                     processed_padded_labels.append(torch.cat((lbl, padding_basket), dim=1))
                                 elif lbl.size(1) > basket_s:
                                     processed_padded_labels.append(lbl[:, :basket_s])
                                 else:
                                     processed_padded_labels.append(lbl) # Should be fine
                            else: # Fallback for unexpected shapes
                                 processed_padded_labels.append(torch.zeros((max_masks, basket_s), dtype=torch.long, device=lbl.device if hasattr(lbl, 'device') else 'cpu'))


                        if all(isinstance(l, torch.Tensor) for l in processed_padded_labels):
                            try:
                                batch_dict[key] = torch.stack(processed_padded_labels)
                            except RuntimeError as e:
                                print(f"Error stacking '{key}': {e}. Shapes: {[l.shape for l in processed_padded_labels]}. Setting to None.")
                                batch_dict[key] = None
                        else:
                            print(f"Not all items for '{key}' are tensors after processing. Setting to None.")
                            batch_dict[key] = None

                    else: # Shapes are different and not all empty
                        print(f"Warning: Could not stack '{key}' due to shape mismatch. Shapes: {[lbl.shape for lbl in padded_labels]}. Setting to None.")
                        batch_dict[key] = None
                else: # padded_labels is empty
                    batch_dict[key] = torch.empty((0, max_masks, basket_s) if max_masks > 0 and basket_s > 0 else (0), dtype=torch.long)

            else: # Standard handling for other keys
                # Ensure all items are tensors before stacking
                if all(isinstance(x, torch.Tensor) for x in all_items_for_key):
                    batch_dict[key] = torch.stack(all_items_for_key)
                else:
                    # This is where the original error likely happened for 'target_multihot' etc.
                    print(f"Error: Not all items for key '{key}' are Tensors. Found types: {[type(x) for x in all_items_for_key]}. Setting to None.")
                    batch_dict[key] = None
        
        except TypeError as te:
            print(f"TypeError during processing for key '{key}': {te}. Values: {[type(v) for v in all_items_for_key]}. Setting to None.")
            batch_dict[key] = None
        except RuntimeError as re:
            print(f"RuntimeError during processing for key '{key}': {re}. Values: {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in all_items_for_key]}. Setting to None.")
            batch_dict[key] = None

    return batch_dict


def load_data_for_training(data_dir):
    """
    Load preprocessed data for training.
    
    Args:
        data_dir: Directory containing preprocessed data files.
    
    Returns:
        config: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
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
        mask_token_id=vocab['mask_token']
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
    
    # Load test data if available (for final evaluation)
    test_dataset = None
    if os.path.exists(os.path.join(data_dir, 'test')):
        try:
            test_sequences = np.load(os.path.join(data_dir, 'test/sequences.npy'))
            test_sequence_masks = np.load(os.path.join(data_dir, 'test/sequence_masks.npy'))
            
            # Try to load basket masks, if available
            try:
                test_basket_masks = np.load(os.path.join(data_dir, 'test/basket_masks.npy'))
            except:
                # If basket masks aren't available, create them
                test_basket_masks = np.ones_like(test_sequences)
                test_basket_masks[test_sequences == config.pad_token_id] = 0
            
            test_targets_padded = np.load(os.path.join(data_dir, 'test/targets_padded.npy'))
            
            # Try to load multihot targets
            try:
                test_targets_multihot = np.load(os.path.join(data_dir, 'test/targets_multihot.npy'))
            except:
                test_targets_multihot = None
            
            # Create test dataset
            test_dataset = BasketBERT4RecDataset(
                sequences=test_sequences,
                sequence_masks=test_sequence_masks,
                basket_masks=test_basket_masks,
                targets_padded=test_targets_padded,
                targets_multihot=test_targets_multihot,
                mode='next_basket'  # Use next basket prediction for testing
            )
        except Exception as e:
            print(f"Error loading test data: {e}")
            test_dataset = None
    
    print(f"Data loaded: {len(train_dataset)} training examples, ", end="")
    print(f"{len(val_dataset) if val_dataset else 0} validation examples, ", end="")
    print(f"{len(test_dataset) if test_dataset else 0} test examples")
    
    return config, train_dataset, val_dataset, test_dataset


def evaluate(predictions_sigmoid, targets, k_list=[5, 10, 15, 20]):
    """
    Evaluate next basket prediction performance using Precision@k, Recall@k, and Hit@k.
    
    Args:
        predictions_sigmoid: List of tensors with sigmoid scores for items
        targets: List of lists containing true item IDs
        k_list: List of k values for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    all_preds = torch.cat(predictions_sigmoid, dim=0)  # (num_samples, vocab_size)
    num_samples = all_preds.size(0)
 
    metrics = {f'precision@{k}': 0.0 for k in k_list}
    metrics.update({f'recall@{k}': 0.0 for k in k_list})
    metrics.update({f'hit@{k}': 0.0 for k in k_list})
 
    # Convert targets to set for faster intersection operations
    targets_set = [set(t) for t in targets]
 
    # Get top-k predictions for all k in k_list
    max_k = max(k_list)
    topk_preds = torch.topk(all_preds, k=max_k, dim=1).indices.cpu().numpy()  # (num_samples, max_k)
 
    for i in range(num_samples):
        true_items = targets_set[i]
        if not true_items:
            continue  # skip if no ground truth
 
        for k in k_list:
            pred_k = set(topk_preds[i, :k].tolist())
            hits = true_items.intersection(pred_k)
            hit_count = len(hits)
 
            metrics[f'precision@{k}'] += hit_count / k
            metrics[f'recall@{k}'] += hit_count / len(true_items)
            metrics[f'hit@{k}'] += 1.0 if hit_count > 0 else 0.0
 
    for k in k_list:
        metrics[f'precision@{k}'] /= num_samples
        metrics[f'recall@{k}'] /= num_samples
        metrics[f'hit@{k}'] /= num_samples
 
    return metrics


def evaluate_model(model, test_dataset, device):
    """
    Evaluate the model on test dataset.
    
    Args:
        model: BasketBERT4Rec model
        test_dataset: Test dataset
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32, # Consider using a larger batch size for evaluation if memory allows
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0 # Keep 0 for Windows, can be >0 on Linux/macOS
    )
    
    # Store predictions and targets
    all_predictions_sigmoid = [] # Renamed to avoid conflict with 'predictions' variable later
    all_targets = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"): # Renamed batch to batch_data
            # Move batch to device, checking for None values
            batch = {}
            if batch_data is None: # Should not happen if collate_fn returns {} for empty batch
                print("Warning: Received None batch from DataLoader. Skipping.")
                continue
            for k, v in batch_data.items(): # Use batch_data here
                if v is not None:
                    batch[k] = v.to(device)
                else:
                    batch[k] = None # Keep it as None if it was None from collate_fn
            
            # Get inputs, checking if they exist in the batch and are not None
            sequences = batch.get('sequences')
            sequence_masks = batch.get('sequence_masks')
            basket_masks = batch.get('basket_masks')
            targets_padded_batch = batch.get('target_padded') # Renamed to avoid conflict

            if sequences is None or sequence_masks is None or targets_padded_batch is None:
                print(f"Warning: Missing critical data in batch (sequences, sequence_masks, or target_padded are None). Skipping batch.")
                continue
            
            # Forward pass to get all outputs
            # Ensure basket_masks is passed if model expects it, otherwise handle its absence
            all_outputs = model(sequences, basket_masks=basket_masks, sequence_mask=sequence_masks)
            
            # Get the last valid basket embedding
            # Ensure sequence_masks is not all zeros to prevent negative indices
            valid_lengths = sequence_masks.sum(dim=1)
            if (valid_lengths == 0).any():
                print("Warning: Batch contains sequences with zero length. Skipping these.")
                # Filter out zero-length sequences if necessary, or handle carefully
                # For now, this might lead to issues if not handled before gather
            
            last_positions = valid_lengths - 1
            # Clamp last_positions to be non-negative
            last_positions = torch.clamp(last_positions, min=0).unsqueeze(1)  # [batch_size, 1]
            
            # Gather last position outputs
            batch_size_current = sequences.size(0) # Use current batch_size
            hidden_size = all_outputs.size(-1)
            
            # Ensure last_positions are within bounds of all_outputs.shape[1]
            if last_positions.max() >= all_outputs.shape[1]:
                print(f"Warning: last_positions out of bounds. Max pos: {last_positions.max()}, Seq len: {all_outputs.shape[1]}. Clamping.")
                last_positions = torch.clamp(last_positions, max=all_outputs.shape[1]-1)


            last_outputs = torch.gather(
                all_outputs,
                dim=1,
                index=last_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            ).squeeze(1)  # [batch_size, hidden_size]
            
            # Predict next basket
            next_basket_logits = model.output_layer(last_outputs)  # [batch_size, vocab_size]
            predictions_sigmoid_batch = torch.sigmoid(next_basket_logits) # Renamed
            
            all_predictions_sigmoid.append(predictions_sigmoid_batch)
            
            # Convert targets to list of non-padding items
            batch_targets_list = [] # Renamed
            for i in range(batch_size_current):
                # Ensure targets_padded_batch[i] is a tensor and not None
                current_target_padded = targets_padded_batch[i]
                if current_target_padded is not None:
                    # Filter out padding tokens (usually 0) and special tokens if any
                    # Assuming pad_token_id is 0. Adjust if different.
                    valid_items = current_target_padded[current_target_padded > 0].cpu().tolist()
                    batch_targets_list.append(valid_items)
                else:
                    batch_targets_list.append([]) # Append empty list if target is None for this item
            
            all_targets.extend(batch_targets_list)
    
    # Evaluate using our metrics
    # Ensure all_predictions_sigmoid is not empty before cat and evaluation
    if not all_predictions_sigmoid:
        print("Warning: No predictions were made during evaluation. Returning empty metrics.")
        return {f'precision@{k}': 0.0 for k in [5, 10, 15, 20]} # Return default empty metrics
        
    metrics = evaluate(all_predictions_sigmoid, all_targets, k_list=[5, 10, 15, 20])
    
    return metrics


def train_model(data_dir='./bert4rec_data/', output_dir='./bert4rec_model/',
                batch_size=32, num_epochs=5, learning_rate=0.001,
                use_gpu=True):
    """
    Train the model using the prepared data.
    
    Args:
        data_dir: Path to directory with prepared data
        output_dir: Path to save model outputs
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        use_gpu: Whether to use GPU if available
        
    Returns:
        Trained model and trainer object
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Use the first available CUDA device if available
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:0')
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    
    # Load data for training
    config, train_dataset, val_dataset, test_dataset = load_data_for_training(data_dir)
    
    # Create model
    model = BasketBERT4Rec(config)
    
    # Create trainer
    trainer = BasketBERT4RecTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=0.05,        # Increased from 0.01
        num_epochs=num_epochs,
        device=device,
        output_dir=output_dir,
        mlm_weight=1.5,           # Increased from 1.0
        nbr_weight=0.5            # Decreased from 1.0
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
    
    # Create a test set from validation set if needed
    if test_dataset is None and val_dataset is not None:
        print("Creating test set from validation set...")
        val_indices = range(len(val_dataset))
        # Ensure test_size is valid if val_dataset is small
        test_split_size = min(0.5, (len(val_dataset) - 1) / len(val_dataset)) if len(val_dataset) > 1 else 0
        if test_split_size > 0:
            _, test_indices = train_test_split(val_indices, test_size=test_split_size, random_state=42)

            # Create test dataset using attributes from val_dataset
            test_dataset = BasketBERT4RecDataset(
                sequences=val_dataset.sequences[test_indices],
                sequence_masks=val_dataset.sequence_masks[test_indices],
                basket_masks=val_dataset.basket_masks[test_indices],
                targets_padded=val_dataset.targets_padded[test_indices],
                # Handle potential numpy array or None for multihot
                targets_multihot=(val_dataset.targets_multihot[test_indices]
                                if isinstance(val_dataset.targets_multihot, np.ndarray)
                                else None),
                mode='next_basket'  # Set mode for evaluation
            )
            print(f"Test set created with {len(test_dataset)} examples")
        else:
            print("Validation set too small to create a test set.")
            test_dataset = None  # Ensure test_dataset remains None
    
    # Evaluate model on test set
    if test_dataset:
        print("\nEvaluating model on test set...")
        test_metrics = evaluate_model(model, test_dataset, device)
        
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Return model and trainer for further analysis
    return model, trainer, test_metrics


class BasketBERT4RecTrainer:
    """Trainer for BasketBERT4Rec model."""
    
    def __init__(self, model, train_dataset, val_dataset=None, 
                 batch_size=32, lr=0.001, weight_decay=0.01,
                 num_epochs=5, device='cuda', output_dir='./model_output/',
                 mlm_weight=1.5, nbr_weight=0.5):
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
            num_workers=0, # Set to 0 for Windows, can be >0 on Linux/macOS
            pin_memory=True if device == 'cuda' else False
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=0, # Set to 0 for Windows
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
            patience=2, # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=1e-6
        )
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize training state
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        self.early_stop_patience = 5 # Number of epochs with no improvement on validation loss to trigger early stopping
    
    def train(self):
        """Train the model for the specified number of epochs."""
        print(f"Starting training for {self.num_epochs} epochs...")
        all_train_metrics = []
        all_val_metrics = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_epoch_metrics = self._train_epoch(epoch)
            all_train_metrics.append({'loss': train_loss, **train_epoch_metrics})
            
            # Validation phase
            val_loss_for_epoch = None
            if self.val_loader:
                val_loss, val_epoch_metrics = self._validate_epoch(epoch)
                all_val_metrics.append({'loss': val_loss, **val_epoch_metrics})
                val_loss_for_epoch = val_loss
                
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
                        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {self.early_stop_patience} epochs.")
                        break
            
            # Always save the latest model (can be configured)
            self._save_model(f'model_epoch_{epoch+1}.pt')
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}", end="")
            if val_loss_for_epoch is not None:
                print(f", Val Loss: {val_loss_for_epoch:.4f}", end="")
            print(f", LR: {self.optimizer.param_groups[0]['lr']:.1e}")
            # Optionally print more metrics from train_epoch_metrics and val_epoch_metrics
            if 'mlm_accuracy' in train_epoch_metrics:
                 print(f"  Train MLM Acc: {train_epoch_metrics['mlm_accuracy']:.4f}", end="")
            if 'recall' in train_epoch_metrics: # Assuming recall is recall@10
                 print(f", Train NBR Rec@10: {train_epoch_metrics['recall']:.4f}", end="")
            if self.val_loader and 'mlm_accuracy' in val_epoch_metrics:
                 print(f" | Val MLM Acc: {val_epoch_metrics['mlm_accuracy']:.4f}", end="")
            if self.val_loader and 'recall' in val_epoch_metrics:
                 print(f", Val NBR Rec@10: {val_epoch_metrics['recall']:.4f}", end="")
            print("")


        print("Training completed!")
        
        # Return metrics for evaluation (e.g., metrics of the last epoch or best epoch)
        # For simplicity, returning metrics of the last completed epoch
        final_train_metrics = all_train_metrics[-1] if all_train_metrics else {}
        final_val_metrics = all_val_metrics[-1] if all_val_metrics else {}

        return {
            'train_loss': final_train_metrics.get('loss'),
            'train_metrics': {k:v for k,v in final_train_metrics.items() if k != 'loss'},
            'val_loss': final_val_metrics.get('loss'),
            'val_metrics': {k:v for k,v in final_val_metrics.items() if k != 'loss'}
        }
    
    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss_sum = 0.0
        total_mlm_loss_sum = 0.0
        total_nbr_loss_sum = 0.0
        num_batches = len(self.train_loader)
        
        total_mlm_items_epoch = 0
        correct_mlm_items_epoch = 0
        
        total_nbr_items_epoch = 0
        weighted_recall_sum_epoch = 0.0 # For NBR recall (e.g., recall@10 * num_items_in_batch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move batch to device, handle None values from collate_fn
            batch = {k: v.to(self.device) if v is not None else None for k, v in batch_data.items()}
            
            self.optimizer.zero_grad()
            
            current_batch_loss, batch_metrics = self._compute_loss(batch)
            
            if current_batch_loss is not None and torch.isfinite(current_batch_loss):
                current_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss_sum += current_batch_loss.item()
            else:
                # Skip optimization if loss is None or not finite (e.g. due to empty batch or bad data)
                print(f"Warning: Skipping batch {batch_idx} in epoch {epoch+1} due to invalid loss: {current_batch_loss}")


            # Accumulate metrics from batch_metrics
            total_mlm_loss_sum += batch_metrics.get('mlm_loss', 0)
            total_nbr_loss_sum += batch_metrics.get('nbr_loss', 0)
            
            if 'mlm_correct' in batch_metrics and 'mlm_total' in batch_metrics and batch_metrics['mlm_total'] > 0:
                total_mlm_items_epoch += batch_metrics['mlm_total']
                correct_mlm_items_epoch += batch_metrics['mlm_correct']
            
            if 'recall' in batch_metrics and 'nbr_total' in batch_metrics and batch_metrics['nbr_total'] > 0:
                total_nbr_items_epoch += batch_metrics['nbr_total']
                weighted_recall_sum_epoch += batch_metrics['recall'] * batch_metrics['nbr_total']
            
            pbar_postfix = {
                'loss': f"{current_batch_loss.item() if current_batch_loss is not None and torch.isfinite(current_batch_loss) else float('nan'):.4f}",
                'mlm_l': f"{batch_metrics.get('mlm_loss', 0):.4f}",
                'nbr_l': f"{batch_metrics.get('nbr_loss', 0):.4f}"
            }
            if batch_metrics.get('mlm_total', 0) > 0:
                pbar_postfix['mlm_acc'] = f"{batch_metrics.get('mlm_accuracy', 0):.2f}"
            if batch_metrics.get('nbr_total', 0) > 0:
                 pbar_postfix['nbr_rec'] = f"{batch_metrics.get('recall', 0):.2f}" # recall is recall@10
            pbar.set_postfix(pbar_postfix)
            
        avg_epoch_loss = total_loss_sum / num_batches if num_batches > 0 else 0
        avg_mlm_loss_epoch = total_mlm_loss_sum / num_batches if num_batches > 0 else 0
        avg_nbr_loss_epoch = total_nbr_loss_sum / num_batches if num_batches > 0 else 0
        
        epoch_metrics_summary = {
            'avg_mlm_loss': avg_mlm_loss_epoch,
            'avg_nbr_loss': avg_nbr_loss_epoch,
            'mlm_accuracy': (correct_mlm_items_epoch / total_mlm_items_epoch) if total_mlm_items_epoch > 0 else 0,
            'recall': (weighted_recall_sum_epoch / total_nbr_items_epoch) if total_nbr_items_epoch > 0 else 0 # This is avg recall@10
        }
        
        return avg_epoch_loss, epoch_metrics_summary

    def _validate_epoch(self, epoch):
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss_sum = 0.0
        total_mlm_loss_sum = 0.0
        total_nbr_loss_sum = 0.0
        num_batches = len(self.val_loader)

        total_mlm_items_epoch = 0
        correct_mlm_items_epoch = 0
        
        total_nbr_items_epoch = 0
        weighted_recall_sum_epoch = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                batch = {k: v.to(self.device) if v is not None else None for k, v in batch_data.items()}
                
                current_batch_loss, batch_metrics = self._compute_loss(batch)

                if current_batch_loss is not None and torch.isfinite(current_batch_loss):
                    total_loss_sum += current_batch_loss.item()

                total_mlm_loss_sum += batch_metrics.get('mlm_loss', 0)
                total_nbr_loss_sum += batch_metrics.get('nbr_loss', 0)

                if 'mlm_correct' in batch_metrics and 'mlm_total' in batch_metrics and batch_metrics['mlm_total'] > 0:
                    total_mlm_items_epoch += batch_metrics['mlm_total']
                    correct_mlm_items_epoch += batch_metrics['mlm_correct']
                
                if 'recall' in batch_metrics and 'nbr_total' in batch_metrics and batch_metrics['nbr_total'] > 0:
                    total_nbr_items_epoch += batch_metrics['nbr_total']
                    weighted_recall_sum_epoch += batch_metrics['recall'] * batch_metrics['nbr_total']
                
                pbar_postfix = {
                    'loss': f"{current_batch_loss.item() if current_batch_loss is not None and torch.isfinite(current_batch_loss) else float('nan'):.4f}",
                    'mlm_l': f"{batch_metrics.get('mlm_loss', 0):.4f}",
                    'nbr_l': f"{batch_metrics.get('nbr_loss', 0):.4f}"
                }
                if batch_metrics.get('mlm_total', 0) > 0:
                    pbar_postfix['mlm_acc'] = f"{batch_metrics.get('mlm_accuracy', 0):.2f}"
                if batch_metrics.get('nbr_total', 0) > 0:
                    pbar_postfix['nbr_rec'] = f"{batch_metrics.get('recall', 0):.2f}"
                pbar.set_postfix(pbar_postfix)

        avg_epoch_loss = total_loss_sum / num_batches if num_batches > 0 else 0
        avg_mlm_loss_epoch = total_mlm_loss_sum / num_batches if num_batches > 0 else 0
        avg_nbr_loss_epoch = total_nbr_loss_sum / num_batches if num_batches > 0 else 0
        
        epoch_metrics_summary = {
            'avg_mlm_loss': avg_mlm_loss_epoch,
            'avg_nbr_loss': avg_nbr_loss_epoch,
            'mlm_accuracy': (correct_mlm_items_epoch / total_mlm_items_epoch) if total_mlm_items_epoch > 0 else 0,
            'recall': (weighted_recall_sum_epoch / total_nbr_items_epoch) if total_nbr_items_epoch > 0 else 0
        }
        
        return avg_epoch_loss, epoch_metrics_summary
    
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
        k_values = [5, 10, 15, 20]
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


if __name__ == "__main__":
    # Train and evaluate the model
    model, trainer, test_metrics = train_model(
        data_dir='./bert4rec_data/',  # Directory with prepared data
        output_dir='./bert4rec_model/',  # Directory to save model outputs
        batch_size=32,  # Reduced from 64
        num_epochs=10,   # Reduced from 10
        learning_rate=0.001
    )