import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt

def validate_bert4rec_data(data_dir='./bert4rec_data/'):
    """
    Validate the processed BERT4Rec data.
    """
    print("Validating BERT4Rec prepared data...")
    
    # 1. Check if required files and directories exist
    expected_files = [
        'metadata.json',
        'vocabulary.json',
        'train/sequences.npy',
        'train/targets_padded.npy',
        'val/sequences.npy',
        'val/targets_padded.npy'
    ]
    
    for file_path in expected_files:
        full_path = os.path.join(data_dir, file_path)
        if not os.path.exists(full_path):
            print(f"ERROR: Missing expected file {full_path}")
            return False
    
    # 2. Load and check metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # 3. Load and check vocabulary
    with open(os.path.join(data_dir, 'vocabulary.json'), 'r') as f:
        vocab = json.load(f)
    
    print(f"\nVocabulary size: {vocab['vocab_size']}")
    print(f"Padding token: {vocab['pad_token']}")
    print(f"Mask token: {vocab['mask_token']}")
    
    # 4. Check training data
    train_sequences = np.load(os.path.join(data_dir, 'train/sequences.npy'))
    train_targets = np.load(os.path.join(data_dir, 'train/targets_padded.npy'))
    
    print("\nTraining data:")
    print(f"  Number of sequences: {len(train_sequences)}")
    print(f"  Sequence shape: {train_sequences.shape}")
    print(f"  Target shape: {train_targets.shape}")
    
    # 5. Check validation data
    val_sequences = np.load(os.path.join(data_dir, 'val/sequences.npy'))
    val_targets = np.load(os.path.join(data_dir, 'val/targets_padded.npy'))
    
    print("\nValidation data:")
    print(f"  Number of sequences: {len(val_sequences)}")
    print(f"  Sequence shape: {val_sequences.shape}")
    print(f"  Target shape: {val_targets.shape}")
    
    # 6. Check BERT masked data
    with open(os.path.join(data_dir, 'train/masked_positions.pkl'), 'rb') as f:
        train_masked_positions = pickle.load(f)
    
    with open(os.path.join(data_dir, 'train/masked_labels.pkl'), 'rb') as f:
        train_masked_labels = pickle.load(f)
    
    print("\nBERT masking:")
    masked_count = sum(1 for pos in train_masked_positions if pos)
    print(f"  Sequences with masking: {masked_count} out of {len(train_masked_positions)}")
    
    # 7. Plot a sample sequence and its target
    if len(train_sequences) > 0:
        sample_idx = 0
        seq = train_sequences[sample_idx]
        target = train_targets[sample_idx]
        
        print("\nSample sequence (first row):")
        print(f"  Shape: {seq.shape}")
        print(f"  First basket: {seq[0][:10]}{'...' if len(seq[0]) > 10 else ''}")
        print(f"  Target basket: {target[:10]}{'...' if len(target) > 10 else ''}")
    
    print("\nValidation complete!")
    return True

if __name__ == "__main__":
    validate_bert4rec_data()