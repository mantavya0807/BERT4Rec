import pandas as pd
import numpy as np
import ast
import json
import os
import pickle
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BERT4RecDataPreparation:
    """
    Class for preparing data for BERT4Rec model adapted for next basket prediction.
    """
    
    def __init__(self, data_file, output_dir='./prepared_data/',
                max_seq_length=50, max_basket_size=None, mask_prob=0.15,
                train_val_split=0.2, random_seed=42):
        """
        Initialize the data preparation class.
        
        Args:
            data_file: Path to the dataset file
            output_dir: Directory to save the prepared data
            max_seq_length: Maximum number of baskets in a sequence
            max_basket_size: Maximum number of items in a basket (calculated if None)
            mask_prob: Probability of masking a basket for BERT pre-training
            train_val_split: Fraction of data to use for validation
            random_seed: Random seed for reproducibility
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.max_basket_size = max_basket_size
        self.mask_prob = mask_prob
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        
        # Special token IDs
        self.PAD_TOKEN = 0
        self.MASK_TOKEN = 1
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize vocabulary and data structures
        self.product_to_id = {}
        self.id_to_product = {}
        self.vocab_size = 0
        self.df = None
        self.all_products = set()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def load_data(self):
        """
        Load the dataset from file and perform basic preprocessing.
        """
        print(f"Loading data from {self.data_file}...")
        
        try:
            # Try reading the file, print column names to help diagnose issues
            self.df = pd.read_csv(self.data_file)
            print(f"Columns found in file: {self.df.columns.tolist()}")
            
            # Check if required columns exist
            required_columns = ['customer_sequence', 'last_basket']
            for col in required_columns:
                if col not in self.df.columns:
                    close_matches = [c for c in self.df.columns if col.lower() in c.lower()]
                    print(f"Warning: Column '{col}' not found. Closest matches: {close_matches}")
            
            # Try to handle potential column name issues
            col_mapping = {}
            
            # Map customer_sequence column
            if 'customer_sequence' in self.df.columns:
                col_mapping['customer_sequence'] = 'customer_sequence'
            else:
                # Try to find a similar column
                sequence_col_candidates = [c for c in self.df.columns if 'sequence' in c.lower() or 'history' in c.lower()]
                if sequence_col_candidates:
                    col_mapping['customer_sequence'] = sequence_col_candidates[0]
                    print(f"Using '{sequence_col_candidates[0]}' as customer_sequence column")
            
            # Map last_basket column
            if 'last_basket' in self.df.columns:
                col_mapping['last_basket'] = 'last_basket'
            else:
                # Try to find a similar column
                basket_col_candidates = [c for c in self.df.columns if 'basket' in c.lower() or 'target' in c.lower()]
                if basket_col_candidates:
                    col_mapping['last_basket'] = basket_col_candidates[0]
                    print(f"Using '{basket_col_candidates[0]}' as last_basket column")
            
            # Ensure we have the needed columns
            if 'customer_sequence' not in col_mapping or 'last_basket' not in col_mapping:
                # Print first few rows to help diagnose
                print("\nFirst few rows of the data:")
                print(self.df.head().to_string())
                raise ValueError("Could not find required columns in the data file. Please check the format.")
            
            # Apply the column mapping
            self.df = self.df.rename(columns={
                col_mapping['customer_sequence']: 'customer_sequence',
                col_mapping['last_basket']: 'last_basket'
            })
            
            # First convert any potential leading/trailing whitespace
            self.df['customer_sequence'] = self.df['customer_sequence'].str.strip() if self.df['customer_sequence'].dtype == 'object' else self.df['customer_sequence']
            self.df['last_basket'] = self.df['last_basket'].str.strip() if self.df['last_basket'].dtype == 'object' else self.df['last_basket']
            
            # If cells contain string representations of lists, parse them
            if self.df['customer_sequence'].dtype == 'object':
                print("Parsing customer_sequence column as string representations of lists...")
                # Try to handle potential formatting issues in the strings
                try:
                    self.df['customer_sequence'] = self.df['customer_sequence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing customer_sequence: {e}")
                    print("Example value:", self.df['customer_sequence'].iloc[0])
                    raise
            
            if self.df['last_basket'].dtype == 'object':
                print("Parsing last_basket column as string representations of lists...")
                try:
                    self.df['last_basket'] = self.df['last_basket'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing last_basket: {e}")
                    print("Example value:", self.df['last_basket'].iloc[0])
                    raise
            
            print(f"Loaded {len(self.df)} customer sequences")
            
            # Collect all unique product IDs
            for sequence in self.df['customer_sequence']:
                for basket in sequence:
                    if isinstance(basket, list):
                        self.all_products.update(basket)
            
            for basket in self.df['last_basket']:
                if isinstance(basket, list):
                    self.all_products.update(basket)
            
            print(f"Found {len(self.all_products)} unique products")
            
            # Determine max_basket_size if not provided
            if self.max_basket_size is None:
                try:
                    self.max_basket_size = max(
                        max(len(basket) if isinstance(basket, list) else 0 
                            for sequence in self.df['customer_sequence'] 
                            for basket in sequence if isinstance(sequence, list)),
                        max(len(basket) if isinstance(basket, list) else 0 
                            for basket in self.df['last_basket'])
                    )
                    print(f"Automatically determined max_basket_size: {self.max_basket_size}")
                except ValueError:
                    self.max_basket_size = 20
                    print(f"Could not determine max_basket_size automatically. Using default: {self.max_basket_size}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            
            # Try to provide more diagnostic information
            try:
                # Peek at the file to see its format
                with open(self.data_file, 'r') as f:
                    print("\nFirst few lines of the file:")
                    for i, line in enumerate(f):
                        if i < 5:  # Print first 5 lines
                            print(line.strip())
                        else:
                            break
            except:
                pass
            
            raise
        
        return self.df
    
    def create_vocabulary(self):
        """
        Create a vocabulary mapping product IDs to token IDs.
        """
        print("Creating vocabulary mapping...")
        
        # Create product ID to token ID mapping (starting from 2)
        # 0 = padding token, 1 = mask token
        self.product_to_id = {pid: idx + 2 for idx, pid in enumerate(sorted(self.all_products))}
        self.id_to_product = {idx: pid for pid, idx in self.product_to_id.items()}
        self.vocab_size = len(self.product_to_id) + 2
        
        print(f"Vocabulary size (including special tokens): {self.vocab_size}")
        
        # Save vocabulary mapping
        vocab_data = {
            'pad_token': self.PAD_TOKEN,
            'mask_token': self.MASK_TOKEN,
            'product_to_id': self.product_to_id,
            'id_to_product': {str(k): v for k, v in self.id_to_product.items()},
            'vocab_size': self.vocab_size
        }
        
        with open(os.path.join(self.output_dir, 'vocabulary.json'), 'w') as f:
            json.dump(vocab_data, f)
        
        return self.vocab_size
    
    def basket_to_padded_ids(self, basket):
        """
        Convert a basket of product IDs to padded token IDs.
        
        Args:
            basket: List of product IDs
            
        Returns:
            Padded list of token IDs
        """
        if not basket or not isinstance(basket, list):
            return [self.PAD_TOKEN] * self.max_basket_size
        
        # Convert product IDs to vocabulary IDs
        ids = [self.product_to_id[pid] for pid in basket if pid in self.product_to_id]
        
        # Truncate or pad to fixed size
        if len(ids) > self.max_basket_size:
            return ids[:self.max_basket_size]
        else:
            return ids + [self.PAD_TOKEN] * (self.max_basket_size - len(ids))
    
    def basket_to_multihot(self, basket):
        """
        Convert a basket to multi-hot encoding.
        
        Args:
            basket: List of product IDs
            
        Returns:
            Multi-hot encoding vector
        """
        encoding = np.zeros(self.vocab_size, dtype=np.float32)
        if isinstance(basket, list):
            for pid in basket:
                if pid in self.product_to_id:
                    encoding[self.product_to_id[pid]] = 1.0
        return encoding
    
    def prepare_sequences(self):
        """
        Prepare sequences for BERT4Rec model.
        
        Returns:
            Dictionary with sequences, masks, and targets
        """
        print("Preparing sequences...")
        
        # Prepare sequences and targets
        sequences = []
        sequence_masks = []
        basket_masks = []  # For masking padding within baskets
        targets_padded = []
        targets_multihot = []
        user_ids = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing sequences"):
            # Get baskets and target
            basket_sequence = row['customer_sequence']
            target_basket = row['last_basket']
            user_id = row.get('user_id', idx)  # Use index if user_id not available
            
            # Skip rows with invalid data
            if not isinstance(basket_sequence, list) or not isinstance(target_basket, list):
                print(f"Skipping row {idx}: Invalid data type")
                continue
            
            # Truncate sequence if needed
            if len(basket_sequence) > self.max_seq_length:
                basket_sequence = basket_sequence[-self.max_seq_length:]
            
            # Convert each basket to padded IDs
            padded_sequence = [self.basket_to_padded_ids(basket) for basket in basket_sequence]
            
            # Create sequence mask (1 for real baskets, 0 for padding)
            seq_mask = [1] * len(padded_sequence)
            
            # Create basket masks (1 for real items, 0 for padding)
            bskt_mask = []
            for basket in basket_sequence:
                if not isinstance(basket, list):
                    mask = [0] * self.max_basket_size
                elif len(basket) > self.max_basket_size:
                    mask = [1] * self.max_basket_size
                else:
                    mask = [1] * len(basket) + [0] * (self.max_basket_size - len(basket))
                bskt_mask.append(mask)
            
            # Pad sequence if shorter than max_seq_length
            padding_basket = [self.PAD_TOKEN] * self.max_basket_size
            padding_mask = [0] * self.max_basket_size
            while len(padded_sequence) < self.max_seq_length:
                padded_sequence.append(padding_basket)
                seq_mask.append(0)
                bskt_mask.append(padding_mask)
            
            # Prepare target representations
            target_padded = self.basket_to_padded_ids(target_basket)
            target_multihot = self.basket_to_multihot(target_basket)
            
            # Add to lists
            sequences.append(padded_sequence)
            sequence_masks.append(seq_mask)
            basket_masks.append(bskt_mask)
            targets_padded.append(target_padded)
            targets_multihot.append(target_multihot)
            user_ids.append(user_id)
        
        # Convert lists to numpy arrays
        sequences = np.array(sequences, dtype=np.int32)
        sequence_masks = np.array(sequence_masks, dtype=np.int32)
        basket_masks = np.array(basket_masks, dtype=np.int32)
        targets_padded = np.array(targets_padded, dtype=np.int32)
        targets_multihot = np.array(targets_multihot, dtype=np.float32)
        user_ids = np.array(user_ids)
        
        data = {
            'sequences': sequences,
            'sequence_masks': sequence_masks,
            'basket_masks': basket_masks,
            'targets_padded': targets_padded,
            'targets_multihot': targets_multihot,
            'user_ids': user_ids
        }
        
        return data
    
    def create_bert_training_data(self, data):
        """
        Create masked sequences for BERT pre-training.
        
        Args:
            data: Dictionary with original sequences and masks
            
        Returns:
            Dictionary with BERT training data
        """
        print("Creating BERT training data with masked sequences...")
        
        sequences = data['sequences']
        sequence_masks = data['sequence_masks']
        
        masked_seqs = []
        masked_positions = []
        masked_labels = []
        
        for seq, mask in tqdm(zip(sequences, sequence_masks), total=len(sequences), desc="Creating masked sequences"):
            # Create a copy of the sequence
            masked_seq = np.copy(seq)
            
            # Get positions of actual baskets (not padding)
            valid_positions = [i for i, m in enumerate(mask) if m == 1]
            
            # Skip sequences that are too short
            if len(valid_positions) <= 1:
                masked_seqs.append(masked_seq)
                masked_positions.append([])
                masked_labels.append([])
                continue
            
            # Randomly select baskets to mask
            num_to_mask = max(1, int(len(valid_positions) * self.mask_prob))
            mask_pos = sorted(random.sample(valid_positions, num_to_mask))
            
            # Save original values and apply masking
            labels = []
            for pos in mask_pos:
                labels.append(np.copy(seq[pos]))
                masked_seq[pos] = np.array([self.MASK_TOKEN] * self.max_basket_size)
            
            masked_seqs.append(masked_seq)
            masked_positions.append(mask_pos)
            masked_labels.append(labels)
        
        bert_data = {
            'masked_sequences': np.array(masked_seqs, dtype=np.int32),
            'masked_positions': masked_positions,
            'masked_labels': masked_labels
        }
        
        return bert_data
    
    def split_train_validation(self, data, bert_data):
        """
        Split data into training and validation sets.
        
        Args:
            data: Dictionary with original sequences and targets
            bert_data: Dictionary with BERT masked sequences
            
        Returns:
            Dictionary with train and validation splits
        """
        print("Splitting data into training and validation sets...")
        
        indices = np.arange(len(data['sequences']))
        train_indices, val_indices = train_test_split(
            indices, test_size=self.train_val_split, random_state=self.random_seed
        )
        
        # Create split data dictionary
        split_data = {
            'train': {
                'indices': train_indices,
                'sequences': data['sequences'][train_indices],
                'sequence_masks': data['sequence_masks'][train_indices],
                'basket_masks': data['basket_masks'][train_indices],
                'targets_padded': data['targets_padded'][train_indices],
                'targets_multihot': data['targets_multihot'][train_indices],
                'user_ids': data['user_ids'][train_indices],
                'masked_sequences': bert_data['masked_sequences'][train_indices],
                'masked_positions': [bert_data['masked_positions'][i] for i in train_indices],
                'masked_labels': [bert_data['masked_labels'][i] for i in train_indices]
            },
            'val': {
                'indices': val_indices,
                'sequences': data['sequences'][val_indices],
                'sequence_masks': data['sequence_masks'][val_indices],
                'basket_masks': data['basket_masks'][val_indices],
                'targets_padded': data['targets_padded'][val_indices],
                'targets_multihot': data['targets_multihot'][val_indices],
                'user_ids': data['user_ids'][val_indices],
                'masked_sequences': bert_data['masked_sequences'][val_indices],
                'masked_positions': [bert_data['masked_positions'][i] for i in val_indices],
                'masked_labels': [bert_data['masked_labels'][i] for i in val_indices]
            }
        }
        
        print(f"Training set size: {len(train_indices)}, Validation set size: {len(val_indices)}")
        
        return split_data
    
    def save_prepared_data(self, split_data):
        """
        Save the prepared data to disk.
        
        Args:
            split_data: Dictionary with train and validation data
        """
        print(f"Saving prepared data to {self.output_dir}...")
        
        # Save metadata
        metadata = {
            'max_seq_length': self.max_seq_length,
            'max_basket_size': self.max_basket_size,
            'pad_token': self.PAD_TOKEN,
            'mask_token': self.MASK_TOKEN,
            'vocab_size': self.vocab_size,
            'num_train': len(split_data['train']['indices']),
            'num_val': len(split_data['val']['indices'])
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Save train and validation data
        for split, data in split_data.items():
            split_dir = os.path.join(self.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            for key, value in data.items():
                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    np.save(os.path.join(split_dir, f"{key}.npy"), value)
                # Handle irregular data (like masked_positions, masked_labels)
                else:
                    with open(os.path.join(split_dir, f"{key}.pkl"), 'wb') as f:
                        pickle.dump(value, f)
        
        print("Data saved successfully")
    
    def run_pipeline(self):
        """
        Run the complete data preparation pipeline.
        
        Returns:
            Dictionary with metadata and split data
        """
        print("===== RUNNING BERT4REC DATA PREPARATION PIPELINE =====")
        
        # Step 1: Load the data
        self.load_data()
        
        # Step 2: Create vocabulary
        self.create_vocabulary()
        
        # Step 3: Prepare sequences
        data = self.prepare_sequences()
        
        # Step 4: Create BERT training data
        bert_data = self.create_bert_training_data(data)
        
        # Step 5: Split into train and validation sets
        split_data = self.split_train_validation(data, bert_data)
        
        # Step 6: Save prepared data
        self.save_prepared_data(split_data)
        
        # Return the prepared data for further use
        result = {
            'metadata': {
                'max_seq_length': self.max_seq_length,
                'max_basket_size': self.max_basket_size,
                'pad_token': self.PAD_TOKEN,
                'mask_token': self.MASK_TOKEN,
                'vocab_size': self.vocab_size,
                'product_to_id': self.product_to_id,
                'id_to_product': self.id_to_product
            },
            'split_data': split_data
        }
        
        print("===== DATA PREPARATION COMPLETED =====")
        return result


def plot_data_statistics(data_pipeline):
    """
    Plot statistics about the prepared data.
    
    Args:
        data_pipeline: BERT4RecDataPreparation instance
    """
    df = data_pipeline.df
    
    # Calculate statistics
    sequence_lengths = [len(seq) if isinstance(seq, list) else 0 for seq in df['customer_sequence']]
    
    basket_sizes = []
    for sequence in df['customer_sequence']:
        if isinstance(sequence, list):
            for basket in sequence:
                if isinstance(basket, list):
                    basket_sizes.append(len(basket))
    
    target_sizes = [len(basket) if isinstance(basket, list) else 0 for basket in df['last_basket']]
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot sequence lengths
    plt.subplot(2, 2, 1)
    plt.hist(sequence_lengths, bins=20)
    plt.axvline(x=data_pipeline.max_seq_length, color='r', linestyle='--', label=f'Max length: {data_pipeline.max_seq_length}')
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Number of Baskets')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot basket sizes
    plt.subplot(2, 2, 2)
    plt.hist(basket_sizes, bins=20)
    plt.axvline(x=data_pipeline.max_basket_size, color='r', linestyle='--', label=f'Max size: {data_pipeline.max_basket_size}')
    plt.title('Distribution of Basket Sizes')
    plt.xlabel('Number of Items')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot target basket sizes
    plt.subplot(2, 2, 3)
    plt.hist(target_sizes, bins=20)
    plt.axvline(x=data_pipeline.max_basket_size, color='r', linestyle='--', label=f'Max size: {data_pipeline.max_basket_size}')
    plt.title('Distribution of Target Basket Sizes')
    plt.xlabel('Number of Items')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot product ID distribution
    product_counts = {}
    for sequence in df['customer_sequence']:
        if isinstance(sequence, list):
            for basket in sequence:
                if isinstance(basket, list):
                    for product in basket:
                        product_counts[product] = product_counts.get(product, 0) + 1
    
    for basket in df['last_basket']:
        if isinstance(basket, list):
            for product in basket:
                product_counts[product] = product_counts.get(product, 0) + 1
    
    if product_counts:
        top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        plt.subplot(2, 2, 4)
        plt.bar([str(p[0]) for p in top_products], [p[1] for p in top_products])
        plt.title('Top 20 Products by Frequency')
        plt.xlabel('Product ID')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_pipeline.output_dir, 'data_statistics.png'))
    plt.close()


def main():
    """
    Main function to run the data preparation pipeline.
    """
    # Set parameters
    data_file = 'data.csv'  # Path to the input data file
    output_dir = './bert4rec_data/'  # Directory to save the prepared data
    max_seq_length = 50  # Maximum number of baskets in a sequence
    max_basket_size = 20  # Maximum number of items in a basket
    mask_prob = 0.15  # Probability of masking a basket for BERT pre-training
    
    # Create and run the data preparation pipeline
    data_pipeline = BERT4RecDataPreparation(
        data_file=data_file,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        max_basket_size=max_basket_size,
        mask_prob=mask_prob
    )
    
    # Run the complete pipeline
    result = data_pipeline.run_pipeline()
    
    # Plot statistics about the data
    plot_data_statistics(data_pipeline)
    
    print(f"\nData preparation complete! Files saved to {output_dir}")
    print("You can now proceed with training your BERT4Rec model for next basket prediction.")


if __name__ == "__main__":
    main()