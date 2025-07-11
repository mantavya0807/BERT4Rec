import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from tqdm import tqdm
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')

class SimilarityEmbeddingCreator:
    """
    Creates embeddings specifically designed for product similarity and grouping.
    Separate from BERT4Rec embeddings which are optimized for recommendations.
    """
    
    def __init__(self):
        self.products_df = None
        self.similarity_embeddings = None
        self.bert4rec_embeddings = None
        self.tfidf_vectorizer = None
        
    def load_data(self):
        """Load product data and existing BERT4Rec embeddings"""
        print("🔄 Loading product data...")
        
        # Load products
        product_paths = [
            './processed_data/products.csv',
            './BERT4Rec/processed_data/products.csv',
            './Product.csv',
            './products.csv'
        ]
        
        products_path = None
        for path in product_paths:
            if os.path.exists(path):
                products_path = path
                break
        
        if products_path is None:
            raise FileNotFoundError("❌ Could not find products file")
            
        self.products_df = pd.read_csv(products_path)
        print(f"✅ Loaded products: {len(self.products_df)} products")
        
        # Load existing BERT4Rec embeddings for reference
        embedding_paths = [
            './BERT4Rec/bert4rec_model_v2/item_embeddings_test.npy',
            './BERT4Rec/bert4rec_model_v2/item_embeddings.npy',
            './bert4rec_model_v2/item_embeddings_test.npy',
            './bert4rec_model_v2/item_embeddings.npy',
            './item_embeddings_test.npy',
            './item_embeddings.npy'
        ]
        
        for path in embedding_paths:
            if os.path.exists(path):
                self.bert4rec_embeddings = np.load(path)
                print(f"✅ Loaded BERT4Rec embeddings: {self.bert4rec_embeddings.shape}")
                break
        
        # Load additional data if available
        self.load_additional_product_data()
        
        return True
    
    def load_additional_product_data(self):
        """Load additional product metadata for better embeddings"""
        try:
            # Load aisles and departments if available
            aisles_paths = [
                './processed_data/aisles.csv',
                './BERT4Rec/processed_data/aisles.csv',
                './aisles.csv'
            ]
            
            departments_paths = [
                './processed_data/departments.csv',
                './BERT4Rec/processed_data/departments.csv',
                './departments.csv'
            ]
            
            for path in aisles_paths:
                if os.path.exists(path):
                    aisles_df = pd.read_csv(path)
                    self.products_df = self.products_df.merge(aisles_df, on='aisle_id', how='left')
                    print(f"✅ Loaded aisles data: {len(aisles_df)} aisles")
                    break
            
            for path in departments_paths:
                if os.path.exists(path):
                    departments_df = pd.read_csv(path)
                    self.products_df = self.products_df.merge(departments_df, on='department_id', how='left')
                    print(f"✅ Loaded departments data: {len(departments_df)} departments")
                    break
                    
        except Exception as e:
            print(f"⚠️  Could not load additional data: {e}")
    
    def preprocess_product_text(self):
        """Create rich text representations for each product"""
        print("🔄 Creating text representations for products...")
        
        product_texts = []
        
        for _, row in tqdm(self.products_df.iterrows(), total=len(self.products_df), desc="Processing products"):
            # Start with product name
            product_name = str(row.get('product_name', ''))
            
            # Add department and aisle info if available
            department = str(row.get('department', ''))
            aisle = str(row.get('aisle', ''))
            
            # Clean and normalize text
            clean_name = self.clean_product_name(product_name)
            
            # Create comprehensive text representation
            # Format: "clean_name [repeated for emphasis] department aisle"
            text_parts = []
            
            # Add cleaned product name multiple times for emphasis
            text_parts.extend([clean_name] * 3)
            
            # Add department and aisle
            if department and department != 'nan':
                text_parts.append(department.lower())
            if aisle and aisle != 'nan':
                text_parts.append(aisle.lower())
            
            # Add brand extraction
            brand = self.extract_brand(product_name)
            if brand:
                text_parts.extend([brand] * 2)  # Brand is important for grouping
            
            # Add category keywords
            category_keywords = self.extract_category_keywords(product_name)
            text_parts.extend(category_keywords)
            
            final_text = ' '.join(text_parts)
            product_texts.append(final_text)
        
        print(f"✅ Created text representations for {len(product_texts)} products")
        return product_texts
    
    def clean_product_name(self, name):
        """Clean product name for better matching"""
        if pd.isna(name):
            return ""
        
        name = str(name).lower()
        
        # Remove size/quantity information that doesn't help with similarity
        name = re.sub(r'\b\d+(\.\d+)?\s*(oz|lb|mg|g|kg|ml|l|ct|count|pack|pk)\b', '', name)
        name = re.sub(r'\b\d+\s*x\s*\d+\b', '', name)  # Remove "12 x 8" patterns
        
        # Remove common non-descriptive words
        stop_words = ['organic', 'natural', 'premium', 'classic', 'original', 'fresh', 
                     'pure', 'select', 'choice', 'quality', 'perfect', 'great', 'best']
        for word in stop_words:
            name = re.sub(rf'\b{word}\b', '', name)
        
        # Clean up extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def extract_brand(self, product_name):
        """Extract brand name from product name"""
        if pd.isna(product_name):
            return ""
        
        # Common brand patterns - usually the first word or two
        words = str(product_name).split()
        if len(words) > 0:
            # Take first 1-2 words as potential brand
            potential_brand = words[0].lower()
            
            # Filter out common generic terms
            generic_terms = ['organic', 'fresh', 'premium', 'select', 'choice', 'great', 
                           'perfect', 'classic', 'original', 'natural', 'pure']
            
            if potential_brand not in generic_terms and len(potential_brand) > 2:
                return potential_brand
        
        return ""
    
    def extract_category_keywords(self, product_name):
        """Extract category-related keywords from product name"""
        if pd.isna(product_name):
            return []
        
        name_lower = str(product_name).lower()
        
        # Define category keywords
        category_keywords = {
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy'],
            'meat': ['chicken', 'beef', 'pork', 'turkey', 'ham', 'bacon', 'sausage', 'meat'],
            'produce': ['apple', 'banana', 'orange', 'lettuce', 'tomato', 'potato', 'onion', 'carrot'],
            'beverages': ['juice', 'soda', 'water', 'tea', 'coffee', 'drink', 'beverage'],
            'bakery': ['bread', 'bagel', 'muffin', 'cake', 'cookie', 'bakery'],
            'frozen': ['frozen', 'ice', 'cream'],
            'pantry': ['pasta', 'rice', 'cereal', 'sauce', 'oil', 'vinegar'],
            'snacks': ['chips', 'crackers', 'nuts', 'snack'],
            'cleaning': ['detergent', 'soap', 'cleaner', 'paper', 'towel'],
            'personal_care': ['shampoo', 'toothpaste', 'lotion', 'deodorant']
        }
        
        found_keywords = []
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    found_keywords.append(category)
                    found_keywords.append(keyword)
                    break  # Only add category once
        
        return found_keywords
    
    def create_tfidf_embeddings(self, product_texts):
        """Create TF-IDF based embeddings for product similarity"""
        print("🔄 Creating TF-IDF similarity embeddings...")
        
        # Configure TF-IDF with parameters optimized for product similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,          # Limit vocabulary size
            min_df=2,                   # Word must appear in at least 2 products
            max_df=0.8,                 # Ignore words that appear in >80% of products
            ngram_range=(1, 2),         # Use unigrams and bigrams
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',  # Better tokenization
            lowercase=True,
            stop_words='english'
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(product_texts)
        
        # Convert to dense array and normalize
        similarity_embeddings = tfidf_matrix.toarray()
        similarity_embeddings = normalize(similarity_embeddings, norm='l2')
        
        print(f"✅ Created TF-IDF embeddings: {similarity_embeddings.shape}")
        print(f"📊 Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return similarity_embeddings
    
    def enhance_with_categorical_features(self, tfidf_embeddings):
        """Enhance embeddings with categorical features"""
        print("🔄 Enhancing embeddings with categorical features...")
        
        # Create categorical feature vectors
        categorical_features = []
        
        for _, row in self.products_df.iterrows():
            features = []
            
            # Department one-hot encoding (simplified)
            department = str(row.get('department', 'unknown')).lower()
            dept_features = [1 if dept in department else 0 for dept in 
                           ['produce', 'dairy', 'meat', 'bakery', 'frozen', 'beverages', 'snacks']]
            features.extend(dept_features)
            
            # Aisle one-hot encoding (simplified)
            aisle = str(row.get('aisle', 'unknown')).lower()
            aisle_features = [1 if keyword in aisle else 0 for keyword in 
                            ['fresh', 'frozen', 'canned', 'packaged', 'organic']]
            features.extend(aisle_features)
            
            categorical_features.append(features)
        
        categorical_features = np.array(categorical_features, dtype=np.float32)
        
        # Normalize categorical features
        categorical_features = normalize(categorical_features, norm='l2')
        
        # Combine TF-IDF and categorical features
        # Weight: 80% TF-IDF, 20% categorical
        combined_embeddings = np.hstack([
            tfidf_embeddings * 0.8,
            categorical_features * 0.2
        ])
        
        # Final normalization
        combined_embeddings = normalize(combined_embeddings, norm='l2')
        
        print(f"✅ Enhanced embeddings: {combined_embeddings.shape}")
        
        return combined_embeddings
    
    def test_similarity_quality(self, embeddings, sample_size=1000):
        """Test the quality of similarity embeddings"""
        print(f"🧪 Testing similarity embedding quality...")
        
        # Sample products for testing
        sample_indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_products = self.products_df.iloc[sample_indices]
        
        print(f"📊 Testing similarity quality with {len(sample_indices)} products...")
        
        # Find most similar products for a few examples
        examples_found = 0
        good_examples = []
        
        for i in range(min(20, len(sample_embeddings))):
            # Calculate similarities
            similarities = cosine_similarity([sample_embeddings[i]], sample_embeddings)[0]
            
            # Get top 5 most similar (excluding self)
            top_indices = np.argsort(similarities)[-6:-1][::-1]  # Exclude self, get top 5
            
            product_name = sample_products.iloc[i]['product_name']
            similar_products = []
            
            for idx in top_indices:
                similar_name = sample_products.iloc[idx]['product_name']
                similarity = similarities[idx]
                similar_products.append((similar_name, similarity))
            
            # Check if this is a good example (similar names/brands)
            is_good_example = self.is_good_similarity_example(product_name, similar_products)
            
            if is_good_example and examples_found < 5:
                good_examples.append({
                    'query': product_name,
                    'similar': similar_products
                })
                examples_found += 1
        
        # Show examples
        if good_examples:
            print(f"\n✅ Quality Examples Found:")
            for i, example in enumerate(good_examples):
                print(f"\n{i+1}. Query: {example['query']}")
                print(f"   Most similar products:")
                for j, (name, sim) in enumerate(example['similar'][:3]):
                    print(f"   {j+1}. {name} (similarity: {sim:.3f})")
        else:
            print(f"⚠️  No clear quality examples found in sample")
        
        # Calculate overall statistics
        all_similarities = []
        for i in range(len(sample_embeddings)):
            sims = cosine_similarity([sample_embeddings[i]], sample_embeddings)[0]
            # Take top similarities (excluding self)
            top_sims = np.sort(sims)[-10:-1]  # Top 9 excluding self
            all_similarities.extend(top_sims)
        
        print(f"\n📈 Similarity Statistics:")
        print(f"   Mean similarity: {np.mean(all_similarities):.4f}")
        print(f"   Std similarity: {np.std(all_similarities):.4f}")
        print(f"   95th percentile: {np.percentile(all_similarities, 95):.4f}")
        
        return np.mean(all_similarities), good_examples
    
    def is_good_similarity_example(self, query_product, similar_products):
        """Check if similarity results make intuitive sense"""
        query_words = set(query_product.lower().split())
        
        # Check if similar products share meaningful words with query
        shared_word_counts = []
        
        for similar_name, _ in similar_products:
            similar_words = set(similar_name.lower().split())
            shared_words = query_words.intersection(similar_words)
            # Filter out common stop words
            meaningful_shared = [w for w in shared_words if len(w) > 3 and 
                               w not in ['organic', 'natural', 'fresh', 'premium']]
            shared_word_counts.append(len(meaningful_shared))
        
        # Good example if most similar products share meaningful words
        return sum(shared_word_counts) >= 2
    
    def save_similarity_embeddings(self, embeddings, filename="similarity_embeddings"):
        """Save the similarity embeddings"""
        print(f"💾 Saving similarity embeddings...")
        
        # Save embeddings
        np.save(f"{filename}.npy", embeddings)
        
        # Save metadata
        metadata = {
            'embedding_type': 'product_similarity',
            'method': 'tfidf_plus_categorical',
            'shape': list(embeddings.shape),
            'created_for': 'product_grouping_and_similarity',
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            'products_count': len(self.products_df)
        }
        
        with open(f"{filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save vocabulary for reference - FIX THE JSON SERIALIZATION ERROR
        if self.tfidf_vectorizer:
            # Convert numpy int64 values to regular Python integers
            vocabulary = {word: int(idx) for word, idx in self.tfidf_vectorizer.vocabulary_.items()}
            with open(f"{filename}_vocabulary.json", 'w') as f:
                json.dump(vocabulary, f, indent=2)
        
        print(f"✅ Saved similarity embeddings:")
        print(f"   - {filename}.npy ({embeddings.shape})")
        print(f"   - {filename}_metadata.json")
        print(f"   - {filename}_vocabulary.json")
        
        return f"{filename}.npy"
    
    def create_similarity_embeddings(self):
        """Complete pipeline to create similarity-focused embeddings"""
        print("🚀 CREATING SIMILARITY-FOCUSED EMBEDDINGS")
        print("Separate from BERT4Rec embeddings - optimized for product grouping")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Create text representations
        product_texts = self.preprocess_product_text()
        
        # Create TF-IDF embeddings
        tfidf_embeddings = self.create_tfidf_embeddings(product_texts)
        
        # Enhance with categorical features
        similarity_embeddings = self.enhance_with_categorical_features(tfidf_embeddings)
        
        # Test quality
        avg_similarity, examples = self.test_similarity_quality(similarity_embeddings)
        
        # Save embeddings
        embedding_file = self.save_similarity_embeddings(similarity_embeddings, "product_similarity_embeddings")
        
        print(f"\n🎉 SIMILARITY EMBEDDINGS CREATED!")
        print(f"✅ Created embeddings: {similarity_embeddings.shape}")
        print(f"✅ Average similarity: {avg_similarity:.4f}")
        print(f"✅ Quality examples: {len(examples)}")
        print(f"✅ Saved to: {embedding_file}")
        
        return similarity_embeddings, embedding_file
    
    def compare_with_bert4rec(self, similarity_embeddings):
        """Compare similarity embeddings with BERT4Rec embeddings"""
        if self.bert4rec_embeddings is None:
            print("⚠️  No BERT4Rec embeddings loaded for comparison")
            return
        
        print(f"\n📊 COMPARISON: Similarity vs BERT4Rec Embeddings")
        print("="*60)
        
        print(f"Similarity Embeddings:")
        print(f"   Shape: {similarity_embeddings.shape}")
        print(f"   Purpose: Product grouping and similarity")
        print(f"   Method: TF-IDF + categorical features")
        
        print(f"\nBERT4Rec Embeddings:")
        print(f"   Shape: {self.bert4rec_embeddings.shape}")
        print(f"   Purpose: Next-basket recommendation")
        print(f"   Method: BERT4Rec training on shopping sequences")
        
        print(f"\n💡 Integration Strategy:")
        print(f"   1. Use SIMILARITY embeddings for product grouping")
        print(f"   2. Use BERT4Rec embeddings for recommendations")
        print(f"   3. Combine both for comprehensive product understanding")


def create_integrated_system():
    """Create an integrated system using both embedding types"""
    print(f"\n🔗 INTEGRATED SYSTEM DESIGN")
    print("="*60)
    
    integration_guide = {
        "system_components": {
            "similarity_embeddings": {
                "purpose": "Product grouping and categorization",
                "use_cases": [
                    "Group similar products (all Pringles flavors)",
                    "Find product substitutes",
                    "Create product families for clustering",
                    "Product search and discovery"
                ],
                "file": "product_similarity_embeddings.npy"
            },
            "bert4rec_embeddings": {
                "purpose": "Shopping behavior and recommendations",
                "use_cases": [
                    "Next basket prediction",
                    "Customer behavior analysis", 
                    "Shopping pattern clustering",
                    "Recommendation systems"
                ],
                "file": "item_embeddings.npy"
            }
        },
        "integration_workflow": {
            "step1": "Use similarity embeddings to group ~50k products into ~5k families",
            "step2": "Use BERT4Rec embeddings for recommendation within families",
            "step3": "Cluster product families (not individual products) for market analysis",
            "step4": "Use both embeddings for comprehensive product understanding"
        },
        "code_example": '''
# Load both embedding types
similarity_embeddings = np.load("product_similarity_embeddings.npy")
bert4rec_embeddings = np.load("item_embeddings.npy")

# Step 1: Group products using similarity embeddings
product_groups = create_product_groups(similarity_embeddings, threshold=0.6)

# Step 2: Use BERT4Rec for recommendations within groups  
recommendations = bert4rec_recommend(bert4rec_embeddings, user_history)

# Step 3: Cluster product groups (not individual products)
cluster_groups(product_groups, n_clusters=200)
'''
    }
    
    # Save integration guide
    with open("embedding_integration_guide.json", 'w') as f:
        json.dump(integration_guide, f, indent=2)
    
    print(f"✅ Integration guide saved to: embedding_integration_guide.json")
    
    return integration_guide


def main():
    """Main function to create similarity embeddings"""
    try:
        # Create similarity embeddings
        creator = SimilarityEmbeddingCreator()
        similarity_embeddings, embedding_file = creator.create_similarity_embeddings()
        
        # Compare with existing BERT4Rec embeddings
        creator.compare_with_bert4rec(similarity_embeddings)
        
        # Create integration guide
        integration_guide = create_integrated_system()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Use {embedding_file} for product grouping")
        print(f"2. Keep BERT4Rec embeddings for recommendations") 
        print(f"3. Follow integration_guide.json for combined system")
        print(f"4. Group products first, then cluster the groups")
        
        return similarity_embeddings, creator
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    result = main()