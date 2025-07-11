#!/usr/bin/env python3
"""
Complete Fuzzy Product Grouper
Single file solution - no imports needed except standard libraries
"""

import pandas as pd
import numpy as np
import json
import os
import re
from collections import defaultdict, Counter
import warnings

# Try to import fuzzywuzzy, fall back to basic similarity if not available
try:
    from fuzzywuzzy import fuzz
    HAS_FUZZYWUZZY = True
    print("Using fuzzywuzzy for similarity matching")
except ImportError:
    HAS_FUZZYWUZZY = False
    print("fuzzywuzzy not available - using basic similarity")
    print("For better results, install: pip install fuzzywuzzy python-Levenshtein")

warnings.filterwarnings('ignore')

def basic_similarity(s1, s2):
    """Basic similarity function if fuzzywuzzy is not available"""
    s1, s2 = s1.lower(), s2.lower()
    
    # Simple Jaccard similarity on words
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0
    
    return int((len(intersection) / len(union)) * 100)

def get_similarity(s1, s2):
    """Get similarity score between two strings"""
    if HAS_FUZZYWUZZY:
        return fuzz.ratio(s1, s2)
    else:
        return basic_similarity(s1, s2)

class SmartProductGrouper:
    """
    Smart product grouper using fuzzy matching and intelligent sorting.
    Groups products by their descriptive terms and similarity.
    """
    
    def __init__(self, similarity_threshold=80, min_family_size=2, max_family_size=15):
        self.similarity_threshold = similarity_threshold
        self.min_family_size = min_family_size
        self.max_family_size = max_family_size
        self.products_df = None
        self.product_families = {}
        
    def load_products(self):
        """Load product data from various possible locations"""
        print("Loading product data...")
        
        possible_paths = [
            './processed_data/products.csv',
            './BERT4Rec/processed_data/products.csv',
            './Product.csv',
            './products.csv',
            'products.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.products_df = pd.read_csv(path)
                    print(f"Loaded {len(self.products_df):,} products from {path}")
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
        else:
            print("No product file found in these locations:")
            for path in possible_paths:
                print(f"   • {path}")
            print("\nFiles in current directory:")
            try:
                files = [f for f in os.listdir('.') if f.endswith('.csv')]
                for f in files:
                    print(f"   • {f}")
            except:
                pass
            return False
        
        # Load additional metadata if available
        self._load_categories()
        return True
    
    def _load_categories(self):
        """Load aisles and departments if available"""
        try:
            # Load aisles
            aisle_paths = [
                './processed_data/aisles.csv', 
                './BERT4Rec/processed_data/aisles.csv', 
                './aisles.csv',
                'aisles.csv'
            ]
            for path in aisle_paths:
                if os.path.exists(path):
                    aisles_df = pd.read_csv(path)
                    self.products_df = self.products_df.merge(aisles_df, on='aisle_id', how='left')
                    print(f"Added aisle information ({len(aisles_df)} aisles)")
                    break
            
            # Load departments  
            dept_paths = [
                './processed_data/departments.csv', 
                './BERT4Rec/processed_data/departments.csv', 
                './departments.csv',
                'departments.csv'
            ]
            for path in dept_paths:
                if os.path.exists(path):
                    dept_df = pd.read_csv(path)
                    self.products_df = self.products_df.merge(dept_df, on='department_id', how='left')
                    print(f"Added department information ({len(dept_df)} departments)")
                    break
                    
        except Exception as e:
            print(f"Could not load category data: {e}")
    
    def extract_key_terms(self, product_name):
        """Extract key descriptive terms from product name"""
        # Clean the name
        name = product_name.lower().strip()
        
        # Remove common filler words
        stop_words = {
            'the', 'and', 'or', 'of', 'in', 'to', 'a', 'an', 'with', 'for', 'on', 'at', 'by',
            'organic', 'natural', 'fresh', 'premium', 'select', 'choice', 'great', 'best',
            'value', 'pack', 'count', 'size', 'oz', 'lb', 'lbs', 'ct', 'piece', 'pieces'
        }
        
        # Split into words and clean
        words = re.findall(r'\b[a-zA-Z]+\b', name)
        words = [w for w in words if len(w) > 2 and w not in stop_words]
        
        if not words:
            return [name]  # fallback
        
        # Extract key terms - focus on last meaningful words (descriptive terms)
        key_terms = []
        
        # Last word is usually most descriptive
        if words:
            key_terms.append(words[-1])
        
        # Second to last word if it's descriptive
        if len(words) > 1:
            key_terms.append(words[-2])
        
        # Look for important middle terms (flavors, types, etc.)
        important_patterns = [
            r'\b(chocolate|vanilla|strawberry|berry|fruit|apple|orange|banana)\b',
            r'\b(cheese|cheddar|swiss|mozzarella|milk|cream|butter)\b',
            r'\b(chicken|beef|pork|turkey|fish|salmon|tuna)\b',
            r'\b(whole|skim|fat|free|low|reduced|diet)\b',
            r'\b(spicy|mild|hot|sweet|sour|salty)\b'
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, name)
            key_terms.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:3]  # Return top 3 key terms
    
    def create_smart_sort_key(self, product_name):
        """Create a smart sorting key that groups similar products together"""
        key_terms = self.extract_key_terms(product_name)
        
        # Create sort key: primary term + secondary term + full name
        if len(key_terms) >= 2:
            sort_key = f"{key_terms[0]}_{key_terms[1]}_{product_name.lower()}"
        elif len(key_terms) == 1:
            sort_key = f"{key_terms[0]}_{product_name.lower()}"
        else:
            sort_key = product_name.lower()
        
        return sort_key
    
    def smart_sort_products(self):
        """Sort products intelligently to group similar items together"""
        print("Smart-sorting products by descriptive terms...")
        
        # Create sort keys for all products
        self.products_df['sort_key'] = self.products_df['product_name'].apply(self.create_smart_sort_key)
        self.products_df['key_terms'] = self.products_df['product_name'].apply(self.extract_key_terms)
        
        # Sort by the smart key
        self.products_df = self.products_df.sort_values('sort_key').reset_index(drop=True)
        
        print("Products sorted - similar items are now adjacent")
        
        # Show some examples of the sorting
        print("\nSmart Sorting Examples (first 15 products):")
        for i in range(min(15, len(self.products_df))):
            row = self.products_df.iloc[i]
            key_terms = ', '.join(row['key_terms']) if row['key_terms'] else 'none'
            print(f"   {i+1:2d}. {row['product_name']} (key: {key_terms})")
    
    def create_fuzzy_families(self):
        """Create product families using fuzzy matching on sorted products"""
        print(f"\nCreating fuzzy families (threshold: {self.similarity_threshold}%)...")
        
        families = {}
        family_id = 0
        used_products = set()
        
        total_products = len(self.products_df)
        processed = 0
        
        print(f"Processing {total_products:,} products...")
        
        for i, row in self.products_df.iterrows():
            processed += 1
            
            # Progress indicator
            if processed % 1000 == 0 or processed <= 20:
                print(f"   Progress: {processed:,}/{total_products:,} ({processed/total_products*100:.1f}%) - Found {len(families)} families")
            
            if i in used_products:
                continue
            
            current_product = {
                'index': i,
                'product_id': row['product_id'],
                'name': row['product_name'],
                'department': row.get('department', 'Unknown'),
                'aisle': row.get('aisle', 'Unknown'),
                'key_terms': row['key_terms']
            }
            
            # Start a new family
            family_products = [current_product]
            used_products.add(i)
            
            # Look for similar products in nearby positions (smart sorting benefit)
            search_window = min(100, total_products - i - 1)  # Look ahead up to 100 products
            
            for j in range(i + 1, min(i + 1 + search_window, total_products)):
                if j in used_products or len(family_products) >= self.max_family_size:
                    break
                
                candidate_row = self.products_df.iloc[j]
                candidate_name = candidate_row['product_name']
                
                # Check fuzzy similarity
                similarity = get_similarity(
                    self._clean_for_comparison(current_product['name']),
                    self._clean_for_comparison(candidate_name)
                )
                
                if similarity >= self.similarity_threshold:
                    # Additional validation checks
                    if self._are_compatible_products(current_product, candidate_row):
                        candidate_product = {
                            'index': j,
                            'product_id': candidate_row['product_id'],
                            'name': candidate_name,
                            'department': candidate_row.get('department', 'Unknown'),
                            'aisle': candidate_row.get('aisle', 'Unknown'),
                            'key_terms': candidate_row['key_terms']
                        }
                        family_products.append(candidate_product)
                        used_products.add(j)
            
            # Create family if it meets size requirements
            if len(family_products) >= self.min_family_size:
                family_name = self._generate_family_name(family_products)
                
                families[family_id] = {
                    'family_id': family_id,
                    'family_name': family_name,
                    'size': len(family_products),
                    'products': family_products,
                    'representative_product': family_products[0]['name'],
                    'departments': list(set([p['department'] for p in family_products])),
                    'aisles': list(set([p['aisle'] for p in family_products])),
                    'avg_similarity': self._calculate_family_similarity(family_products)
                }
                family_id += 1
        
        self.product_families = families
        
        print(f"\nCreated {len(families):,} product families")
        print(f"Grouped {len(used_products):,} products ({len(used_products)/total_products*100:.1f}%)")
        
        return families
    
    def _clean_for_comparison(self, name):
        """Clean product name for better fuzzy matching"""
        # Remove common variations that don't affect similarity
        cleaned = name.lower()
        
        # Remove size/quantity indicators
        cleaned = re.sub(r'\b\d+\s*(oz|lb|lbs|ct|count|pack|piece|pieces|size)\b', '', cleaned)
        
        # Remove brand-specific terms that might interfere
        remove_terms = ['organic', 'natural', 'premium', 'select', 'choice', 'great value']
        for term in remove_terms:
            cleaned = cleaned.replace(term, '')
        
        # Clean up extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _are_compatible_products(self, product1, product2_row):
        """Check if two products are compatible for grouping"""
        # Check if departments are compatible
        dept1 = product1['department']
        dept2 = product2_row.get('department', 'Unknown')
        
        # Some departments should not be mixed
        incompatible_pairs = [
            ('meat seafood', 'produce'),
            ('dairy eggs', 'household'),
            ('personal care', 'pantry'),
            ('pets', 'babies')
        ]
        
        for pair in incompatible_pairs:
            if (dept1 in pair and dept2 in pair) and dept1 != dept2:
                return False
        
        # Check for obvious category conflicts in names
        name1_lower = product1['name'].lower()
        name2_lower = product2_row['product_name'].lower()
        
        conflicts = [
            (['tea', 'coffee'], ['peas', 'beans']),
            (['soap', 'detergent'], ['food', 'snack']),
            (['dog', 'cat', 'pet'], ['baby', 'infant'])
        ]
        
        for group1, group2 in conflicts:
            has_group1 = any(term in name1_lower for term in group1)
            has_group2 = any(term in name2_lower for term in group2)
            
            if has_group1 and has_group2:
                return False
        
        return True
    
    def _generate_family_name(self, family_products):
        """Generate a meaningful family name"""
        # Find common terms across all products
        all_names = [p['name'].lower() for p in family_products]
        
        # Extract all key terms
        all_key_terms = []
        for product in family_products:
            all_key_terms.extend(product['key_terms'])
        
        # Find most common key terms
        term_counts = Counter(all_key_terms)
        common_terms = [term for term, count in term_counts.most_common(3) if count > 1]
        
        if len(common_terms) >= 2:
            return f"{common_terms[0].title()} {common_terms[1].title()} Family"
        elif len(common_terms) == 1:
            return f"{common_terms[0].title()} Products Family"
        
        # Fallback: use most descriptive words from first product
        first_product = family_products[0]
        if first_product['key_terms']:
            return f"{first_product['key_terms'][0].title()} Family"
        
        # Final fallback
        return f"{first_product['name'][:30]} Family"
    
    def _calculate_family_similarity(self, family_products):
        """Calculate average similarity within family"""
        if len(family_products) < 2:
            return 100.0
        
        similarities = []
        for i in range(len(family_products)):
            for j in range(i + 1, len(family_products)):
                sim = get_similarity(
                    self._clean_for_comparison(family_products[i]['name']),
                    self._clean_for_comparison(family_products[j]['name'])
                )
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0
    
    def show_family_examples(self, top_n=15):
        """Show examples of created families"""
        if not self.product_families:
            print("No families created yet")
            return
        
        # Sort families by size and similarity
        sorted_families = sorted(
            self.product_families.values(),
            key=lambda x: (x['size'], x['avg_similarity']),
            reverse=True
        )
        
        print(f"\nTOP {top_n} PRODUCT FAMILIES")
        print("=" * 70)
        
        for i, family in enumerate(sorted_families[:top_n]):
            print(f"\n{i+1:2d}. {family['family_name']}")
            print(f"    Size: {family['size']} products | Avg Similarity: {family['avg_similarity']:.1f}%")
            print(f"    Departments: {', '.join(family['departments'])}")
            print("    Products:")
            
            for j, product in enumerate(family['products']):
                print(f"       {j+1}. {product['name']}")
    
    def analyze_results(self):
        """Analyze the grouping results"""
        if not self.product_families:
            return
        
        total_products = len(self.products_df)
        grouped_products = sum(f['size'] for f in self.product_families.values())
        
        # Size distribution
        family_sizes = [f['size'] for f in self.product_families.values()]
        
        # Similarity distribution
        similarities = [f['avg_similarity'] for f in self.product_families.values()]
        
        print(f"\nFUZZY GROUPING ANALYSIS")
        print("=" * 50)
        print(f"Total Products: {total_products:,}")
        print(f"Products Grouped: {grouped_products:,} ({grouped_products/total_products*100:.1f}%)")
        print(f"Families Created: {len(self.product_families):,}")
        print(f"Average Family Size: {sum(family_sizes)/len(family_sizes):.1f}")
        print(f"Largest Family: {max(family_sizes)} products")
        print(f"Average Similarity: {sum(similarities)/len(similarities):.1f}%")
        
        # Size distribution
        size_dist = Counter(family_sizes)
        print(f"\nFamily Size Distribution:")
        for size in sorted(size_dist.keys()):
            print(f"   {size} products: {size_dist[size]:,} families")
        
        # Department analysis
        dept_counts = defaultdict(int)
        for family in self.product_families.values():
            for dept in family['departments']:
                dept_counts[dept] += 1
        
        print(f"\nTop Departments:")
        for dept, count in sorted(dept_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {dept}: {count:,} families")
    
    def save_results(self, filename='fuzzy_product_families.json'):
        """Save the results to a JSON file"""
        if not self.product_families:
            print("No families to save")
            return
        
        # Prepare data for JSON serialization
        save_data = {
            'metadata': {
                'total_families': len(self.product_families),
                'total_products_grouped': sum(f['size'] for f in self.product_families.values()),
                'similarity_threshold': self.similarity_threshold,
                'min_family_size': self.min_family_size,
                'max_family_size': self.max_family_size,
                'method': 'fuzzy_matching_with_smart_sorting',
                'avg_family_size': sum(f['size'] for f in self.product_families.values()) / len(self.product_families),
                'avg_similarity': sum(f['avg_similarity'] for f in self.product_families.values()) / len(self.product_families),
                'has_fuzzywuzzy': HAS_FUZZYWUZZY
            },
            'families': {}
        }
        
        # Convert families to JSON-serializable format
        for family_id, family_data in self.product_families.items():
            family_copy = family_data.copy()
            
            # Convert products to simple format
            simple_products = []
            for product in family_data['products']:
                simple_product = {
                    'product_id': int(product['product_id']),
                    'name': product['name'],
                    'department': product['department'],
                    'aisle': product['aisle']
                }
                simple_products.append(simple_product)
            
            family_copy['products'] = simple_products
            family_copy['avg_similarity'] = float(family_copy['avg_similarity'])
            
            save_data['families'][str(family_id)] = family_copy
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        print(f"Saved {len(self.product_families):,} families with {save_data['metadata']['total_products_grouped']:,} products")
    
    def search_families(self, search_term):
        """Search for families containing a specific product"""
        if not self.product_families:
            print("No families available. Run grouping first!")
            return
        
        search_term_lower = search_term.lower()
        matches = []
        
        for family_data in self.product_families.values():
            for product in family_data['products']:
                if search_term_lower in product['name'].lower():
                    matches.append((family_data, product))
        
        if not matches:
            print(f"No products found containing '{search_term}'")
            return
        
        print(f"\nSEARCH RESULTS for '{search_term}'")
        print("=" * 60)
        
        for i, (family_data, matched_product) in enumerate(matches[:10], 1):
            print(f"\n{i}. Found: {matched_product['name']}")
            print(f"   Family: {family_data['family_name']}")
            print(f"   Family size: {family_data['size']} products (avg similarity: {family_data['avg_similarity']:.1f}%)")
            
            if family_data['size'] > 1:
                print("   Related products:")
                for product in family_data['products']:
                    if product['product_id'] != matched_product['product_id']:
                        print(f"      • {product['name']}")
    
    def run_smart_grouping(self, similarity_threshold=None):
        """Run the complete smart product grouping pipeline"""
        if similarity_threshold:
            self.similarity_threshold = similarity_threshold
            
        print("SMART PRODUCT GROUPER")
        print("Fuzzy matching with intelligent sorting")
        print("=" * 60)
        print(f"Similarity threshold: {self.similarity_threshold}%")
        print(f"Family size range: {self.min_family_size}-{self.max_family_size} products")
        
        # Step 1: Load products
        if not self.load_products():
            return None
        
        # Step 2: Smart sort products
        self.smart_sort_products()
        
        # Step 3: Create fuzzy families
        self.create_fuzzy_families()
        
        # Step 4: Analyze results
        self.analyze_results()
        
        # Step 5: Show examples
        self.show_family_examples()
        
        # Step 6: Save results
        self.save_results()
        
        print(f"\nSMART GROUPING COMPLETE!")
        print(f"Created {len(self.product_families):,} families using fuzzy matching")
        print("Smart sorting improved grouping efficiency")
        print("Results saved to fuzzy_product_families.json")
        
        return self.product_families

def main():
    """Main function with interactive options"""
    print("FUZZY PRODUCT GROUPER")
    print("=" * 50)
    
    # Try to install fuzzywuzzy if not available
    if not HAS_FUZZYWUZZY:
        print("\nFor better results, install fuzzywuzzy:")
        print("   pip install fuzzywuzzy python-Levenshtein")
        
        try:
            choice = input("\nTry to install now? (y/n): ").lower().strip()
            if choice == 'y':
                import subprocess
                import sys
                
                print("Installing fuzzywuzzy...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fuzzywuzzy', 'python-Levenshtein'])
                print("Installation complete! Please restart the script.")
                return
        except:
            print("Installation failed. Continuing with basic similarity...")
    
    # Check if results already exist
    if os.path.exists('fuzzy_product_families.json'):
        print(f"\nFound existing results: fuzzy_product_families.json")
        choice = input("What would you like to do?\n1. Create new families\n2. Search existing families\nChoice (1/2): ").strip()
        
        if choice == '2':
            # Load and search existing families
            try:
                with open('fuzzy_product_families.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                families = data['families']
                metadata = data['metadata']
                
                print(f"\nLoaded {len(families)} families")
                print(f"Total products grouped: {metadata['total_products_grouped']:,}")
                print(f"Average similarity: {metadata['avg_similarity']:.1f}%")
                
                while True:
                    search_term = input("\nEnter product name to search (or 'quit'): ").strip()
                    if search_term.lower() == 'quit':
                        break
                    
                    if search_term:
                        found = False
                        for family_data in families.values():
                            for product in family_data['products']:
                                if search_term.lower() in product['name'].lower():
                                    print(f"\nFound: {product['name']}")
                                    print(f"   Family: {family_data['family_name']}")
                                    print(f"   Size: {family_data['size']} products")
                                    print("   Related products:")
                                    for other in family_data['products']:
                                        if other['product_id'] != product['product_id']:
                                            print(f"      • {other['name']}")
                                    found = True
                                    break
                            if found:
                                break
                        
                        if not found:
                            print(f"No product found containing '{search_term}'")
                
                return
            except Exception as e:
                print(f"Error loading existing results: {e}")
                print("Creating new families...")
    
    # Create new families
    print(f"\nCREATING NEW PRODUCT FAMILIES")
    
    # Get similarity threshold
    try:
        threshold = input(f"\nSimilarity threshold (70-90, default 80): ").strip()
        threshold = int(threshold) if threshold else 80
        threshold = max(50, min(95, threshold))  # Clamp to reasonable range
    except:
        threshold = 80
    
    try:
        grouper = SmartProductGrouper(
            similarity_threshold=threshold,
            min_family_size=2,
            max_family_size=12
        )
        
        families = grouper.run_smart_grouping()
        
        if families:
            print(f"\nUSAGE TIPS:")
            print(f"• Run again to search families: python {__file__}")
            print(f"• Lower threshold ({threshold-10}) for more families")
            print(f"• Higher threshold ({threshold+10}) for stricter grouping")
            print(f"• Results saved in fuzzy_product_families.json")
            
            # Offer quick search
            search_term = input(f"\nTry searching for a product (or press Enter to finish): ").strip()
            if search_term:
                grouper.search_families(search_term)
        
        return families
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()