#!/usr/bin/env python3
"""
Simple SVD-SCD Implementation
Direct implementation of the paper methodology: S = UΣV^T and v_j = Σ^(-1)U^T s_j
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import time

class SimpleSVD_SCD:
    """
    Simple SVD-SCD implementation following the paper exactly:
    1. S = UΣV^T (SVD decomposition)
    2. Reduce from m to r dimensions
    3. Map components: v_j = Σ^(-1)U^T s_j
    4. Create composite products from fuzzy families
    """
    
    def __init__(self, r_components=50):
        self.r_components = r_components
        
        # Core SVD matrices
        self.S = None          # Original matrix (m×n)
        self.U = None          # Left vectors (m×r) 
        self.Sigma = None      # Singular values (r×r)
        self.V = None          # Right vectors (n×r)
        self.Sigma_inv = None  # Inverse for mapping
        
        # Data
        self.embeddings = None
        self.families = None
        self.composite_products = {}
        self.analysis_results = None
        
        print(f"Simple SVD-SCD initialized (r={r_components})")
    
    def load_data(self):
        """Load embeddings and fuzzy families"""
        print("Loading data...")
        
        # Load embeddings
        embedding_paths = [
            './bert4rec_model_v2/item_embeddings_test.npy',
            './item_embeddings_test.npy'
        ]
        
        for path in embedding_paths:
            if os.path.exists(path):
                self.embeddings = np.load(path)[2:]  # Skip special tokens
                print(f"Loaded embeddings: {self.embeddings.shape}")
                break
        
        if self.embeddings is None:
            raise FileNotFoundError("No embedding file found")
        
        # Load fuzzy families
        family_paths = [
            './fuzzy_product_families.json',
            '../fuzzy_product_families.json'
        ]
        
        for path in family_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:  # Fixed: specify UTF-8 encoding
                    data = json.load(f)
                    self.families = data['families']
                print(f"Loaded families: {len(self.families)}")
                break
        
        if self.families is None:
            raise FileNotFoundError("No families file found")
        
        # Create S matrix (transpose embeddings to get m×n format)
        self.S = self.embeddings.T
        print(f"S matrix shape: {self.S.shape} (m×n)")
        
        return True
    
    def perform_svd(self):
        """Perform SVD decomposition: S = UΣV^T"""
        print("Performing SVD decomposition...")
        
        # Center the data
        S_mean = np.mean(self.S, axis=1, keepdims=True)
        S_centered = self.S - S_mean
        self.S_mean = S_mean
        
        # Don't use StandardScaler - just work with centered data
        # StandardScaler causes dimension mismatch issues
        S_normalized = S_centered
        
        # SVD using TruncatedSVD
        svd = TruncatedSVD(n_components=self.r_components, random_state=42)
        
        # Fit on transposed matrix (sklearn expects samples×features)
        svd.fit(S_normalized.T)
        
        # Extract matrices following paper notation
        self.U = svd.components_.T                    # m×r (features × components)
        self.Sigma = np.diag(svd.singular_values_)    # r×r diagonal
        self.V = svd.transform(S_normalized.T) / svd.singular_values_  # n×r (products × components)
        
        # Create inverse of Sigma for mapping formula
        self.Sigma_inv = np.diag(1.0 / svd.singular_values_)
        
        # Store for later use
        self.svd_model = svd
        
        explained_var = np.sum(svd.explained_variance_ratio_)
        
        print(f"SVD complete:")
        print(f"   U: {self.U.shape}, Σ: {self.Sigma.shape}, V: {self.V.shape}")
        print(f"   Explained variance: {explained_var:.3f} ({explained_var*100:.1f}%)")
        print(f"   Top singular values: {svd.singular_values_[:5]}")
        
        return explained_var
    
    def create_composite_products(self):
        """Create composite products from fuzzy families"""
        print("Creating composite products from families...")
        
        composite_count = 0
        
        for family_id, family_data in self.families.items():
            products = family_data['products']
            
            # Get embeddings for products in this family
            family_embeddings = []
            valid_products = []
            
            for product in products:
                # Simple approach: use product index directly if available
                # This assumes the product_id maps to embedding index
                try:
                    product_id = int(product['product_id'])
                    if 0 <= product_id < len(self.embeddings):
                        family_embeddings.append(self.embeddings[product_id])
                        valid_products.append(product)
                except:
                    continue
            
            # Create composite if we have enough products
            if len(family_embeddings) >= 2:
                family_embeddings = np.array(family_embeddings)
                
                # Composite is the centroid of family members
                composite_embedding = np.mean(family_embeddings, axis=0)
                
                # Map to reduced space using the formula: v_j = Σ^(-1)U^T s_j
                composite_reduced = self.map_to_reduced_space(composite_embedding)
                
                # Map each component to reduced space
                component_reduced = []
                for emb in family_embeddings:
                    comp_reduced = self.map_to_reduced_space(emb)
                    component_reduced.append(comp_reduced)
                
                # Store composite product
                self.composite_products[family_id] = {
                    'family_name': family_data['family_name'],
                    'size': len(valid_products),
                    'products': valid_products,
                    'composite_original': composite_embedding,
                    'composite_reduced': composite_reduced,
                    'components_original': family_embeddings,
                    'components_reduced': np.array(component_reduced),
                    'avg_similarity': family_data['avg_similarity']
                }
                
                composite_count += 1
        
        print(f"Created {composite_count} composite products")
        return composite_count
    
    def map_to_reduced_space(self, embedding):
        """
        Map product embedding to reduced space using: v_j = Σ^(-1)U^T s_j
        """
        # Center the embedding (don't use StandardScaler)
        s_j = embedding - self.S_mean.flatten()
        
        # Apply mapping formula: v_j = Σ^(-1)U^T s_j
        v_j = self.Sigma_inv @ self.U.T @ s_j
        
        return v_j
    
    def analyze_families(self):
        """Simple analysis of family quality"""
        print("Analyzing family quality...")
        
        if not self.composite_products:
            print("No composite products to analyze")
            return
        
        # Calculate intra-family vs inter-family distances
        intra_distances = []
        inter_distances = []
        
        families = list(self.composite_products.values())
        
        # Intra-family distances
        for family in families:
            components = family['components_reduced']
            composite = family['composite_reduced']
            
            # Distance from each component to composite
            for comp in components:
                dist = np.linalg.norm(comp - composite)
                intra_distances.append(dist)
        
        # Inter-family distances (sample)
        for i in range(min(50, len(families))):
            for j in range(i+1, min(50, len(families))):
                dist = np.linalg.norm(families[i]['composite_reduced'] - families[j]['composite_reduced'])
                inter_distances.append(dist)
        
        # Statistics
        avg_intra = np.mean(intra_distances)
        avg_inter = np.mean(inter_distances)
        separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
        
        print(f"Analysis Results:")
        print(f"   Average intra-family distance: {avg_intra:.4f}")
        print(f"   Average inter-family distance: {avg_inter:.4f}")
        print(f"   Separation ratio: {separation_ratio:.2f}")
        
        if separation_ratio > 1.5:
            print("   Good family separation!")
        elif separation_ratio > 1.2:
            print("   Moderate separation")
        else:
            print("   Poor separation")
        
        return {
            'avg_intra_distance': avg_intra,
            'avg_inter_distance': avg_inter,
            'separation_ratio': separation_ratio,
            'intra_distances': intra_distances,
            'inter_distances': inter_distances
        }
    
    def visualize_results(self, output_dir='./svd_results'):
        """Create simple visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating visualizations in {output_dir}...")
        
        # 1. Singular values plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        singular_vals = np.diag(self.Sigma)
        plt.plot(range(1, len(singular_vals) + 1), singular_vals, 'bo-')
        plt.title('Singular Values')
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.grid(True, alpha=0.3)
        
        # 2. Explained variance
        plt.subplot(1, 2, 2)
        var_explained = singular_vals**2 / np.sum(singular_vals**2)
        cumsum_var = np.cumsum(var_explained)
        plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Component')
        plt.ylabel('Cumulative Variance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/svd_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Family sizes
        if self.composite_products:
            family_sizes = [fam['size'] for fam in self.composite_products.values()]
            
            plt.figure(figsize=(10, 6))
            plt.hist(family_sizes, bins=range(2, max(family_sizes)+2), alpha=0.7, edgecolor='black')
            plt.title('Distribution of Family Sizes')
            plt.xlabel('Family Size')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/family_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 2D visualization of composite products
            if len(self.composite_products) > 1:
                composites = np.array([fam['composite_reduced'][:2] for fam in self.composite_products.values()])
                sizes = [fam['size'] for fam in self.composite_products.values()]
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(composites[:, 0], composites[:, 1], 
                                    c=sizes, s=60, alpha=0.7, cmap='viridis')
                plt.colorbar(scatter, label='Family Size')
                plt.title('Composite Products in Reduced Space\n(First Two Components)')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{output_dir}/composite_products_2d.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_dir='./svd_results'):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save composite products
        save_data = {}
        for comp_id, comp_data in self.composite_products.items():
            save_data[comp_id] = {
                'family_name': comp_data['family_name'],
                'size': comp_data['size'],
                'avg_similarity': comp_data['avg_similarity'],
                'composite_reduced': comp_data['composite_reduced'].tolist(),
                'products': comp_data['products']
            }
        
        with open(f'{output_dir}/composite_products.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Save SVD results
        svd_data = {
            'r_components': self.r_components,
            'singular_values': np.diag(self.Sigma).tolist(),
            'explained_variance': float(np.sum(self.svd_model.explained_variance_ratio_)),
            'matrix_shapes': {
                'U': self.U.shape,
                'Sigma': self.Sigma.shape,
                'V': self.V.shape
            }
        }
        
        with open(f'{output_dir}/svd_info.json', 'w', encoding='utf-8') as f:
            json.dump(svd_data, f, indent=2)
        
        # Save matrices
        np.save(f'{output_dir}/U_matrix.npy', self.U)
        np.save(f'{output_dir}/Sigma_matrix.npy', self.Sigma)
        np.save(f'{output_dir}/V_matrix.npy', self.V)
        
        print(f"Results saved to {output_dir}")
    
    def run_analysis(self):
        """Run complete SVD-SCD analysis"""
        print("RUNNING SIMPLE SVD-SCD ANALYSIS")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Load data
            self.load_data()
            
            # Perform SVD
            explained_var = self.perform_svd()
            
            # Create composites
            composite_count = self.create_composite_products()
            
            # Analyze
            self.analysis_results = self.analyze_families()
            
            # Visualize
            self.visualize_results()
            
            # Save
            self.save_results()
            
            total_time = time.time() - start_time
            
            print(f"\nANALYSIS COMPLETE!")
            print(f"Time: {total_time:.1f} seconds")
            print(f"Reduced from {self.S.shape[0]} to {self.r_components} dimensions")
            print(f"Explained variance: {explained_var*100:.1f}%")
            print(f"Created {composite_count} composite products")
            
            if self.analysis_results:
                print(f"Separation ratio: {self.analysis_results['separation_ratio']:.2f}")
            
            print(f"Results in ./svd_results/")
            
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run SVD-SCD analysis and show results"""
    print("RUNNING SVD-SCD ANALYSIS")
    print("=" * 50)
    
    # Create analyzer and run
    analyzer = SimpleSVD_SCD(r_components=50)
    success = analyzer.run_analysis()
    
    if success and analyzer.composite_products:
        print("\nRESULTS SUMMARY:")
        print("=" * 50)
        
        # Show largest families
        families = sorted(analyzer.composite_products.items(), 
                         key=lambda x: x[1]['size'], reverse=True)
        
        print(f"TOP 10 LARGEST FAMILIES:")
        for i, (family_id, family_data) in enumerate(families[:10], 1):
            print(f"{i:2d}. {family_data['family_name'][:50]}")
            print(f"     Size: {family_data['size']} products | Similarity: {family_data['avg_similarity']:.1f}%")
            
            # Show first few products
            for j, product in enumerate(family_data['products'][:2]):
                print(f"       • {product['name'][:40]}")
            if len(family_data['products']) > 2:
                print(f"       ... and {len(family_data['products']) - 2} more")
            print()
        
        # SVD info
        print(f"SVD ANALYSIS:")
        print(f"   Original dimensions: {analyzer.S.shape[0]}")
        print(f"   Reduced dimensions: {analyzer.r_components}")
        explained_var = np.sum(analyzer.svd_model.explained_variance_ratio_)
        print(f"   Explained variance: {explained_var*100:.1f}%")
        print(f"   Top 5 singular values: {np.diag(analyzer.Sigma)[:5]}")
        
        # Family statistics
        family_sizes = [fam['size'] for fam in analyzer.composite_products.values()]
        print(f"\nFAMILY STATISTICS:")
        print(f"   Total composite products: {len(analyzer.composite_products)}")
        print(f"   Average family size: {np.mean(family_sizes):.1f}")
        print(f"   Largest family: {max(family_sizes)} products")
        print(f"   Smallest family: {min(family_sizes)} products")
        
        # Show analysis results if available
        if hasattr(analyzer, 'analysis_results') and analyzer.analysis_results:
            print(f"\nQUALITY ANALYSIS:")
            print(f"   Separation ratio: {analyzer.analysis_results['separation_ratio']:.2f}")
            if analyzer.analysis_results['separation_ratio'] > 1.5:
                print("   Good family separation!")
            elif analyzer.analysis_results['separation_ratio'] > 1.2:
                print("   Moderate separation")
            else:
                print("   Poor separation")
        
        print(f"\nResults saved to: ./svd_results/")
        print("   • composite_products.json - Family data")
        print("   • svd_info.json - SVD analysis")
        print("   • U_matrix.npy, Sigma_matrix.npy, V_matrix.npy - SVD matrices")
        print("   • svd_analysis.png, family_sizes.png - Visualizations")
    else:
        print("Analysis failed or no composite products created")
    
    return analyzer

if __name__ == "__main__":
    main()