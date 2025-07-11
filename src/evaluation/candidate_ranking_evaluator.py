#!/usr/bin/env python3
"""
CORRECTED BERT4Rec Evaluation - Debug and Fix Issues
Fixes the bugs causing 100% frequency baseline and improves methodology
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
import os
import time
import random

class DebugCandidateRankingEvaluator:
    """
    CORRECTED evaluation with proper debugging and validation
    Fixes the impossible 100% frequency baseline
    """
    
    def __init__(self, n_candidates=100, n_negatives=99, debug=True):
        self.original_embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.user_data = None
        self.n_candidates = n_candidates
        self.n_negatives = n_negatives
        self.debug = debug
        
    def load_data(self):
        """Load embeddings and user data with proper validation"""
        print("Loading data with DEBUGGING enabled...")
        
        # Load embeddings
        embedding_paths = [
            './bert4rec_model_v2/item_embeddings_test.npy',
            './item_embeddings_test.npy'
        ]
        
        for path in embedding_paths:
            if os.path.exists(path):
                embeddings = np.load(path)
                self.original_embeddings = embeddings[2:]  # Skip special tokens
                print(f"Loaded embeddings: {self.original_embeddings.shape}")
                break
        
        if self.original_embeddings is None:
            print("❌ Could not find embeddings file!")
            return False
        
        # Create 10D reduced embeddings
        print("Creating 10D embeddings...")
        embeddings_centered = self.original_embeddings - np.mean(self.original_embeddings, axis=0)
        svd = TruncatedSVD(n_components=10, random_state=42)
        self.reduced_embeddings = svd.fit_transform(embeddings_centered)
        explained_var = np.sum(svd.explained_variance_ratio_)
        print(f"10D embeddings: {self.reduced_embeddings.shape}, explained variance: {explained_var:.3f}")
        
        # Create clusters
        kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, max_iter=300, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.reduced_embeddings)
        
        # Load and parse user data with STRICT validation
        df = pd.read_csv('./data.csv')
        self.user_data = []
        
        print("Parsing user sequences with STRICT validation...")
        parsing_stats = {
            'total_rows': len(df),
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'data_leakage_detected': 0,
            'test_cases_created': 0
        }
        
        for idx, row in df.head(1000).iterrows():
            try:
                # Parse sequence
                sequence_str = row['sequence_idx'].strip('[]')
                baskets = []
                for basket_str in sequence_str.split('], ['):
                    basket_str = basket_str.strip('[]')
                    if basket_str:
                        products = [int(x.strip()) for x in basket_str.split(',') if x.strip().isdigit()]
                        if products:
                            baskets.append(products)
                
                # Parse target
                target_str = row['last_basket_idx'].strip('[]')
                target_products = [int(x.strip()) for x in target_str.split(',') if x.strip().isdigit()]
                
                if len(baskets) >= 2 and target_products:
                    parsing_stats['valid_sequences'] += 1
                    
                    # Create train/test split: use first 80% of baskets as history
                    split_point = max(1, int(len(baskets) * 0.8))
                    history_baskets = baskets[:split_point]
                    test_baskets = baskets[split_point:]
                    
                    # Flatten history
                    all_history = [item for basket in history_baskets for item in basket 
                                 if 0 <= item < len(self.original_embeddings)]
                    
                    # Use remaining baskets + final target as test items
                    test_items = []
                    for basket in test_baskets:
                        test_items.extend(basket)
                    test_items.extend(target_products)
                    
                    # Filter valid test items
                    valid_test_items = [item for item in test_items if 0 <= item < len(self.original_embeddings)]
                    
                    if all_history and valid_test_items:
                        unique_history = list(set(all_history))
                        
                        # Create test cases with STRICT validation
                        for test_item in set(valid_test_items):
                            # CRITICAL: Ensure NO data leakage
                            if test_item in unique_history:
                                parsing_stats['data_leakage_detected'] += 1
                                continue  # Skip this test case
                            
                            # Create valid test case
                            user_data = {
                                'user_id': row['user_id'],
                                'history': unique_history,
                                'positive_item': test_item,
                                'history_size': len(unique_history),
                                'row_idx': idx
                            }
                            self.user_data.append(user_data)
                            parsing_stats['test_cases_created'] += 1
                else:
                    parsing_stats['invalid_sequences'] += 1
                        
            except Exception as e:
                parsing_stats['invalid_sequences'] += 1
                if self.debug:
                    print(f"Parse error row {idx}: {e}")
        
        # Print detailed statistics
        print(f"\n📊 PARSING STATISTICS:")
        print(f"   Total rows processed: {parsing_stats['total_rows']}")
        print(f"   Valid sequences: {parsing_stats['valid_sequences']}")
        print(f"   Invalid sequences: {parsing_stats['invalid_sequences']}")
        print(f"   Data leakage cases detected: {parsing_stats['data_leakage_detected']}")
        print(f"   Final test cases created: {parsing_stats['test_cases_created']}")
        
        if parsing_stats['data_leakage_detected'] > 0:
            print(f"   ⚠️  WARNING: {parsing_stats['data_leakage_detected']} data leakage cases prevented!")
        
        # Analyze dataset
        if self.user_data:
            history_lengths = [len(user['history']) for user in self.user_data]
            print(f"\n📈 DATASET ANALYSIS:")
            print(f"   Test cases: {len(self.user_data)}")
            print(f"   Avg history length: {np.mean(history_lengths):.1f}")
            print(f"   Min history length: {np.min(history_lengths)}")
            print(f"   Max history length: {np.max(history_lengths)}")
            
            # Validation check
            if self.debug:
                print(f"\n🔍 VALIDATION CHECK:")
                leakage_count = 0
                for user in self.user_data[:100]:  # Check first 100
                    if user['positive_item'] in user['history']:
                        leakage_count += 1
                
                if leakage_count > 0:
                    print(f"   ❌ DATA LEAKAGE DETECTED: {leakage_count}/100 test cases have positive items in history!")
                    return False
                else:
                    print(f"   ✅ No data leakage detected in sample")
            
            return True
        else:
            print("❌ No valid test cases created!")
            return False
    
    def generate_negative_samples(self, user_history, positive_item, n_negatives):
        """Generate negative samples with validation"""
        all_items = set(range(len(self.original_embeddings)))
        history_set = set(user_history)
        excluded_items = history_set | {positive_item}
        candidate_negatives = list(all_items - excluded_items)
        
        if len(candidate_negatives) < n_negatives:
            print(f"Warning: Only {len(candidate_negatives)} negative candidates available, need {n_negatives}")
            return candidate_negatives
        
        negatives = random.sample(candidate_negatives, n_negatives)
        
        # Validation: ensure no overlap
        if self.debug:
            overlap_with_history = set(negatives) & history_set
            if overlap_with_history:
                print(f"ERROR: Negative samples overlap with history: {overlap_with_history}")
            
            if positive_item in negatives:
                print(f"ERROR: Positive item {positive_item} in negatives!")
        
        return negatives
    
    def rank_candidates_frequency_corrected(self, user_history, candidates):
        """CORRECTED frequency baseline - should NOT get 100% Hit@10"""
        # Count frequency of each item in history
        item_counts = Counter(user_history)
        
        # Score candidates - items NOT in history get score 0
        candidate_scores = []
        for item in candidates:
            score = item_counts.get(item, 0)  # 0 if not in history
            candidate_scores.append((item, score))
        
        # Sort by score (descending), then by item ID for consistency
        candidate_scores.sort(key=lambda x: (-x[1], x[0]))
        ranked_items = [item for item, score in candidate_scores]
        
        # DEBUG: Check if positive item gets non-zero score
        if self.debug and len(candidates) > 0:
            positive_item = candidates[0]  # First candidate should be positive
            positive_score = item_counts.get(positive_item, 0)
            if positive_score > 0:
                print(f"WARNING: Positive item {positive_item} has frequency {positive_score} in history!")
        
        return ranked_items
    
    def rank_candidates_random_baseline(self, user_history, candidates):
        """Random baseline - should get ~10% Hit@10"""
        random.seed(42)  # For reproducibility
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        return shuffled
    
    def rank_candidates_embedding_128d(self, user_history, candidates):
        """Rank candidates using 128D embeddings"""
        if not user_history:
            return candidates
        
        # Create user profile
        user_profile = np.mean(self.original_embeddings[user_history], axis=0)
        
        # Calculate similarities
        candidate_embeddings = self.original_embeddings[candidates]
        similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        # Sort by similarity (descending)
        candidate_scores = list(zip(candidates, similarities))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_items = [item for item, score in candidate_scores]
        
        return ranked_items
    
    def rank_candidates_embedding_10d(self, user_history, candidates):
        """Rank candidates using 10D embeddings"""
        if not user_history:
            return candidates
        
        # Create user profile
        user_profile = np.mean(self.reduced_embeddings[user_history], axis=0)
        
        # Calculate similarities
        candidate_embeddings = self.reduced_embeddings[candidates]
        similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        # Sort by similarity (descending)
        candidate_scores = list(zip(candidates, similarities))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_items = [item for item, score in candidate_scores]
        
        return ranked_items
    
    def evaluate_method(self, ranking_func, method_name, k=10):
        """Evaluate with detailed debugging"""
        print(f"\nEvaluating: {method_name}")
        print("-" * 50)
        
        hits = 0
        total_tests = 0
        
        # Debug statistics
        debug_stats = {
            'positive_in_top_1': 0,
            'positive_in_top_5': 0,
            'positive_ranks': [],
            'negative_sampling_errors': 0
        }
        
        start_time = time.time()
        random.seed(42)  # Reproducible results
        
        for i, user in enumerate(self.user_data):
            try:
                # Generate negative samples
                negatives = self.generate_negative_samples(
                    user['history'], 
                    user['positive_item'], 
                    self.n_negatives
                )
                
                if len(negatives) < self.n_negatives:
                    debug_stats['negative_sampling_errors'] += 1
                    continue
                
                # Create candidate set: positive + negatives
                candidates = [user['positive_item']] + negatives
                
                # Rank candidates
                ranked_candidates = ranking_func(user['history'], candidates)
                
                # Find rank of positive item
                positive_item = user['positive_item']
                try:
                    positive_rank = ranked_candidates.index(positive_item) + 1  # 1-indexed
                    debug_stats['positive_ranks'].append(positive_rank)
                    
                    # Count hits
                    if positive_rank <= k:
                        hits += 1
                    if positive_rank <= 1:
                        debug_stats['positive_in_top_1'] += 1
                    if positive_rank <= 5:
                        debug_stats['positive_in_top_5'] += 1
                        
                except ValueError:
                    print(f"ERROR: Positive item {positive_item} not found in ranked candidates!")
                    continue
                
                total_tests += 1
                
                # Progress update
                if (i + 1) % 500 == 0:
                    current_hit_rate = hits / total_tests if total_tests > 0 else 0
                    print(f"   Progress: {i+1}/{len(self.user_data)} ({current_hit_rate:.1%} Hit@{k})")
                    
            except Exception as e:
                if self.debug:
                    print(f"Evaluation error: {e}")
                continue
        
        eval_time = time.time() - start_time
        
        # Calculate final metrics
        hit_rate = hits / total_tests if total_tests > 0 else 0
        
        print(f"Final Hit@{k}: {hits}/{total_tests} = {hit_rate:.4f} ({hit_rate*100:.2f}%)")
        print(f"Evaluation time: {eval_time:.2f} seconds")
        
        # Debug information
        if self.debug and debug_stats['positive_ranks']:
            avg_rank = np.mean(debug_stats['positive_ranks'])
            median_rank = np.median(debug_stats['positive_ranks'])
            print(f"   📊 Debug Stats:")
            print(f"      Avg positive rank: {avg_rank:.1f}")
            print(f"      Median positive rank: {median_rank:.1f}")
            print(f"      Hit@1: {debug_stats['positive_in_top_1']}/{total_tests} ({debug_stats['positive_in_top_1']/total_tests*100:.1f}%)")
            print(f"      Hit@5: {debug_stats['positive_in_top_5']}/{total_tests} ({debug_stats['positive_in_top_5']/total_tests*100:.1f}%)")
            if debug_stats['negative_sampling_errors'] > 0:
                print(f"      ⚠️  Negative sampling errors: {debug_stats['negative_sampling_errors']}")
        
        return {
            'hit_rate': hit_rate,
            'hit_rate_percent': hit_rate * 100,
            'eval_time': eval_time,
            'total_hits': hits,
            'total_tests': total_tests,
            'avg_rank': np.mean(debug_stats['positive_ranks']) if debug_stats['positive_ranks'] else 0,
            'debug_stats': debug_stats
        }
    
    def run_comparison(self):
        """Run corrected comparison with proper baselines"""
        print("🎯 CORRECTED CANDIDATE RANKING EVALUATION")
        print("="*60)
        print(f"Testing with {self.n_candidates} candidates ({self.n_negatives} negatives + 1 positive)")
        print("Fixed bugs and added proper validation")
        
        methods = {
            'Random Baseline': self.rank_candidates_random_baseline,
            'Frequency Baseline (Corrected)': self.rank_candidates_frequency_corrected,
            'Original 128D Embeddings': self.rank_candidates_embedding_128d,
            'Reduced 10D Embeddings': self.rank_candidates_embedding_10d,
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            result = self.evaluate_method(method_func, method_name)
            results[method_name] = result
        
        # Analysis
        print("\n" + "="*60)
        print("CORRECTED RESULTS ANALYSIS")
        print("="*60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['hit_rate'], reverse=True)
        
        print("PERFORMANCE RANKING:")
        for i, (method, metrics) in enumerate(sorted_results, 1):
            hit_percent = metrics['hit_rate_percent']
            avg_rank = metrics.get('avg_rank', 0)
            
            print(f"{i}. {method}: {hit_percent:.2f}% Hit@10 (avg rank: {avg_rank:.1f})")
            
            # Validate results
            if 'Random' in method:
                expected_random = 10.0  # 10% for random
                if abs(hit_percent - expected_random) > 3:
                    print(f"   ⚠️  Random baseline unusual: expected ~{expected_random}%, got {hit_percent:.1f}%")
                else:
                    print(f"   ✅ Random baseline looks correct")
            
            elif 'Frequency' in method:
                if hit_percent > 50:
                    print(f"   ⚠️  Frequency baseline too high - possible data leakage")
                elif hit_percent < 5:
                    print(f"   ✅ Frequency baseline looks correct (low due to no overlap)")
                else:
                    print(f"   📊 Frequency baseline reasonable")
            
            elif 'Embedding' in method:
                if hit_percent >= 50:
                    print(f"   ✅ Good embedding performance")
                elif hit_percent >= 30:
                    print(f"   📊 Moderate embedding performance")
                else:
                    print(f"   ⚠️  Low embedding performance")
        
        # Best embedding method
        embedding_results = {k: v for k, v in results.items() if 'Embedding' in k}
        if embedding_results:
            best_embedding = max(embedding_results.items(), key=lambda x: x[1]['hit_rate'])
            print(f"\n🏆 BEST EMBEDDING METHOD: {best_embedding[0]}")
            print(f"Hit@10: {best_embedding[1]['hit_rate_percent']:.2f}%")
        
        # 10D compression analysis
        original_perf = results.get('Original 128D Embeddings', {}).get('hit_rate_percent', 0)
        reduced_perf = results.get('Reduced 10D Embeddings', {}).get('hit_rate_percent', 0)
        
        if original_perf > 0 and reduced_perf > 0:
            print(f"\n📊 10D COMPRESSION ANALYSIS:")
            print(f"Original 128D: {original_perf:.2f}% Hit@10")
            print(f"Reduced 10D: {reduced_perf:.2f}% Hit@10")
            retention = (reduced_perf / original_perf) * 100
            print(f"Performance retention: {retention:.1f}%")
        
        # Overall assessment
        max_embedding_perf = max([v['hit_rate_percent'] for k, v in results.items() if 'Embedding' in k], default=0)
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if max_embedding_perf >= 50:
            print(f"   ✅ Excellent: {max_embedding_perf:.1f}% Hit@10")
            print(f"   🎉 Embeddings are working well!")
        elif max_embedding_perf >= 30:
            print(f"   📊 Good: {max_embedding_perf:.1f}% Hit@10")
            print(f"   💡 Room for improvement but functional")
        else:
            print(f"   ⚠️  Needs improvement: {max_embedding_perf:.1f}% Hit@10")
            print(f"   🔧 Consider: different model, better training, or different embeddings")
        
        return results

def main():
    evaluator = DebugCandidateRankingEvaluator(debug=True)
    
    if not evaluator.load_data():
        print("❌ Failed to load data properly")
        return
    
    results = evaluator.run_comparison()
    
    print(f"\n🎉 CORRECTED EVALUATION COMPLETE!")
    print(f"Fixed bugs and added proper validation")

if __name__ == "__main__":
    main()