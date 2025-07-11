# BERT4Rec Project - Directory Structure & File Documentation

## 📁 Project Overview
This is a BERT4Rec (BERT for Recommendation) implementation focused on next basket prediction for grocery shopping data. The project includes data preparation, model training, evaluation, and advanced analysis components.

---

## 📂 Directory Structure

```
BERT4Rec/
├── 📁 .git/                                    # Git version control
├── 📄 .gitignore                              # Git ignore patterns
├── 📁 bert4rec_model_v2/                      # 🎯 Main Model Outputs & Results
├── 📁 recommendation_evaluation/               # 📊 Evaluation Results
├── 📁 src/                                    # 💻 Source Code
├── 📁 svd_results/                            # 🔍 SVD Analysis Results
└── 📄 README_STRUCTURE.md                     # 📋 This documentation
```

---

## 🎯 bert4rec_model_v2/ - Main Model Outputs

**Purpose**: Contains the trained BERT4Rec model files, embeddings, and comprehensive analysis results.

### Core Model Files
- **`best_model.pt`** (PyTorch Model)
  - Best performing BERT4Rec model checkpoint
  - Optimized for next basket prediction
  - Size: Large binary file

- **`final_model.pt`** (PyTorch Model)
  - Final trained model state
  - Last epoch checkpoint
  - Size: Large binary file

### Embeddings & Data
- **`item_embeddings.npy`** (NumPy Array)
  - Learned item embeddings from BERT4Rec training
  - Shape: [num_items, embedding_dim] - likely [49468, 128]
  - Used for similarity calculations and clustering

- **`item_embeddings_test.npy`** (NumPy Array)
  - Test set item embeddings
  - For evaluation and analysis purposes

### Analysis & Visualizations
- **`basket_size_distribution.png`** (Image)
  - Distribution chart of basket sizes in the dataset
  - Shows shopping behavior patterns

- **`prediction_distribution.png`** (Image)
  - Distribution of model prediction scores
  - Model confidence analysis

- **`test_results.json`** (JSON)
  - Comprehensive test metrics and model configuration
  - Contains: Hit rates, NDCG, MRR, model hyperparameters
  - Example metrics: HR@5, HR@10, NDCG@5, NDCG@10

### 📁 embedding_analysis_comprehensive/
**Purpose**: Advanced clustering and embedding analysis

- **`best_clustering_detailed.png`** - Detailed visualization of optimal clustering
- **`best_clustering_visualization.png`** - Best clustering method visualization  
- **`clustering_comparison_comprehensive.png`** - Comparison of different clustering approaches
- **`comprehensive_final_report.json`** - Complete analysis report with metrics
- **`comprehensive_metrics_summary.png`** - Summary visualization of all metrics
- **`memory_optimized_analysis_report.json`** - Optimized analysis results

---

## 📊 recommendation_evaluation/ - Evaluation Results

**Purpose**: Contains evaluation configurations and results for recommendation performance.

- **`configuration.json`** (JSON Configuration)
  - SVD dimensions: 10
  - K-means clusters: 5
  - Explained variance: ~9.18%
  - Test sequences: 131,208
  - Embedding shapes: Original [49468, 128] → Reduced [49468, 10]

- **`evaluation_results.json`** (JSON Results)
  - Recommendation evaluation metrics
  - Performance comparison between different methods
  - Ranking and recommendation quality metrics

---

## 💻 src/ - Source Code

**Purpose**: Core implementation modules for data processing, modeling, and analysis.

### 📁 data_preparation/
- **`bert4rec_data_preparation.py`** (640 lines)
  - **Class**: `BERT4RecDataPreparation`
  - **Purpose**: Converts raw grocery data into BERT4Rec training format
  - **Features**:
    - Sequence creation from user shopping history
    - Masking for BERT pre-training (15% mask probability)
    - Train/validation splitting
    - Basket size handling and optimization
  - **Key Parameters**: `max_seq_length=50`, `max_basket_size=auto`

### 📁 data_processing/
- **Status**: Empty directory (cleanup completed)

### 📁 evaluation/
- **`candidate_ranking_evaluator.py`** (484 lines)
  - **Class**: `DebugCandidateRankingEvaluator`
  - **Purpose**: Evaluates recommendation quality with candidate ranking
  - **Features**:
    - SVD dimensionality reduction
    - Clustering-based evaluation
    - Debug and validation features
    - Fixes for baseline evaluation issues
  - **Metrics**: Hit rate, NDCG, ranking quality

### 📁 model/
- **Status**: Empty directory (main model implementation likely in experiments)

### 📁 product_grouping/
Contains algorithms for product similarity and grouping:

- **`fuzzy_product_grouper.py`** (702 lines)
  - **Purpose**: Groups similar products using fuzzy string matching
  - **Features**:
    - Fuzzy string similarity (fuzzywuzzy or basic fallback)
    - Product name standardization
    - Family grouping creation
    - JSON export of product families
  - **Dependencies**: Optional fuzzywuzzy for better results

- **`similarity_embedding_creator.py`** (555 lines)
  - **Class**: `SimilarityEmbeddingCreator`
  - **Purpose**: Creates embeddings specifically for product similarity
  - **Features**:
    - TF-IDF vectorization of product descriptions
    - Cosine similarity calculations
    - Integration with BERT4Rec embeddings
    - Separate similarity-focused embeddings

### 📁 svd_scd/
- **`svd_scd_analyzer.py`** (462 lines)
  - **Class**: `SimpleSVD_SCD`
  - **Purpose**: Implements SVD-SCD (Singular Value Decomposition - Sparse Component Decomposition)
  - **Algorithm**: S = UΣV^T decomposition with dimension reduction
  - **Features**:
    - Matrix factorization for product relationships
    - Composite product creation
    - Dimension reduction (default 50 components)
    - Component mapping: v_j = Σ^(-1)U^T s_j

### 📁 training/
- **Status**: Empty directory (training logic likely in experiments)

### 📁 utils/
- **Status**: Empty directory (utilities distributed across modules)

---

## 🔍 svd_results/ - SVD Analysis Results

**Purpose**: Results from SVD-SCD analysis of product relationships and patterns.

### Analysis Data
- **`svd_info.json`** (JSON)
  - SVD configuration: 50 components
  - Singular values array (70 total values)
  - Mathematical decomposition parameters

- **`composite_products.json`** (JSON)
  - Composite product definitions from fuzzy families
  - Product groupings and relationships

### Matrix Files
- **`U_matrix.npy`** (NumPy Array)
  - Left singular vectors matrix (m×r)
  - Product feature representations

- **`V_matrix.npy`** (NumPy Array)  
  - Right singular vectors matrix (n×r)
  - Component relationships

- **`Sigma_matrix.npy`** (NumPy Array)
  - Singular values diagonal matrix (r×r)
  - Importance weights for components

### Visualizations
- **`svd_analysis.png`** - SVD decomposition analysis chart
- **`composite_products_2d.png`** - 2D visualization of composite products
- **`family_sizes.png`** - Distribution of product family sizes

---

## 🔧 Technical Specifications

### Model Architecture
- **Type**: BERT4Rec (Transformer-based)
- **Hidden Size**: 128
- **Layers**: 2 hidden layers
- **Attention Heads**: 4
- **Vocabulary Size**: 49,470 products
- **Max Sequence Length**: 15 baskets
- **Max Basket Size**: 8 items

### Data Characteristics
- **Test Sequences**: 131,208
- **Products**: 49,468 unique items
- **Embedding Dimension**: 128 → 10 (after SVD reduction)
- **Explained Variance**: 9.18% (with 10 components)

### Key Technologies
- **PyTorch**: Deep learning framework
- **Transformers**: BERT architecture
- **Scikit-learn**: SVD, clustering, metrics
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Visualizations

---

## 🚀 Usage Workflow

1. **Data Preparation**: Use `src/data_preparation/bert4rec_data_preparation.py`
2. **Model Training**: Load trained models from `bert4rec_model_v2/`
3. **Product Grouping**: Apply `src/product_grouping/` for similarity analysis
4. **SVD Analysis**: Use `src/svd_scd/` for dimensional reduction
5. **Evaluation**: Apply `src/evaluation/` for performance metrics
6. **Results**: Check `recommendation_evaluation/` and analysis outputs

---

## 📈 Performance & Results

- **Model**: BERT4Rec optimized for grocery basket prediction
- **Clustering**: 5 optimal clusters with K-means
- **Dimensionality**: 128D → 10D reduction while maintaining performance
- **Evaluation**: Comprehensive ranking metrics and baseline comparisons
- **Analysis**: Advanced embedding clustering and product relationship mapping
