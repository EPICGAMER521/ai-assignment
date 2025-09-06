import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
from typing import List, Tuple, Dict, Optional
from scipy.sparse import csr_matrix, save_npz, load_npz

class OptimizedContentBasedRecommender:
    """
    Memory-optimized content-based recommender system for fashion items.
    
    This version reduces memory usage by:
    1. Not storing the full similarity matrix (computed on-demand)
    2. Using sparse matrices where possible
    3. Storing only essential data for recommendations
    4. Implementing efficient similarity computation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.data = None
        self.feature_matrix = None
        self.item_indices = None
        self.preprocessor = None
        self.text_vectorizer = None
        self.is_fitted = False
        
        self.categorical_features = ['gender', 'masterCategory', 'subCategory', 
                                   'articleType', 'baseColour', 'season', 'usage']
        self.numerical_features = ['year']
        self.text_features = ['productDisplayName']
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess the fashion dataset.
        
        Args:
            csv_path: Path to the CSV file containing fashion data
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading data from {csv_path}...")
        try:
            self.data = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception as e:
            print(f"Error reading CSV with pandas: {e}")
            print("Trying alternative approach...")
            import csv
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for i, row in enumerate(reader):
                    if len(row) == len(header):
                        rows.append(row)
                    else:
                        print(f"Skipping malformed line {i+2}: expected {len(header)} fields, got {len(row)}")
            
            self.data = pd.DataFrame(rows, columns=header)
        
        self.data = self.data.fillna({
            'gender': 'Unknown',
            'masterCategory': 'Unknown',
            'subCategory': 'Unknown',
            'articleType': 'Unknown',
            'baseColour': 'Unknown',
            'season': 'Unknown',
            'usage': 'Unknown',
            'productDisplayName': '',
            'year': self.data['year'].median()
        })
        
        self.item_indices = {item_id: idx for idx, item_id in enumerate(self.data['id'])}
        
        print(f"Loaded {len(self.data)} items")
        return self.data
        
    def _preprocess_features(self) -> csr_matrix:
        """
        Preprocess all features and combine them into a sparse feature matrix.
        
        Returns:
            Sparse combined feature matrix
        """
        print("Preprocessing features...")
        
        categorical_data = self.data[self.categorical_features]
        categorical_preprocessor = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        categorical_features_encoded = categorical_preprocessor.fit_transform(categorical_data)
        
        numerical_data = self.data[self.numerical_features]
        numerical_preprocessor = StandardScaler()
        numerical_features_scaled = numerical_preprocessor.fit_transform(numerical_data)
        
        text_data = self.data['productDisplayName'].fillna('')
        self.text_vectorizer = TfidfVectorizer(
            max_features=500, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        text_features_tfidf = self.text_vectorizer.fit_transform(text_data)
        
        from scipy.sparse import hstack, csr_matrix
        
        numerical_sparse = csr_matrix(numerical_features_scaled)
        
        combined_features = hstack([
            categorical_features_encoded,
            numerical_sparse,
            text_features_tfidf
        ])
        
        self.preprocessor = {
            'categorical': categorical_preprocessor,
            'numerical': numerical_preprocessor,
            'text': self.text_vectorizer
        }
        
        print(f"Feature matrix shape: {combined_features.shape}")
        print(f"Feature matrix sparsity: {1 - combined_features.nnz / (combined_features.shape[0] * combined_features.shape[1]):.4f}")
        return combined_features
    
    def fit(self, csv_path: str) -> 'OptimizedContentBasedRecommender':
        """
        Fit the recommender system on the fashion dataset.
        
        Args:
            csv_path: Path to the CSV file containing fashion data
            
        Returns:
            Self for method chaining
        """

        self.load_data(csv_path)
        
        self.feature_matrix = self._preprocess_features()
        
        self.is_fitted = True
        print("Optimized recommender system fitted successfully!")
        return self
    
    def _compute_similarities(self, item_idx: int, n_recommendations: int = 10) -> np.ndarray:
        """
        Compute similarities for a specific item on-demand.
        
        Args:
            item_idx: Index of the item to compute similarities for
            n_recommendations: Number of recommendations needed
            
        Returns:
            Array of similarity scores
        """

        target_features = self.feature_matrix[item_idx:item_idx+1]
        
        similarities = cosine_similarity(target_features, self.feature_matrix).flatten()
        
        # Apply penalty for items with identical categorical features but different IDs
        # to prevent 1.0000 scores for non-identical items
        target_item_id = self.data.iloc[item_idx]['id']
        categorical_cols = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
        target_categorical = self.data.iloc[item_idx][categorical_cols].values
        
        for i, similarity in enumerate(similarities):
            if i != item_idx and similarity > 0.99:  # Only check very high similarities
                item_id = self.data.iloc[i]['id']
                item_categorical = self.data.iloc[i][categorical_cols].values
                
                # If categorical features are identical but IDs are different, apply penalty
                if np.array_equal(target_categorical, item_categorical) and target_item_id != item_id:
                    similarities[i] = min(0.99, similarity * 0.95)  # Cap at 0.99 and apply penalty
        
        return similarities
    
    def get_recommendations(self, item_id: int, n_recommendations: int = 10, 
                          include_similarity_scores: bool = True) -> List[Dict]:
        """
        Get recommendations for a specific item.
        
        Args:
            item_id: ID of the item to get recommendations for
            n_recommendations: Number of recommendations to return
            include_similarity_scores: Whether to include similarity scores
            
        Returns:
            List of recommended items with details
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before making recommendations")
        
        if item_id not in self.item_indices:
            raise ValueError(f"Item ID {item_id} not found in dataset")
        
        item_idx = self.item_indices[item_id]
        
        similarities = self._compute_similarities(item_idx, n_recommendations + 1)
        
        # Get all indices sorted by similarity (excluding the item itself)
        all_indices = np.argsort(similarities)[::-1]
        
        # Filter out the input item itself and get top N recommendations
        similar_indices = []
        for idx in all_indices:
            if idx != item_idx:  # Exclude the input item itself
                similar_indices.append(idx)
                if len(similar_indices) >= n_recommendations:
                    break
        
        recommendations = []
        for idx in similar_indices:
            item_data = self.data.iloc[idx]
            rec = {
                'id': int(item_data['id']),
                'productDisplayName': item_data['productDisplayName'],
                'gender': item_data['gender'],
                'masterCategory': item_data['masterCategory'],
                'subCategory': item_data['subCategory'],
                'articleType': item_data['articleType'],
                'baseColour': item_data['baseColour'],
                'season': item_data['season'],
                'usage': item_data['usage'],
                'year': int(item_data['year'])
            }
            
            if include_similarity_scores:
                rec['similarity_score'] = float(similarities[idx])
            
            recommendations.append(rec)
        
        return recommendations
    
    def get_item_details(self, item_id: int) -> Dict:
        """
        Get detailed information about a specific item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Dictionary containing item details
        """
        if item_id not in self.item_indices:
            raise ValueError(f"Item ID {item_id} not found in dataset")
        
        item_data = self.data[self.data['id'] == item_id].iloc[0]
        return item_data.to_dict()
    
    def find_similar_by_features(self, gender: str = None, master_category: str = None,
                                sub_category: str = None, article_type: str = None,
                                base_colour: str = None, season: str = None,
                                usage: str = None, n_recommendations: int = 10) -> List[Dict]:
        """
        Find items similar to specified features.
        
        Args:
            Various feature filters
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of items matching the criteria
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before searching")
        
        conditions = []
        if gender:
            conditions.append(self.data['gender'] == gender)
        if master_category:
            conditions.append(self.data['masterCategory'] == master_category)
        if sub_category:
            conditions.append(self.data['subCategory'] == sub_category)
        if article_type:
            conditions.append(self.data['articleType'] == article_type)
        if base_colour:
            conditions.append(self.data['baseColour'] == base_colour)
        if season:
            conditions.append(self.data['season'] == season)
        if usage:
            conditions.append(self.data['usage'] == usage)
        
        if not conditions:
            filtered_data = self.data.sample(n=min(n_recommendations, len(self.data)))
        else:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition &= condition
            
            filtered_data = self.data[combined_condition]
        
        results = []
        for _, item in filtered_data.head(n_recommendations).iterrows():
            results.append({
                'id': int(item['id']),
                'productDisplayName': item['productDisplayName'],
                'gender': item['gender'],
                'masterCategory': item['masterCategory'],
                'subCategory': item['subCategory'],
                'articleType': item['articleType'],
                'baseColour': item['baseColour'],
                'season': item['season'],
                'usage': item['usage'],
                'year': int(item['year'])
            })
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted recommender model to disk (single file optimized version).
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted recommender")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'data': self.data,
            'item_indices': self.item_indices,
            'preprocessor': self.preprocessor,
            'text_vectorizer': self.text_vectorizer,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'text_features': self.text_features,
            'feature_matrix': self.feature_matrix 
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Optimized model saved to {filepath} (single file)")
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"Model file size: {file_size:.2f} MB")
    
    def load_model(self, filepath: str) -> 'OptimizedContentBasedRecommender':
        """
        Load a fitted recommender model from disk (single file version).
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.data = model_data['data']
        self.item_indices = model_data['item_indices']
        self.preprocessor = model_data['preprocessor']
        self.text_vectorizer = model_data['text_vectorizer']
        self.categorical_features = model_data['categorical_features']
        self.numerical_features = model_data['numerical_features']
        self.text_features = model_data['text_features']
        
        if 'feature_matrix' in model_data:
            self.feature_matrix = model_data['feature_matrix']
        else:
            feature_matrix_path = model_data['feature_matrix_path']
            if not os.path.exists(feature_matrix_path):
                base_path = filepath.replace('.pkl', '_features.npz')
                if os.path.exists(base_path):
                    feature_matrix_path = base_path
                else:
                    feature_matrix_path = filepath.replace('_optimized.pkl', '_features.npz').replace('.pkl', '_features.npz')
            self.feature_matrix = load_npz(feature_matrix_path)
        
        self.is_fitted = True
        
        print(f"Optimized model loaded from {filepath} (single file)")
        return self
    
    def get_feature_importance(self, item_id: int) -> Dict:
        """
        Analyze feature importance for a given item.
        
        Args:
            item_id: ID of the item to analyze
            
        Returns:
            Dictionary containing feature importance analysis
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before analyzing features")
        
        if item_id not in self.item_indices:
            raise ValueError(f"Item {item_id} not found in dataset")
        
        item_idx = self.item_indices[item_id]
        item_features = self.feature_matrix[item_idx].toarray().flatten()       

        feature_names = []
        
        if hasattr(self.preprocessor['categorical'], 'get_feature_names_out'):
            cat_features = self.preprocessor['categorical'].get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_features)
        else:
            for feature in self.categorical_features:
                if feature in self.data.columns:
                    unique_values = self.data[feature].unique()
                    feature_names.extend([f"{feature}_{val}" for val in unique_values])
        
        feature_names.extend(self.numerical_features)
        
        if hasattr(self.text_vectorizer, 'get_feature_names_out'):
            text_features = self.text_vectorizer.get_feature_names_out()
            feature_names.extend(text_features)
        
        if len(feature_names) != len(item_features):
            feature_names = [f"feature_{i}" for i in range(len(item_features))]
        
        non_zero_indices = np.nonzero(item_features)[0]
        important_features = {
            feature_names[i]: float(item_features[i]) 
            for i in non_zero_indices
        }
        
        sorted_features = dict(sorted(important_features.items(), 
                                    key=lambda x: abs(x[1]), reverse=True))
        
        item_details = self.get_item_details(item_id)
        
        return {
            'item_id': item_id,
            'item_details': item_details,
            'total_features': len(item_features),
            'active_features': len(non_zero_indices),
            'sparsity': 1 - (len(non_zero_indices) / len(item_features)),
            'top_features': dict(list(sorted_features.items())[:20]),  # Top 20 features
            'feature_distribution': {
                'categorical': len([f for f in sorted_features.keys() if any(cat in f for cat in self.categorical_features)]),
                'numerical': len([f for f in sorted_features.keys() if f in self.numerical_features]),
                'text': len([f for f in sorted_features.keys() if f not in self.categorical_features and f not in self.numerical_features])
            }
        }
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "total_items": len(self.data),
            "unique_genders": self.data['gender'].nunique(),
            "unique_categories": self.data['masterCategory'].nunique(),
            "unique_subcategories": self.data['subCategory'].nunique(),
            "unique_article_types": self.data['articleType'].nunique(),
            "unique_colors": self.data['baseColour'].nunique(),
            "year_range": f"{self.data['year'].min()}-{self.data['year'].max()}",
            "feature_matrix_shape": self.feature_matrix.shape if self.feature_matrix is not None else None,
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024 / 1024
        }

def test_optimized_recommender(csv_path: str):
    """
    Test the optimized recommender system.
    
    Args:
        csv_path: Path to the CSV file
    """
    print("Testing Optimized Content-Based Recommender")
    print("=" * 50)
    
    recommender = OptimizedContentBasedRecommender()
    recommender.fit(csv_path)
    
    stats = recommender.get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    sample_item_id = recommender.data['id'].iloc[0]
    print(f"\nGetting recommendations for item {sample_item_id}:")
    
    recommendations = recommender.get_recommendations(sample_item_id, n_recommendations=5)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['productDisplayName']} (ID: {rec['id']}, Similarity: {rec.get('similarity_score', 'N/A'):.3f})")
    
    model_path = "models/optimized_content_based_model.pkl"
    print(f"\nSaving model to {model_path}...")
    recommender.save_model(model_path)
    
    print("Loading model...")
    new_recommender = OptimizedContentBasedRecommender()
    new_recommender.load_model(model_path)
    
    test_recommendations = new_recommender.get_recommendations(sample_item_id, n_recommendations=3)
    print(f"\nTest recommendations from loaded model:")
    for i, rec in enumerate(test_recommendations, 1):
        print(f"  {i}. {rec['productDisplayName']} (ID: {rec['id']})")

if __name__ == "__main__":
    csv_path = "data/styles.csv"
    if os.path.exists(csv_path):
        test_optimized_recommender(csv_path)
    else:
        print(f"Dataset not found at {csv_path}")
        print("Please ensure the styles.csv file is in the data/ directory")