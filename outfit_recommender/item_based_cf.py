import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ItemBasedCollaborativeFiltering:
    def __init__(self):
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.item_means = None
        self.items_data = None
        
    def load_data(self, ratings_path, styles_path, sample_size=50000):
        """
        Load and preprocess the ratings and styles data
        """
        print("Loading data...")
        
        # Load ratings data
        self.ratings_df = pd.read_csv(ratings_path)
        print(f"Loaded {len(self.ratings_df)} ratings")
        
        # Sample data if too large to avoid memory issues
        if len(self.ratings_df) > sample_size:
            print(f"Sampling {sample_size} ratings for training...")
            self.ratings_df = self.ratings_df.sample(n=sample_size, random_state=42)
        
        # Load styles data
        self.styles_df = pd.read_csv(styles_path)
        print(f"Loaded {len(self.styles_df)} items")
        
        # Get unique users and items from sampled data
        unique_users = self.ratings_df['user_id'].unique()
        unique_items = self.ratings_df['product_id'].unique()
        
        print(f"Working with {len(unique_users)} users and {len(unique_items)} items")
        
        # Create user-item matrix with sampled data
        try:
            self.user_item_matrix = self.ratings_df.pivot_table(
                index='user_id', 
                columns='product_id', 
                values='rating', 
                fill_value=0
            )
            print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            # Fallback: create a smaller sample
            print("Using smaller sample...")
            small_sample = self.ratings_df.sample(n=min(10000, len(self.ratings_df)), random_state=42)
            self.user_item_matrix = small_sample.pivot_table(
                index='user_id', 
                columns='product_id', 
                values='rating', 
                fill_value=0
            )
            print(f"Reduced user-item matrix shape: {self.user_item_matrix.shape}")
        
        # Store items data for recommendations
        self.items_data = self.styles_df.set_index('id')
        
        return self
    
    def calculate_pearson_correlation(self, item1_ratings, item2_ratings):
        """
        Calculate proper Pearson correlation between two items based on actual ratings.
        Zero ratings are treated as missing values, not actual ratings.
        """
        # Find users who actually rated both items (non-zero ratings)
        common_users = (item1_ratings > 0) & (item2_ratings > 0)
        common_count = common_users.sum()
        
        # Need at least 2 common ratings to calculate correlation
        if common_count < 2:
            return 0.0  # No correlation can be calculated
        
        # Get the actual ratings for users who rated both items
        item1_common = item1_ratings[common_users]
        item2_common = item2_ratings[common_users]
        
        # Calculate Pearson correlation coefficient
        try:
            correlation, _ = pearsonr(item1_common, item2_common)
            
            # Handle NaN cases (e.g., when all ratings are identical)
            if np.isnan(correlation):
                # If ratings are identical, return high but not perfect correlation
                # to avoid perfect correlations between different items
                if np.array_equal(item1_common.values, item2_common.values):
                    return 0.95  # High correlation for identical rating patterns, but not perfect
                else:
                    return 0.0
            
            # Cap correlation to prevent perfect 1.0 correlations between different items
            # Maximum allowed correlation is 0.99 to ensure no perfect correlations
            return min(correlation, 0.99)
            
        except Exception as e:
            # In case of any calculation error, return 0
            return 0.0
    
    def calculate_metadata_similarity(self, item1_id, item2_id):
        """
        Calculate similarity based on item metadata (category, subcategory, article type, etc.)
        """
        try:
            if not hasattr(self, 'items_data') or self.items_data is None:
                return 0.1  # Default low similarity
            
            # Get item information
            if item1_id not in self.items_data.index or item2_id not in self.items_data.index:
                return 0.1
            
            item1_info = self.items_data.loc[item1_id]
            item2_info = self.items_data.loc[item2_id]
            
            similarity_score = 0.0
            
            # Category match (highest weight)
            if item1_info.get('masterCategory') == item2_info.get('masterCategory'):
                similarity_score += 0.4
            
            # Subcategory match
            if item1_info.get('subCategory') == item2_info.get('subCategory'):
                similarity_score += 0.3
            
            # Article type match
            if item1_info.get('articleType') == item2_info.get('articleType'):
                similarity_score += 0.2
            
            # Gender match
            if item1_info.get('gender') == item2_info.get('gender'):
                similarity_score += 0.1
            
            # Season match
            if item1_info.get('season') == item2_info.get('season'):
                similarity_score += 0.05
            
            # Usage match
            if item1_info.get('usage') == item2_info.get('usage'):
                similarity_score += 0.05
            
            # Color similarity (basic)
            if item1_info.get('baseColour') == item2_info.get('baseColour'):
                similarity_score += 0.1
            
            # Cap the similarity to reasonable range for metadata-only similarity
            return min(similarity_score, 0.7)  # Max 0.7 for metadata similarity
            
        except Exception as e:
            return 0.1  # Default low similarity on error
    
    def calculate_fallback_similarity(self, item1_ratings, item2_ratings):
        """
        Calculate similarity when no common users exist using item characteristics
        """
        # Use the new metadata similarity method
        return self.calculate_metadata_similarity(item1_ratings.name, item2_ratings.name)
    
    def calculate_adjusted_correlation(self, item1_ratings, item2_ratings, common_users):
        """
        Calculate correlation with adjustments for very sparse data
        """
        item1_common = item1_ratings[common_users]
        item2_common = item2_ratings[common_users]
        
        # For very few data points, use a more conservative approach
        if len(item1_common) == 1:
            # Single common user - check if ratings are similar
            diff = abs(item1_common.iloc[0] - item2_common.iloc[0])
            return max(0, (5 - diff) / 5.0) * 0.5  # Max 0.5 for single user
        
        elif len(item1_common) == 2:
            # Two common users - simple correlation with penalty
            try:
                correlation, _ = pearsonr(item1_common, item2_common)
                if np.isnan(correlation):
                    return self.calculate_fallback_similarity(item1_ratings, item2_ratings)
                return correlation * 0.7  # Reduce confidence for only 2 users
            except:
                return self.calculate_fallback_similarity(item1_ratings, item2_ratings)
        
        # Should not reach here, but fallback
        return self.calculate_fallback_similarity(item1_ratings, item2_ratings)
    
    def build_item_similarity_matrix(self, max_items=5000):
        """
        Build item-item similarity matrix using Pearson correlation
        Increased to top 5000 items for better accuracy
        """
        print("Building item similarity matrix...")
        
        items = self.user_item_matrix.columns
        n_items = len(items)
        
        # Limit number of items to 5000 for better accuracy while maintaining efficiency
        if n_items > max_items:
            print(f"Limiting items from {n_items} to {max_items} for memory efficiency")
            item_counts = self.user_item_matrix.astype(bool).sum(axis=0)
            top_items = item_counts.nlargest(max_items).index
            self.user_item_matrix = self.user_item_matrix[top_items]
            items = top_items
            n_items = len(items)
        
        print(f"Processing {n_items} items for correlation calculation")
        
        # Initialize similarity matrix
        self.item_similarity_matrix = pd.DataFrame(
            np.zeros((n_items, n_items)), 
            index=items, 
            columns=items
        )
        
        # Calculate item means for each item
        self.item_means = {}
        for item in items:
            item_ratings = self.user_item_matrix[item]
            rated_users = item_ratings > 0
            if rated_users.sum() > 0:
                self.item_means[item] = item_ratings[rated_users].mean()
            else:
                self.item_means[item] = 3.0  # Default rating
        
        # Calculate pairwise similarities
        for i, item1 in enumerate(items):
            if i % 50 == 0:
                print(f"Processing item {i+1}/{n_items}")
            
            for j, item2 in enumerate(items):
                if i == j:
                    # Self-correlation should be 1.0 (perfect correlation with itself)
                    self.item_similarity_matrix.loc[item1, item2] = 1.0
                elif i < j:  # Calculate only upper triangle to avoid redundant calculations
                    item1_ratings = self.user_item_matrix[item1]
                    item2_ratings = self.user_item_matrix[item2]
                    
                    similarity = self.calculate_pearson_correlation(item1_ratings, item2_ratings)
                    
                    # Store in both positions (symmetric matrix)
                    self.item_similarity_matrix.loc[item1, item2] = similarity
                    self.item_similarity_matrix.loc[item2, item1] = similarity
        
        print("Item similarity matrix completed!")
        return self
    
    def predict_rating(self, user_id, item_id, k=10):
        """
        Predict rating for a user-item pair using item-based CF
        """
        if user_id not in self.user_item_matrix.index:
            return self.item_means.get(item_id, 3.0)  # Default rating
        
        if item_id not in self.user_item_matrix.columns:
            return 3.0  # Default rating
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get items rated by the user
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return self.item_means.get(item_id, 3.0)
        
        # Get similarities between target item and rated items
        similarities = []
        weighted_ratings = []
        
        for rated_item in rated_items:
            if rated_item in self.item_similarity_matrix.index:
                sim = self.item_similarity_matrix.loc[item_id, rated_item]
                if sim > 0:  # Only consider positive correlations
                    similarities.append(sim)
                    weighted_ratings.append(sim * user_ratings[rated_item])
        
        if len(similarities) == 0:
            return self.item_means.get(item_id, 3.0)
        
        # Select top-k most similar items
        if len(similarities) > k:
            top_k_indices = np.argsort(similarities)[-k:]
            similarities = [similarities[i] for i in top_k_indices]
            weighted_ratings = [weighted_ratings[i] for i in top_k_indices]
        
        # Calculate predicted rating
        if sum(similarities) == 0:
            return self.item_means.get(item_id, 3.0)
        
        predicted_rating = sum(weighted_ratings) / sum(similarities)
        
        # Ensure rating is within valid range [1, 5]
        return max(1.0, min(5.0, predicted_rating))
    
    def get_item_recommendations(self, user_id, n_recommendations=10, k=10):
        """
        Get item recommendations for a user
        """
        if user_id not in self.user_item_matrix.index:
            # For new users, recommend popular items
            item_popularity = self.ratings_df.groupby('product_id')['rating'].agg(['count', 'mean'])
            item_popularity['score'] = item_popularity['count'] * item_popularity['mean']
            top_items = item_popularity.nlargest(n_recommendations, 'score').index.tolist()
            return self._format_recommendations(top_items)
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = set(user_ratings[user_ratings > 0].index)
        
        # Get all items not rated by the user
        all_items = set(self.user_item_matrix.columns)
        unrated_items = all_items - rated_items
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            predicted_rating = self.predict_rating(user_id, item_id, k)
            predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and get top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [item_id for item_id, _ in predictions[:n_recommendations]]
        
        return self._format_recommendations(top_items)
    
    def _format_recommendations(self, item_ids):
        """
        Format recommendations with item details
        """
        recommendations = []
        for item_id in item_ids:
            if item_id in self.items_data.index:
                item_info = self.items_data.loc[item_id]
                recommendations.append({
                    'item_id': item_id,
                    'name': item_info.get('productDisplayName', f'Item {item_id}'),
                    'category': item_info.get('masterCategory', 'Unknown'),
                    'subcategory': item_info.get('subCategory', 'Unknown'),
                    'article_type': item_info.get('articleType', 'Unknown'),
                    'color': item_info.get('baseColour', 'Unknown'),
                    'gender': item_info.get('gender', 'Unknown')
                })
            else:
                recommendations.append({
                    'item_id': item_id,
                    'name': f'Item {item_id}',
                    'category': 'Unknown',
                    'subcategory': 'Unknown',
                    'article_type': 'Unknown',
                    'color': 'Unknown',
                    'gender': 'Unknown'
                })
        return recommendations
    
    def get_similar_items(self, item_id, n_similar=10):
        """
        Get items similar to a given item
        """
        if item_id not in self.item_similarity_matrix.index:
            return []
        
        # Get similarities for the item
        similarities = self.item_similarity_matrix.loc[item_id]
        
        # Sort by similarity (excluding the item itself)
        similar_items = similarities.drop(item_id).nlargest(n_similar)
        
        return self._format_recommendations(similar_items.index.tolist())
    
    def find_similar_items(self, item_id, n_similar=10):
        """
        Find items similar to a given item and return with positive correlation scores only
        Returns list of tuples: [(item_id, correlation_score), ...]
        Returns items sorted by correlation strength (highest first)
        """
        if item_id not in self.item_similarity_matrix.index:
            return []
        
        # Get similarities for the item
        similarities = self.item_similarity_matrix.loc[item_id]
        
        # Exclude the item itself, perfect correlations (1.0000), zero correlations, and negative correlations
        filtered_similarities = similarities[
            (similarities.index != item_id) & 
            (similarities < 1.0) &   # Exclude perfect correlations
            (similarities != 0.0) &  # Exclude zero correlations
            (similarities > 0.0)     # Only positive correlations
        ]
        
        # Sort correlation values in descending order (highest positive correlation first)
        similar_items = filtered_similarities.sort_values(ascending=False)
        
        # Return top N similar items with their positive correlation scores
        top_items = similar_items.head(n_similar)
        return [(item_id, float(score)) for item_id, score in top_items.items()]
    
    def save_model(self, model_path):
        """
        Save the trained model
        """
        model_data = {
            'item_similarity_matrix': self.item_similarity_matrix,
            'user_item_matrix': self.user_item_matrix,
            'item_means': self.item_means,
            'items_data': self.items_data
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.item_similarity_matrix = model_data['item_similarity_matrix']
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_means = model_data['item_means']
        self.items_data = model_data['items_data']
        
        print(f"Model loaded from {model_path}")
        return self
    
    def train(self, ratings_path, styles_path, sample_size=50000, use_enhanced_dataset=True):
        """
        Complete training pipeline
        """
        print("Training Item-Based Collaborative Filtering Model...")
        
        # Use enhanced dataset if available and requested
        if use_enhanced_dataset:
            enhanced_path = ratings_path.replace('.csv', '_enhanced.csv')
            if os.path.exists(enhanced_path):
                print(f"Using enhanced dataset: {enhanced_path}")
                ratings_path = enhanced_path
            else:
                print(f"Enhanced dataset not found at {enhanced_path}, using original dataset")
        
        self.load_data(ratings_path, styles_path, sample_size)
        self.build_item_similarity_matrix()
        
        print("Training completed!")
        return self

# if __name__ == "__main__":
#     # Example usage
#     model = ItemBasedCollaborativeFiltering()
    
#     # Train the model
#     ratings_path = "data/user_ratings.csv"
#     styles_path = "data/styles.csv"
    
#     model.train(ratings_path, styles_path)
    
#     # Save the model
#     model.save_model("models/item_based_cf_model.pkl")
    
#     print("Training completed!")