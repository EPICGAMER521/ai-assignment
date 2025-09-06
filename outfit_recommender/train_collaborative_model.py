#!/usr/bin/env python3
"""
Training script for Item-Based Collaborative Filtering Model
Uses the dataset with synthetic ratings and improved parameters
"""

import os
import sys
import numpy as np
from item_based_cf import ItemBasedCollaborativeFiltering

def main():
    print("Starting Item-Based Collaborative Filtering Training...")
    print("=" * 70)
    
    # Define paths - using dataset
    ratings_path = "data/user_ratings.csv"
    styles_path = "data/styles.csv"
    model_save_path = "models/item_based_cf_model.pkl"
    
    # Check if data files exist
    if not os.path.exists(ratings_path):
        print(f"Error: {ratings_path} not found!")
        print("Please run the dataset script first.")
        return
    
    if not os.path.exists(styles_path):
        print(f"Error: {styles_path} not found!")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Initialize and train the model with parameters
        print("Initializing Item-Based Collaborative Filtering model...")
        model = ItemBasedCollaborativeFiltering()
        
        print("Training the model with dataset...")
        print("Using larger sample size for better correlation calculations...")
        
        # Use larger sample size with dataset (100,000 instead of 50,000)
        model.train(ratings_path, styles_path, sample_size=100000)
        
        # Save the trained model
        print("Saving the trained model...")
        model.save_model(model_save_path)
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print(f"Model saved to: {model_save_path}")
        
        # Display statistics
        print("\nModel Statistics:")
        print(f"- Number of users: {model.user_item_matrix.shape[0]}")
        print(f"- Number of items: {model.user_item_matrix.shape[1]}")
        print(f"- Similarity matrix shape: {model.item_similarity_matrix.shape}")
        
        # Calculate and display correlation statistics
        print("\nCorrelation Analysis:")
        similarity_matrix = model.item_similarity_matrix
        
        # Convert to numpy array if it's a pandas DataFrame
        if hasattr(similarity_matrix, 'values'):
            sim_array = similarity_matrix.values
        else:
            sim_array = similarity_matrix
        
        # Count non-zero correlations
        total_correlations = int(sim_array.size)
        zero_correlations = int(np.sum(sim_array == 0.0))
        non_zero_correlations = total_correlations - zero_correlations
        
        print(f"- Total correlations: {total_correlations:,}")
        print(f"- Zero correlations: {zero_correlations:,} ({zero_correlations/total_correlations*100:.2f}%)")
        print(f"- Non-zero correlations: {non_zero_correlations:,} ({non_zero_correlations/total_correlations*100:.2f}%)")
        
        if non_zero_correlations > 0:
            non_zero_values = sim_array[sim_array != 0.0]
            print(f"- Non-zero correlation range: {non_zero_values.min():.4f} to {non_zero_values.max():.4f}")
            print(f"- Average non-zero correlation: {non_zero_values.mean():.4f}")
        
        # Test with a sample recommendation
        print("\nTesting with sample recommendations...")
        sample_user_id = model.user_item_matrix.index[0]
        recommendations = model.get_item_recommendations(sample_user_id, n_recommendations=5)
        
        print(f"\nSample recommendations for user {sample_user_id}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} ({rec['category']} - {rec['article_type']})")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nTraining process completed!")
    print("The model should now have better correlation scores due to:")
    print("- Dataset with synthetic ratings")
    print("- Larger sample size for training")
    print("- Improved data density")

if __name__ == "__main__":
    main()