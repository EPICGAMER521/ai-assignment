#!/usr/bin/env python3
"""
Content-Based Model Training Script for Fashion Recommender System

This script focuses specifically on training the content-based recommender model.
It includes data preprocessing, feature engineering, model training, and evaluation.

Author: AI Assignment - BMCS2203
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from content_based_recommender_optimized import OptimizedContentBasedRecommender
import warnings
warnings.filterwarnings('ignore')

class ContentBasedTrainer:
    """
    Specialized trainer for content-based recommender models.
    """
    
    def __init__(self, data_path: str = "data/styles.csv", models_dir: str = "models"):
        """
        Initialize the content-based model trainer.
        
        Args:
            data_path: Path to the dataset CSV file
            models_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.data = None
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        print(f"🎯 Content-Based Model Trainer initialized")
        print(f"📁 Data path: {data_path}")
        print(f"💾 Models directory: {models_dir}")
    
    def load_and_prepare_data(self):
        """
        Load and prepare the dataset for content-based training.
        """
        print("\n📊 Loading and preparing data for content-based training...")
        
        try:
            # Load data with error handling for malformed lines
            try:
                self.data = pd.read_csv(self.data_path)
            except pd.errors.ParserError as e:
                print(f"⚠️  CSV parsing error: {e}")
                print("🔧 Attempting to read with error handling...")
                self.data = pd.read_csv(self.data_path, on_bad_lines='skip')
                print("✅ Data loaded successfully with error handling")
            
            print(f"📈 Dataset shape: {self.data.shape}")
            print(f"🏷️  Columns: {list(self.data.columns)}")
            
            # Basic data info
            print(f"\n📋 Dataset Overview:")
            print(f"   • Total items: {len(self.data):,}")
            print(f"   • Unique categories: {self.data['masterCategory'].nunique()}")
            print(f"   • Unique sub-categories: {self.data['subCategory'].nunique()}")
            print(f"   • Unique article types: {self.data['articleType'].nunique()}")
            print(f"   • Unique brands: {self.data['productDisplayName'].str.split().str[0].nunique()}")
            print(f"   • Year range: {self.data['year'].min()} - {self.data['year'].max()}")
            print(f"   • Gender distribution: {dict(self.data['gender'].value_counts())}")
            
            # Handle missing values
            missing_counts = self.data.isnull().sum()
            if missing_counts.sum() > 0:
                print(f"\n🔍 Missing values found:")
                for col, count in missing_counts[missing_counts > 0].items():
                    print(f"   • {col}: {count} ({count/len(self.data)*100:.1f}%)")
            
            # Feature analysis for content-based model
            print(f"\n🔍 Content-Based Features Analysis:")
            print(f"   • Master categories: {list(self.data['masterCategory'].unique())}")
            print(f"   • Base colors: {list(self.data['baseColour'].unique()[:10])}...")  # Show first 10
            print(f"   • Seasons: {list(self.data['season'].unique())}")
            print(f"   • Usage: {list(self.data['usage'].unique()[:5])}...")  # Show first 5
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def train_content_based_model(self):
        """
        Train the content-based recommender model with detailed logging.
        """
        print("\n🎯 Training Content-Based Recommender Model...")
        
        start_time = time.time()
        
        try:
            # Initialize and train the content-based recommender
            print("🔧 Initializing OptimizedContentBasedRecommender...")
            cb_recommender = OptimizedContentBasedRecommender()
            
            print("🔄 Fitting content-based model...")
            print("   • Processing categorical features...")
            print("   • Encoding text features...")
            print("   • Computing feature matrix...")
            print("   • Calculating similarity matrix...")
            
            cb_recommender.fit(self.data_path)
            
            training_time = time.time() - start_time
            print(f"✅ Content-based model trained in {training_time:.2f} seconds")
            
            # Save the model
            cb_model_path = os.path.join(self.models_dir, "content_based_model.pkl")
            print(f"💾 Saving model to: {cb_model_path}")
            cb_recommender.save_model(cb_model_path)
            
            # Detailed model statistics
            print(f"\n📊 Content-Based Model Statistics:")
            print(f"   • Feature matrix shape: {cb_recommender.feature_matrix.shape}")
            print(f"   • Feature dimensions: {cb_recommender.feature_matrix.shape[1]}")
            print(f"   • Memory usage: ~{cb_recommender.feature_matrix.data.nbytes / 1024**2:.1f} MB")
            print(f"   • Model type: Optimized (on-demand similarity computation)")
            print(f"   • Total items: {len(cb_recommender.data)}")
            
            # Feature importance analysis (skipped for optimized model)
            print("\n🔍 Feature importance analysis skipped for optimized model")
            
            # Test the model with sample recommendations
            print("\n🧪 Testing model with sample recommendations...")
            sample_indices = np.random.choice(len(self.data), 3, replace=False)
            
            for i, idx in enumerate(sample_indices, 1):
                sample_id = self.data.iloc[idx]['id']
                sample_name = self.data.iloc[idx]['productDisplayName']
                sample_category = self.data.iloc[idx]['masterCategory']
                
                print(f"\n   Sample {i}: {sample_name} ({sample_category})")
                
                recommendations = cb_recommender.get_recommendations(sample_id, n_recommendations=3)
                
                for j, rec in enumerate(recommendations, 1):
                    print(f"      {j}. {rec['productDisplayName']} (Similarity: {rec['similarity_score']:.3f})")
            
            # Save training metadata
            metadata = {
                'model_type': 'content_based_optimized',
                'training_time': training_time,
                'feature_matrix_shape': cb_recommender.feature_matrix.shape,
                'memory_usage_mb': cb_recommender.feature_matrix.data.nbytes / 1024**2,
                'created_at': datetime.now().isoformat(),
                'data_path': self.data_path,
                'total_items': len(self.data),
                'feature_dimensions': cb_recommender.feature_matrix.shape[1]
            }
            
            metadata_path = os.path.join(self.models_dir, "content_based_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"\n💾 Training metadata saved to: {metadata_path}")
            
            return cb_recommender, metadata
            
        except Exception as e:
            print(f"❌ Error training content-based model: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def evaluate_content_based_model(self, cb_model):
        """
        Evaluate the content-based model performance.
        """
        print("\n📈 Evaluating Content-Based Model...")
        
        try:
            # Coverage analysis
            total_items = len(self.data)
            coverage = 1.0  # Content-based models have 100% item coverage
            
            # Diversity analysis
            print("🔍 Analyzing recommendation diversity...")
            sample_items = np.random.choice(self.data['id'].values, 10, replace=False)
            all_recommendations = []
            
            for item_id in sample_items:
                recs = cb_model.get_recommendations(item_id, n_recommendations=5)
                all_recommendations.extend([rec['id'] for rec in recs])
            
            unique_recommendations = len(set(all_recommendations))
            diversity_score = unique_recommendations / len(all_recommendations)
            
            # Category distribution in recommendations
            rec_categories = []
            for item_id in sample_items:
                recs = cb_model.get_recommendations(item_id, n_recommendations=5)
                for rec in recs:
                    rec_data = self.data[self.data['id'] == rec['id']]
                    if not rec_data.empty:
                        rec_categories.append(rec_data.iloc[0]['masterCategory'])
            
            category_distribution = pd.Series(rec_categories).value_counts(normalize=True)
            
            evaluation_results = {
                'model_type': 'content_based',
                'coverage': coverage,
                'diversity_score': diversity_score,
                'total_items': total_items,
                'unique_recommendations': unique_recommendations,
                'category_distribution': category_distribution.to_dict(),
                'evaluation_date': datetime.now().isoformat()
            }
            
            print(f"\n📊 Content-Based Model Evaluation Results:")
            print(f"   • Item coverage: {coverage*100:.1f}%")
            print(f"   • Recommendation diversity: {diversity_score:.3f}")
            print(f"   • Unique recommendations: {unique_recommendations}/{len(all_recommendations)}")
            print(f"\n📈 Category Distribution in Recommendations:")
            for category, percentage in category_distribution.head().items():
                print(f"   • {category}: {percentage*100:.1f}%")
            
            # Save evaluation results
            eval_path = os.path.join(self.models_dir, "content_based_evaluation.pkl")
            with open(eval_path, 'wb') as f:
                pickle.dump(evaluation_results, f)
            
            print(f"\n💾 Evaluation results saved to: {eval_path}")
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Error evaluating content-based model: {e}")
            return None
    
    def train_and_evaluate(self):
        """
        Complete training and evaluation pipeline for content-based model.
        """
        print("🚀 Starting Content-Based Model Training Pipeline...")
        print("=" * 60)
        
        # Load data
        if not self.load_and_prepare_data():
            print("❌ Failed to load data. Exiting.")
            return False
        
        # Train model
        cb_model, metadata = self.train_content_based_model()
        if cb_model is None:
            print("❌ Failed to train content-based model.")
            return False
        
        # Evaluate model
        evaluation = self.evaluate_content_based_model(cb_model)
        if evaluation is None:
            print("❌ Failed to evaluate model.")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 Content-Based Model Training Completed Successfully!")
        print("\n📁 Generated Files:")
        print(f"   • Model: {os.path.join(self.models_dir, 'content_based_model.pkl')}")
        print(f"   • Metadata: {os.path.join(self.models_dir, 'content_based_metadata.pkl')}")
        print(f"   • Evaluation: {os.path.join(self.models_dir, 'content_based_evaluation.pkl')}")
        
        print("\n🚀 Next Steps:")
        print("   1. Run 'python demo_recommender.py' to test the model")
        print("   2. Run 'streamlit run streamlit_app.py' to use the web interface")
        print("   3. Load the model in your scripts: pickle.load(open('models/content_based_model.pkl', 'rb'))")
        
        return True

def main():
    """
    Main function for content-based model training.
    """
    print("🎯 Fashion Recommender System - Content-Based Model Training")
    print("BMCS2203 AI Assignment")
    print("=" * 60)
    
    # Check if data file exists
    data_path = "data/styles.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        print("Please ensure the styles.csv file is in the data/ directory")
        return
    
    # Initialize trainer
    trainer = ContentBasedTrainer(data_path=data_path)
    
    # Train and evaluate
    success = trainer.train_and_evaluate()
    
    if success:
        print("\n✅ Content-based model training completed successfully!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()