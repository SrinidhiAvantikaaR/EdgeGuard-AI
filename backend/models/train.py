"""
Training script using REAL system data
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
from models.detector import create_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(filename="training_data.json"):
    """Load collected training data"""
    if not os.path.exists(filename):
        logger.error(f"❌ Training data file {filename} not found!")
        logger.info("Please run collect_training_data.py first")
        return None, None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    normal_data = data['normal']
    ransomware_data = data['ransomware']
    
    logger.info(f"✅ Loaded {len(normal_data)} normal samples")
    logger.info(f"✅ Loaded {len(ransomware_data)} ransomware samples")
    
    return normal_data, ransomware_data

def prepare_training_data(normal_data, ransomware_data):
    """Prepare data for training"""
    all_data = []
    labels = []
    
    # Add normal data (label 1 for normal)
    for sample in normal_data:
        # Remove metadata fields
        clean_sample = {k: v for k, v in sample.items() 
                       if k not in ['process_name', 'pid', 'timestamp']}
        all_data.append(clean_sample)
        labels.append(1)  # Normal
    
    # Add ransomware data (label -1 for anomaly)
    for sample in ransomware_data:
        clean_sample = {k: v for k, v in sample.items() 
                       if k not in ['process_name', 'pid', 'timestamp']}
        all_data.append(clean_sample)
        labels.append(-1)  # Anomaly
    
    return all_data, labels

def analyze_data_distribution(data):
    """Analyze the distribution of features"""
    df = pd.DataFrame(data)
    
    print("\n📊 Data Distribution Analysis:")
    print("-" * 40)
    for col in df.columns:
        if col not in ['process_name', 'pid', 'timestamp']:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.2f}")
            print(f"  Max: {df[col].max():.2f}")
            print(f"  25%: {df[col].quantile(0.25):.2f}")
            print(f"  75%: {df[col].quantile(0.75):.2f}")

def train_model():
    """Train model with REAL system data"""
    logger.info("=" * 60)
    logger.info("Training EdgeGuard AI with REAL System Data")
    logger.info("=" * 60)
    
    # Load data
    normal_data, ransomware_data = load_training_data()
    if normal_data is None:
        return None
    
    # Prepare data
    all_data, labels = prepare_training_data(normal_data, ransomware_data)
    
    # Analyze distribution
    analyze_data_distribution(all_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Handle any missing values
    df = df.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"\n📊 Training set: {len(X_train)} samples")
    logger.info(f"📊 Test set: {len(X_test)} samples")
    
    # Initialize detector
    detector = create_detector()
    
    # Train model
    logger.info("\n🚀 Training model...")
    accuracy = detector.train(X_train.to_dict('records'))
    
    # Test on separate test set
    logger.info("\n🧪 Testing on unseen data...")
    y_pred = []
    for _, row in X_test.iterrows():
        score = detector.predict(row.to_dict())
        # Convert score to label (threshold 0.5)
        pred_label = -1 if score > 0.5 else 1
        y_pred.append(pred_label)
    
    # Calculate test accuracy
    test_accuracy = np.mean(np.array(y_pred) == np.array(y_test))
    
    logger.info(f"\n✅ Training Results:")
    logger.info(f"   Training accuracy: {accuracy:.2f}")
    logger.info(f"   Test accuracy: {test_accuracy:.2f}")
    
    # Show feature importance
    print("\n📈 Feature Analysis:")
    print("-" * 40)
    if hasattr(detector.model, 'feature_importances_'):
        importances = detector.model.feature_importances_
        for name, imp in zip(detector.feature_names, importances):
            print(f"  {name}: {imp:.3f}")
    
    # Test with some real processes
    print("\n🖥️ Testing with current processes:")
    print("-" * 40)
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 1:
                # Create feature dict
                test_feat = {
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.memory_percent(),
                    'file_writes_rate': 0,
                    'entropy': 0.3,
                    'num_threads': proc.num_threads(),
                    'connections': len(proc.connections(kind='inet')),
                    'cpu_burst': 1 if proc.info['cpu_percent'] > 70 else 0,
                    'sudden_cpu_change': 0,
                    'file_type_changes': 0
                }
                score = detector.predict(test_feat)
                if score > 0.6:
                    print(f"  ⚠️ {proc.info['name']}: {score:.2f}")
        except:
            continue
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"Model saved to: {detector.model_path}")
    print("=" * 60)
    
    return detector

if __name__ == "__main__":
    train_model()