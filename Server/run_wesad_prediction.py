#!/usr/bin/env python3
"""
Simple runner script for physiological model prediction
"""

import sys
import os
import pandas as pd
import pickle


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All dependencies are available!")
    return True

def main():
    """Main function to run physiological model prediction"""
    print("🚀 Physiological Model Prediction Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Path to your CSV file (update as needed)
    csv_path = r"C:\Users\Dell\OneDrive\Desktop\hardware\sensor_data_with_temp.csv"
    if not os.path.exists(csv_path):
        print(f"\n❌ CSV file not found: {csv_path}")
        print("Please ensure the sensor_data_with_temp.csv file exists at the specified location.")
        return
    print(f"\n✅ CSV file found: {csv_path}")
    
    # Path to the physiological model
    model_path = "models/Physiological.pkl"
    if not os.path.exists(model_path):
        print(f"\n❌ Physiological model not found: {model_path}")
        return
    print(f"✅ Physiological model found: {model_path}")
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Load the CSV data
    try:
        data = pd.read_csv(csv_path)
        print(f"Data loaded! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\nFirst 5 rows of data:")
        print(data.head())
    except Exception as e:
        print(f"\n❌ Error loading CSV: {e}")
        return
    
    # Make predictions
    try:
        print("\nMaking predictions...")
        predictions = model.predict(data)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            print(f"\nPrediction probabilities shape: {probabilities.shape}")
            print("Sample probabilities:")
            for i in range(min(5, len(probabilities))):
                print(f"Row {i}: {probabilities[i]}")
        else:
            probabilities = None
        print(f"\nPredictions shape: {predictions.shape}")
        print("Sample predictions:")
        for i in range(min(10, len(predictions))):
            print(f"Row {i}: {predictions[i]}")
        # Save predictions to a new CSV
        output_path = "physiological_predictions.csv"
        results_df = data.copy()
        results_df['prediction'] = predictions
        if probabilities is not None:
            prob_cols = [f'prob_class_{i}' for i in range(probabilities.shape[1])]
            for i, col in enumerate(prob_cols):
                results_df[col] = probabilities[:, i]
        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        return

if __name__ == "__main__":
    main() 