import pickle
import pandas as pd
import numpy as np
import os

def load_model_and_predict():
    # Path to the model
    model_path = "models/Physiological.pkl"
    
    # Path to your CSV file
    csv_path = r"C:\Users\Dell\OneDrive\Desktop\hardware\sensor_data.csv"
    
    try:
        # Load the model
        print("Loading Physiological model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        
        # Load the CSV data
        print(f"Loading data from: {csv_path}")
        data = pd.read_csv(csv_path)
        print(f"Data loaded! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows of data:")
        print(data.head())
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(data)
        
        # If it's a classification model, also get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            print(f"\nPrediction probabilities shape: {probabilities.shape}")
            print("Sample probabilities:")
            for i in range(min(5, len(probabilities))):
                print(f"Row {i}: {probabilities[i]}")
        
        print(f"\nPredictions shape: {predictions.shape}")
        print("Sample predictions:")
        for i in range(min(10, len(predictions))):
            print(f"Row {i}: {predictions[i]}")
        
        # Save predictions to a new CSV
        output_path = "physiological_predictions.csv"
        results_df = data.copy()
        results_df['prediction'] = predictions
        
        if hasattr(model, 'predict_proba'):
            prob_cols = [f'prob_class_{i}' for i in range(probabilities.shape[1])]
            for i, col in enumerate(prob_cols):
                results_df[col] = probabilities[:, i]
        
        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        return predictions, results_df
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    predictions, results = load_model_and_predict() 