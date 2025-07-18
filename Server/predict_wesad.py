import os
import pickle
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Feature extraction libraries
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt

# ======================== CONFIG ========================
CFG = {
    "orig_fs": 700,
    "fs": 100,
    "window_sec": 10,
    "stride_sec": 5,
    "sensors": ["ECG", "EDA", "EMG", "Temp"],
    "label_map": {0: 0, 3: 0, 2: 1, 1: 2},
    "class_names": ["Low", "Medium", "High"],
    "seed": 42,
}

DOWN_F = CFG["orig_fs"] // CFG["fs"]
STEP = CFG["window_sec"] * CFG["fs"]
STRIDE = CFG["stride_sec"] * CFG["fs"]

# ======================== FEATURE EXTRACTION FUNCTIONS ========================
def clean_data(X):
    """Convert data to float and handle NaN/infinite values"""
    try:
        X = np.asarray(X)
        if X.dtype.kind not in 'f':
            X = X.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=np.nanmax(X[X != np.inf]), neginf=np.nanmin(X[X != -np.inf]))
        return X
    except Exception as e:
        print(f"⚠ Error in clean_data: {str(e)}")
        return np.zeros_like(X) if isinstance(X, np.ndarray) else np.zeros((1, 180))

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def downsample(x):
    return x[::DOWN_F]

def extract_time_features(signal_data):
    """Extract time-domain features"""
    try:
        signal_data = np.asarray(signal_data).astype(float)
        features = [
            np.mean(signal_data),
            np.std(signal_data),
            np.var(signal_data),
            skew(signal_data),
            kurtosis(signal_data),
            np.min(signal_data),
            np.max(signal_data),
            np.ptp(signal_data),
            np.median(signal_data),
            np.percentile(signal_data, 25),
            np.percentile(signal_data, 75),
            np.mean(np.abs(np.diff(signal_data))),
            np.sqrt(np.mean(signal_data**2))
        ]
        return features
    except:
        return [0.0] * 13

def extract_freq_features(signal_data, fs=100):
    """Extract frequency-domain features"""
    try:
        signal_data = np.asarray(signal_data).astype(float)
        if len(signal_data) < 8:
            return [0.0] * 11
        
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)//4))
        
        bands = {
            'very_low': (0.0, 0.04),
            'low': (0.04, 0.15),
            'mid': (0.15, 0.4),
            'high': (0.4, 0.5)
        }
        
        total_power = np.sum(psd)
        features = []
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            features.append(band_power)
            features.append(band_power / (total_power + 1e-8))
        
        features.extend([
            np.mean(freqs),
            np.std(freqs),
            freqs[np.argmax(psd)]
        ])
        return features
    except:
        return [0.0] * 11

def extract_wavelet_features(signal_data):
    """Extract wavelet-domain features"""
    try:
        signal_data = np.asarray(signal_data).astype(float)
        coeffs = pywt.wavedec(signal_data, 'db4', level=4)
        features = []
        for coeff in coeffs:
            if len(coeff) > 0:
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.var(coeff),
                    np.max(np.abs(coeff))
                ])
        return features if features else [0.0] * 16
    except:
        return [0.0] * 16

def extract_ecg_features(signal_data, fs=100):
    """Extract ECG-specific features"""
    try:
        signal_data = np.asarray(signal_data).astype(float)
        peaks, _ = signal.find_peaks(signal_data, height=np.std(signal_data), distance=fs//3)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs * 1000
            return [
                np.mean(rr_intervals),
                np.std(rr_intervals),
                np.sqrt(np.mean(np.diff(rr_intervals)**2)),
                len(peaks) / (len(signal_data) / fs) * 60
            ]
        return [0.0] * 4
    except:
        return [0.0] * 4

def extract_window_features(sigs):
    """Extract comprehensive features from a window"""
    features = []
    feature_names = []
    
    for sensor, signal_data in sigs.items():
        # Time features
        time_feats = extract_time_features(signal_data)
        features.extend(time_feats)
        time_names = [f"{sensor}_mean", f"{sensor}_std", f"{sensor}_var", f"{sensor}_skew", 
                     f"{sensor}_kurt", f"{sensor}_min", f"{sensor}_max", f"{sensor}_ptp",
                     f"{sensor}_median", f"{sensor}_q25", f"{sensor}_q75", f"{sensor}_mad", f"{sensor}_rms"]
        feature_names.extend(time_names)
        
        # Frequency features
        freq_feats = extract_freq_features(signal_data)
        features.extend(freq_feats)
        freq_names = [f"{sensor}_vl_power", f"{sensor}_vl_rel", f"{sensor}_l_power", f"{sensor}_l_rel",
                     f"{sensor}_m_power", f"{sensor}_m_rel", f"{sensor}_h_power", f"{sensor}_h_rel",
                     f"{sensor}_freq_mean", f"{sensor}_freq_std", f"{sensor}_dom_freq"]
        feature_names.extend(freq_names)
        
        # Wavelet features
        wavelet_feats = extract_wavelet_features(signal_data)
        features.extend(wavelet_feats)
        wavelet_names = [f"{sensor}wl{i}" for i in range(len(wavelet_feats))]
        feature_names.extend(wavelet_names)
        
        if sensor == "ECG":
            ecg_feats = extract_ecg_features(signal_data)
            features.extend(ecg_feats)
            ecg_names = [f"{sensor}_mean_rr", f"{sensor}_sdnn", f"{sensor}_rmssd", f"{sensor}_hr"]
            feature_names.extend(ecg_names)
    
    features = np.array(features, dtype=float)
    return clean_data(features), feature_names

def process_csv_data(csv_path):
    """Process CSV data and extract features using WESAD methodology"""
    try:
        print(f"Loading data from: {csv_path}")
        data = pd.read_csv(csv_path)
        print(f"Data loaded! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows of data:")
        print(data.head())
        
        # Check if we have the required sensors
        required_sensors = CFG["sensors"]
        available_columns = list(data.columns)
        
        print(f"\nRequired sensors: {required_sensors}")
        print(f"Available columns: {available_columns}")
        
        # Map available columns to required sensors
        sensor_mapping = {}
        for sensor in required_sensors:
            # Try exact match first
            if sensor in available_columns:
                sensor_mapping[sensor] = sensor
            # Try case-insensitive match
            elif sensor.lower() in [col.lower() for col in available_columns]:
                matching_col = [col for col in available_columns if col.lower() == sensor.lower()][0]
                sensor_mapping[sensor] = matching_col
            # Try partial match
            else:
                partial_matches = [col for col in available_columns if sensor.lower() in col.lower()]
                if partial_matches:
                    sensor_mapping[sensor] = partial_matches[0]
                else:
                    print(f"⚠ Warning: No column found for sensor {sensor}")
                    sensor_mapping[sensor] = None
        
        print(f"Sensor mapping: {sensor_mapping}")
        
        # Extract features for each window
        all_features = []
        feature_names = None
        
        # Process data in windows
        window_size = STEP
        stride_size = STRIDE
        
        for start_idx in range(0, len(data) - window_size + 1, stride_size):
            end_idx = start_idx + window_size
            
            # Extract window data
            window_data = data.iloc[start_idx:end_idx]
            
            # Prepare signals dictionary
            signals = {}
            for sensor, col_name in sensor_mapping.items():
                if col_name is not None and col_name in window_data.columns:
                    sig = window_data[col_name].values
                    signals[sensor] = zscore(sig.flatten())
                else:
                    # Create dummy signal if sensor not available
                    signals[sensor] = np.zeros(window_size)
            
            # Extract features for this window
            features, names = extract_window_features(signals)
            
            if feature_names is None:
                feature_names = names
            
            all_features.append(features)
        
        if not all_features:
            raise ValueError("No features extracted from data")
        
        # Convert to numpy array
        X = np.array(all_features, dtype=float)
        
        print(f"\nFeature extraction completed:")
        print(f"   Number of windows: {len(X)}")
        print(f"   Number of features: {X.shape[1]}")
        
        
        return X, feature_names
        
    except Exception as e:
        print(f"Error processing CSV data: {str(e)}")
        return None, None

def load_wesad_model():
    """Load the WESAD model"""
    try:
        # Try to load from result_WESAD directory first
        model_paths = [
            "result_WESAD/regularized_global_model.pkl",
            "models/regularized_global_model.pkl",
            "regularized_global_model.pkl"
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading model from: {path}")
                model = joblib.load(path)
                print("Model loaded successfully!")
                break
        
        if model is None:
            print("❌ No WESAD model found. Please ensure the model is saved as 'regularized_global_model.pkl'")
            return None
            
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_with_wesad_model():
    """Main function to load model and make predictions"""
    # Path to your CSV file
    csv_path = r"C:\Users\Dell\OneDrive\Desktop\hardware\sensor_data_with_temp.csv"
    
    try:
        # Load the WESAD model
        model = load_wesad_model()
        if model is None:
            return None, None
        
        # Process the CSV data
        X, feature_names = process_csv_data(csv_path)
        if X is None:
            return None, None
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(X)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            print(f"\nPrediction probabilities shape: {probabilities.shape}")
            print("Sample probabilities:")
            for i in range(min(5, len(probabilities))):
                print(f"Window {i}: {probabilities[i]} -> Class: {CFG['class_names'][predictions[i]]}")
        else:
            probabilities = None
        
        print(f"\nPredictions shape: {predictions.shape}")
        print("Sample predictions:")
        for i in range(min(10, len(predictions))):
            class_name = CFG['class_names'][predictions[i]]
            print(f"Window {i}: {predictions[i]} -> {class_name}")
        
        # Save predictions to a new CSV
        output_path = "wesad_predictions.csv"
        
        # Create results DataFrame
        results_df = pd.DataFrame()
        
        # Add window information
        results_df['window_id'] = range(len(predictions))
        results_df['prediction'] = predictions
        results_df['predicted_class'] = [CFG['class_names'][pred] for pred in predictions]
        
        # Add probabilities if available
        if probabilities is not None:
            for i, class_name in enumerate(CFG['class_names']):
                results_df[f'prob_{class_name}'] = probabilities[:, i]
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        # Print summary statistics
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"Total windows processed: {len(predictions)}")
        print(f"Class distribution:")
        for i, class_name in enumerate(CFG['class_names']):
            count = np.sum(predictions == i)
            percentage = (count / len(predictions)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return predictions, results_df
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    predictions, results = predict_with_wesad_model() 