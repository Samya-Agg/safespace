
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import pickle
import io
import json
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
from typing import List, Optional, Dict, Any
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# Import your existing fusion model
from latefusion_final import PhysioDominantFusion

# CORS Setup
app = FastAPI(title="SafeSpace Stress Detection API with XAI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Configuration
CFG = {
    "orig_fs": 700,
    "fs": 100,
    "window_sec": 10,
    "stride_sec": 5,
    "sensors": ["ECG", "EDA", "EMG", "Temp"],
}

DOWN_F = CFG["orig_fs"] // CFG["fs"]
STEP = CFG["window_sec"] * CFG["fs"]
STRIDE = CFG["stride_sec"] * CFG["fs"]

# Feature names for interpretability
FEATURE_NAMES = {
    "ECG": [
        "mean", "std", "var", "skew", "kurtosis", "min", "max", "ptp", "median", 
        "q25", "q75", "mean_abs_diff", "rms",
        "vlow_power", "vlow_rel", "low_power", "low_rel", "mid_power", "mid_rel", 
        "high_power", "high_rel", "freq_mean", "freq_std", "peak_freq",
        "wav_d1_mean", "wav_d1_std", "wav_d1_var", "wav_d1_max", "wav_d2_mean", 
        "wav_d2_std", "wav_d2_var", "wav_d2_max", "wav_d3_mean", "wav_d3_std", 
        "wav_d3_var", "wav_d3_max", "wav_d4_mean", "wav_d4_std", "wav_d4_var", 
        "wav_d4_max", "wav_a4_mean", "wav_a4_std", "wav_a4_var", "wav_a4_max",
        "mean_rr", "std_rr", "rmssd", "heart_rate"
    ],
    "EDA": [
        "mean", "std", "var", "skew", "kurtosis", "min", "max", "ptp", "median", 
        "q25", "q75", "mean_abs_diff", "rms",
        "vlow_power", "vlow_rel", "low_power", "low_rel", "mid_power", "mid_rel", 
        "high_power", "high_rel", "freq_mean", "freq_std", "peak_freq",
        "wav_d1_mean", "wav_d1_std", "wav_d1_var", "wav_d1_max", "wav_d2_mean", 
        "wav_d2_std", "wav_d2_var", "wav_d2_max", "wav_d3_mean", "wav_d3_std", 
        "wav_d3_var", "wav_d3_max", "wav_d4_mean", "wav_d4_std", "wav_d4_var", 
        "wav_d4_max", "wav_a4_mean", "wav_a4_std", "wav_a4_var", "wav_a4_max"
    ],
    "EMG": [
        "mean", "std", "var", "skew", "kurtosis", "min", "max", "ptp", "median", 
        "q25", "q75", "mean_abs_diff", "rms",
        "vlow_power", "vlow_rel", "low_power", "low_rel", "mid_power", "mid_rel", 
        "high_power", "high_rel", "freq_mean", "freq_std", "peak_freq",
        "wav_d1_mean", "wav_d1_std", "wav_d1_var", "wav_d1_max", "wav_d2_mean", 
        "wav_d2_std", "wav_d2_var", "wav_d2_max", "wav_d3_mean", "wav_d3_std", 
        "wav_d3_var", "wav_d3_max", "wav_d4_mean", "wav_d4_std", "wav_d4_var", 
        "wav_d4_max", "wav_a4_mean", "wav_a4_std", "wav_a4_var", "wav_a4_max"
    ],
    "Temp": [
        "mean", "std", "var", "skew", "kurtosis", "min", "max", "ptp", "median", 
        "q25", "q75", "mean_abs_diff", "rms",
        "vlow_power", "vlow_rel", "low_power", "low_rel", "mid_power", "mid_rel", 
        "high_power", "high_rel", "freq_mean", "freq_std", "peak_freq",
        "wav_d1_mean", "wav_d1_std", "wav_d1_var", "wav_d1_max", "wav_d2_mean", 
        "wav_d2_std", "wav_d2_var", "wav_d2_max", "wav_d3_mean", "wav_d3_std", 
        "wav_d3_var", "wav_d3_max", "wav_d4_mean", "wav_d4_std", "wav_d4_var", 
        "wav_d4_max", "wav_a4_mean", "wav_a4_std", "wav_a4_var", "wav_a4_max"
    ]
}

# Create flat feature names list
ALL_FEATURE_NAMES = []
for sensor in CFG["sensors"]:
    for feature in FEATURE_NAMES[sensor]:
        ALL_FEATURE_NAMES.append(f"{sensor}_{feature}")

DASS21_FEATURE_NAMES = [
    "DASS21_Q1_breathing_difficulty",
    "DASS21_Q2_dry_mouth", 
    "DASS21_Q3_positive_feelings",
    "DASS21_Q4_breathing_shortness",
    "DASS21_Q5_action_initiative",
    "DASS21_Q6_overreact_tendency",
    "DASS21_Q7_trembling_hands"
]

class XAIExplainer:
    """Explainable AI component for stress detection models"""
    
    def __init__(self):
        self.physio_explainer = None
        self.dass21_explainer = None
        self.feature_importance_cache = {}
        
    def setup_physio_explainer(self, model, X_background):
        """Setup SHAP explainer for physiological model"""
        try:
            # Use a subset of background data for efficiency
            background_sample = X_background[:min(100, len(X_background))]
            
            # Try TreeExplainer first (for tree-based models)
            try:
                self.physio_explainer = shap.TreeExplainer(model)
                print("✓ TreeExplainer initialized for physiological model")
            except:
                # Fall back to KernelExplainer
                def model_predict(X):
                    return model.predict_proba(X)
                self.physio_explainer = shap.KernelExplainer(model_predict, background_sample)
                print("✓ KernelExplainer initialized for physiological model")
                
        except Exception as e:
            print(f"⚠ Failed to initialize physio explainer: {e}")
            
    def setup_dass21_explainer(self, model, scaler, X_background):
        """Setup SHAP explainer for DASS-21 model"""
        try:
            background_sample = X_background[:min(100, len(X_background))]
            
            def model_predict(X):
                X_scaled = scaler.transform(X)
                return model.predict_proba(X_scaled)
                
            self.dass21_explainer = shap.KernelExplainer(model_predict, background_sample)
            print("✓ KernelExplainer initialized for DASS-21 model")
            
        except Exception as e:
            print(f"⚠ Failed to initialize DASS-21 explainer: {e}")
    
    def explain_physio_prediction(self, X_sample, top_k=10):
        """Generate explanations for physiological predictions"""
        explanations = {
            "available": False,
            "method": "SHAP",
            "feature_importance": [],
            "summary": ""
        }
        
        if self.physio_explainer is None:
            return explanations
            
        try:
            # Get SHAP values
            shap_values = self.physio_explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # For multi-class, use the predicted class
                predicted_class = np.argmax(X_sample.sum(axis=0))  # Simple heuristic
                shap_values_class = shap_values[predicted_class]
            else:
                shap_values_class = shap_values
                
            # Average SHAP values across windows
            mean_shap = np.mean(shap_values_class, axis=0)
            
            # Get top features
            feature_importance = []
            for i, importance in enumerate(mean_shap):
                if i < len(ALL_FEATURE_NAMES):
                    feature_importance.append({
                        "feature": ALL_FEATURE_NAMES[i],
                        "importance": float(importance),
                        "abs_importance": float(abs(importance))
                    })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: x["abs_importance"], reverse=True)
            
            explanations.update({
                "available": True,
                "feature_importance": feature_importance[:top_k],
                "summary": self._generate_physio_summary(feature_importance[:top_k])
            })
            
        except Exception as e:
            print(f"⚠ Failed to generate physio explanations: {e}")
            explanations["error"] = str(e)
            
        return explanations
    
    def explain_dass21_prediction(self, X_sample, top_k=7):
        """Generate explanations for DASS-21 predictions"""
        explanations = {
            "available": False,
            "method": "SHAP",
            "feature_importance": [],
            "summary": ""
        }
        
        if self.dass21_explainer is None:
            return explanations
            
        try:
            # Get SHAP values
            shap_values = self.dass21_explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # Average across classes or use highest probability class
                mean_shap = np.mean([sv[0] for sv in shap_values], axis=0)
            else:
                mean_shap = shap_values[0]
            
            # Get feature importance
            feature_importance = []
            for i, importance in enumerate(mean_shap):
                if i < len(DASS21_FEATURE_NAMES):
                    feature_importance.append({
                        "feature": DASS21_FEATURE_NAMES[i],
                        "importance": float(importance),
                        "abs_importance": float(abs(importance)),
                        "value": float(X_sample[0][i])
                    })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: x["abs_importance"], reverse=True)
            
            explanations.update({
                "available": True,
                "feature_importance": feature_importance[:top_k],
                "summary": self._generate_dass21_summary(feature_importance[:top_k])
            })
            
        except Exception as e:
            print(f"⚠ Failed to generate DASS-21 explanations: {e}")
            explanations["error"] = str(e)
            
        return explanations
    
    def explain_fusion_decision(self, fusion_input, fusion_probs):
        """Generate explanations for fusion model decisions"""
        explanations = {
            "available": True,
            "method": "Weight Analysis",
            "modality_contributions": [],
            "summary": ""
        }
        
        try:
            # Analyze modality contributions
            modalities = ["physiological", "questionnaire", "voice"]
            contributions = []
            
            for i, modality in enumerate(modalities):
                if modality == "physiological":
                    prob_dist = fusion_input["phys"]
                elif modality == "questionnaire":
                    prob_dist = fusion_input["text"]
                else:  # voice
                    prob_dist = fusion_input["voice"]
                
                # Calculate contribution metrics
                entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
                confidence = np.max(prob_dist)
                predicted_class = np.argmax(prob_dist)
                
                contributions.append({
                    "modality": modality,
                    "probabilities": prob_dist.tolist(),
                    "predicted_class": int(predicted_class),
                    "confidence": float(confidence),
                    "entropy": float(entropy),
                    "contribution_score": float(confidence * (1 - entropy/np.log(3)))
                })
            
            # Sort by contribution score
            contributions.sort(key=lambda x: x["contribution_score"], reverse=True)
            
            explanations.update({
                "modality_contributions": contributions,
                "summary": self._generate_fusion_summary(contributions, fusion_probs)
            })
            
        except Exception as e:
            print(f"⚠ Failed to generate fusion explanations: {e}")
            explanations["error"] = str(e)
            
        return explanations
    
    def _generate_physio_summary(self, feature_importance):
        """Generate human-readable summary for physiological explanations"""
        if not feature_importance:
            return "No physiological features available for explanation."
        
        top_features = feature_importance[:3]
        summary_parts = []
        
        for feat in top_features:
            feature_name = feat["feature"]
            importance = feat["importance"]
            
            # Parse sensor and feature type
            sensor = feature_name.split("_")[0]
            feature_type = "_".join(feature_name.split("_")[1:])
            
            direction = "increases" if importance > 0 else "decreases"
            summary_parts.append(f"{sensor} {feature_type} {direction} stress likelihood")
        
        return f"Key factors: {'; '.join(summary_parts[:2])}."
    
    def _generate_dass21_summary(self, feature_importance):
        """Generate human-readable summary for DASS-21 explanations"""
        if not feature_importance:
            return "No questionnaire responses available for explanation."
        
        top_features = feature_importance[:2]
        summary_parts = []
        
        for feat in top_features:
            feature_name = feat["feature"].replace("DASS21_", "").replace("_", " ")
            importance = feat["importance"]
            value = feat["value"]
            
            impact = "increases" if importance > 0 else "decreases"
            summary_parts.append(f"{feature_name} (score: {value:.1f}) {impact} stress")
        
        return f"Main questionnaire factors: {'; '.join(summary_parts)}."
    
    def _generate_fusion_summary(self, contributions, fusion_probs):
        """Generate human-readable summary for fusion explanations"""
        predicted_class = np.argmax(fusion_probs)
        confidence = np.max(fusion_probs)
        class_names = ["Low", "Medium", "High"]
        
        # Find most contributing modality
        top_modality = contributions[0]["modality"]
        
        summary = f"Predicted stress level: {class_names[predicted_class]} " \
                 f"(confidence: {confidence:.2f}). "
        summary += f"Primary evidence from {top_modality} signals."
        
        return summary

# Initialize XAI explainer
xai_explainer = XAIExplainer()

# === Feature Extraction Functions ===
def zscore(x):
    """Z-score normalization with numerical stability"""
    x = np.asarray(x)
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std

def extract_time_features(signal_data):
    """Extract time-domain features from signal"""
    signal_data = np.asarray(signal_data).astype(float)
    
    # Handle edge cases
    if len(signal_data) == 0:
        return [0.0] * 13
    
    # Basic statistics
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data)
    var_val = np.var(signal_data)
    
    # Handle constant signals
    if std_val == 0:
        skew_val = kurtosis_val = 0.0
    else:
        skew_val = skew(signal_data)
        kurtosis_val = kurtosis(signal_data)
    
    return [
        mean_val, std_val, var_val, skew_val, kurtosis_val,
        np.min(signal_data), np.max(signal_data), np.ptp(signal_data),
        np.median(signal_data), np.percentile(signal_data, 25),
        np.percentile(signal_data, 75),
        np.mean(np.abs(np.diff(signal_data))) if len(signal_data) > 1 else 0.0,
        np.sqrt(np.mean(signal_data**2))
    ]

def extract_freq_features(signal_data, fs=100):
    """Extract frequency-domain features using Welch's method"""
    signal_data = np.asarray(signal_data).astype(float)
    
    if len(signal_data) < 8:
        return [0.0] * 11
    
    try:
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)//4))
        bands = {
            'very_low': (0.0, 0.04), 
            'low': (0.04, 0.15), 
            'mid': (0.15, 0.4), 
            'high': (0.4, 0.5)
        }
        
        total_power = np.sum(psd)
        if total_power == 0:
            return [0.0] * 11
        
        features = []
        for low, high in bands.values():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[mask])
            features.append(band_power)
            features.append(band_power / total_power)  # Relative power
        
        # Additional frequency features
        features.extend([
            np.mean(freqs), 
            np.std(freqs), 
            freqs[np.argmax(psd)]  # Peak frequency
        ])
        
        return features
        
    except Exception as e:
        print(f"Warning: Frequency feature extraction failed: {e}")
        return [0.0] * 11

def extract_wavelet_features(signal_data):
    """Extract wavelet-based features"""
    signal_data = np.asarray(signal_data).astype(float)
    
    try:
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
        
        # Pad or truncate to consistent length
        target_length = 20  # 5 levels * 4 features
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
            
        return features
        
    except Exception as e:
        print(f"Warning: Wavelet feature extraction failed: {e}")
        return [0.0] * 20

def extract_ecg_features(signal_data, fs=100):
    """Extract ECG-specific features (heart rate variability)"""
    signal_data = np.asarray(signal_data).astype(float)
    
    try:
        # Find R-peaks
        peaks, _ = signal.find_peaks(
            signal_data, 
            height=np.std(signal_data), 
            distance=fs//3  # Minimum 200ms between peaks
        )
        
        if len(peaks) > 1:
            # Calculate RR intervals in milliseconds
            rr_intervals = np.diff(peaks) / fs * 1000
            
            # HRV features
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
            heart_rate = len(peaks) / (len(signal_data) / fs) * 60
            
            return [mean_rr, std_rr, rmssd, heart_rate]
        else:
            return [0.0] * 4
            
    except Exception as e:
        print(f"Warning: ECG feature extraction failed: {e}")
        return [0.0] * 4

def extract_window_features(sigs):
    """Extract all features for a window of signals"""
    features = []
    
    for sensor in CFG["sensors"]:
        if sensor in sigs:
            data = sigs[sensor]
            
            # Time-domain features
            features.extend(extract_time_features(data))
            
            # Frequency-domain features
            features.extend(extract_freq_features(data))
            
            # Wavelet features
            features.extend(extract_wavelet_features(data))
            
            # ECG-specific features
            if sensor == "ECG":
                features.extend(extract_ecg_features(data))
        else:
            # If sensor data is missing, pad with zeros
            features.extend([0.0] * 13)  # Time features
            features.extend([0.0] * 11)  # Frequency features
            features.extend([0.0] * 20)  # Wavelet features
            if sensor == "ECG":
                features.extend([0.0] * 4)  # ECG features
    
    return np.array(features)

def process_csv_data(csv_buffer):
    """Process CSV data into feature windows"""
    try:
        data = pd.read_csv(csv_buffer)
        
        # Validate required columns
        missing_sensors = [sensor for sensor in CFG["sensors"] if sensor not in data.columns]
        if missing_sensors:
            print(f"Warning: Missing sensors: {missing_sensors}")
        
        window_size = STEP
        stride_size = STRIDE
        all_features = []
        
        for start_idx in range(0, len(data) - window_size + 1, stride_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]
            
            # Extract signals for each sensor
            signals = {}
            for sensor in CFG["sensors"]:
                if sensor in window_data.columns:
                    signals[sensor] = zscore(window_data[sensor].values)
                else:
                    signals[sensor] = np.zeros(window_size)
            
            # Extract features
            features = extract_window_features(signals)
            all_features.append(features)
        
        if not all_features:
            raise ValueError("No features extracted from data. Check data length and format.")
        
        return np.array(all_features)
        
    except Exception as e:
        print(f"Error processing CSV data: {e}")
        raise

def validate_and_parse_dass21(dass21_responses: str):
    """Validate and parse DASS-21 responses with comprehensive error handling"""
    print(f"Raw DASS-21 input: '{dass21_responses}'")
    print(f"Input type: {type(dass21_responses)}")
    
    try:
        # Clean the input
        dass21_responses = dass21_responses.strip()
        
        # Try to parse as JSON first
        try:
            dass21_list = json.loads(dass21_responses)
            print(f"Parsed as JSON: {dass21_list}")
        except json.JSONDecodeError:
            # Try to parse as comma-separated string
            try:
                # Remove brackets if present
                clean_input = dass21_responses.strip("[](){}")
                dass21_list = [float(x.strip()) for x in clean_input.split(",")]
                print(f"Parsed as comma-separated: {dass21_list}")
            except (ValueError, AttributeError) as e:
                print(f"Failed to parse DASS-21 responses: {e}")
                raise ValueError(f"Invalid DASS-21 format. Expected 7 comma-separated numbers or JSON array, got: {dass21_responses}")
        
        # Validate length
        if len(dass21_list) != 7:
            print(f"Invalid DASS-21 length: {len(dass21_list)} (expected 7)")
            raise ValueError(f"DASS-21 must contain exactly 7 values, got {len(dass21_list)}")
        
        # Validate values are in range [0, 3]
        for i, val in enumerate(dass21_list):
            if not isinstance(val, (int, float)) or val < 0 or val > 3:
                print(f"Invalid DASS-21 value at index {i}: {val}")
                raise ValueError(f"DASS-21 values must be between 0 and 3, got {val} at index {i}")
        
        # Convert to floats
        dass21_list = [float(x) for x in dass21_list]
        print(f"Validated DASS-21 responses: {dass21_list}")
        return dass21_list
        
    except Exception as e:
        print(f"DASS-21 validation failed: {e}")
        raise

def validate_and_parse_voice_probs(voice_probs_str: str):
    """Validate and parse voice probabilities"""
    print(f"Raw voice probabilities input: '{voice_probs_str}'")
    print(f"Input type: {type(voice_probs_str)}")
    
    try:
        # Clean the input
        voice_probs_str = voice_probs_str.strip()
        
        # Try to parse as JSON first
        try:
            voice_probs = json.loads(voice_probs_str)
            print(f"Parsed as JSON: {voice_probs}")
        except json.JSONDecodeError:
            # Try to parse as comma-separated string
            try:
                # Remove brackets if present
                clean_input = voice_probs_str.strip("[](){}")
                voice_probs = [float(x.strip()) for x in clean_input.split(",")]
                print(f"Parsed as comma-separated: {voice_probs}")
            except (ValueError, AttributeError) as e:
                print(f"Failed to parse voice probabilities: {e}")
                raise ValueError(f"Invalid voice probabilities format. Expected 3 comma-separated numbers or JSON array, got: {voice_probs_str}")
        
        # Validate length
        if len(voice_probs) != 3:
            print(f"Invalid voice probabilities length: {len(voice_probs)} (expected 3)")
            raise ValueError(f"Voice probabilities must contain exactly 3 values, got {len(voice_probs)}")
        
        # Validate values are probabilities (0-1)
        for i, val in enumerate(voice_probs):
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                print(f"Invalid voice probability value at index {i}: {val}")
                raise ValueError(f"Voice probabilities must be between 0 and 1, got {val} at index {i}")
        
        # Validate probabilities sum to approximately 1
        prob_sum = sum(voice_probs)
        if abs(prob_sum - 1.0) > 0.01:  # Allow small floating point errors
            print(f"Voice probabilities don't sum to 1: {prob_sum}")
            raise ValueError(f"Voice probabilities must sum to 1, got sum: {prob_sum}")
        
        # Convert to floats
        voice_probs = [float(x) for x in voice_probs]
        print(f"Validated voice probabilities: {voice_probs}")
        return voice_probs
        
    except Exception as e:
        print(f"Voice probabilities validation failed: {e}")
        raise

# === Load Models ===
try:
    print("Loading models...")
    physio_model = joblib.load("models/regularized_global_model.pkl")
    dass21_model = joblib.load("models/stacking_classifier_model.pkl")
    dass21_scaler = joblib.load("models/scaler.pkl")
    
    # Updated fusion model to handle voice modality
    fusion_model = PhysioDominantFusion(
        class_weights={0: 0.7, 1: 0.0, 2: 0.3}
    )
    
    print("All models loaded successfully")
    
    # Initialize XAI explainers with some background data
    # Note: In production, you would need actual background data
    dummy_physio_data = np.random.rand(100, len(ALL_FEATURE_NAMES))
    dummy_dass21_data = np.random.rand(100, 7) * 3
    
    xai_explainer.setup_physio_explainer(physio_model, dummy_physio_data)
    xai_explainer.setup_dass21_explainer(dass21_model, dass21_scaler, dummy_dass21_data)
    
except Exception as e:
    print(f"Error loading models: {e}")
    raise



@app.post("/predict")
async def predict(
    physiological_file: UploadFile = File(..., description="CSV file with physiological data"),
    dass21_responses: str = Form(..., description="DASS-21 responses as comma-separated values or JSON array"),
    voice_probabilities: Optional[str] = Form(None, description="Voice probabilities as comma-separated values or JSON array (optional)")
):
    """
    Predict stress level using physiological data, DASS-21 responses, and optional voice probabilities
    
    Args:
        physiological_file: CSV file with columns: ECG, EDA, EMG, Temp
        dass21_responses: 7 values between 0-3, format: "[1,2,0,3,1,2,0]" or "1,2,0,3,1,2,0"
        voice_probabilities: 3 probabilities for [Low, Medium, High] classes, format: "[0.33,0.34,0.33]" or "0.33,0.34,0.33"
    
    Returns:
        JSON with individual model probabilities, fusion results, predictions, and explanations
    """
    print("\n" + "="*50)
    print("🚀 Starting prediction request")
    print("="*50)
    
    try:
        # === Validate File ===
        if not physiological_file.filename.endswith('.csv'):
            raise ValueError("Physiological file must be a CSV file")
        
        # === Process Physiological Data ===
        print("📊 Processing physiological data...")
        file_content = await physiological_file.read()
        buffer = io.StringIO(file_content.decode('utf-8'))
        
        try:
            X_physio = process_csv_data(buffer)
            X_physio = np.nan_to_num(X_physio, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"✅ Physiological data shape: {X_physio.shape}")
            
            if X_physio.shape[0] == 0:
                raise ValueError("No valid windows extracted from physiological data")
                
        except Exception as e:
            print(f"❌ Physiological data processing failed: {e}")
            raise ValueError(f"Failed to process physiological data: {str(e)}")

        # === Physiological Prediction ===
        try:
            physio_probs = physio_model.predict_proba(X_physio)
            # Average across all windows
            physio_probs_avg = physio_probs.mean(axis=0)
            print(f"✅ Physiological probabilities: {physio_probs_avg}")
        except Exception as e:
            print(f"❌ Physiological prediction failed: {e}")
            raise ValueError(f"Physiological model prediction failed: {str(e)}")

        # === Process DASS-21 Data ===
        print("\n📋 Processing DASS-21 responses...")
        try:
            dass21_list = validate_and_parse_dass21(dass21_responses)
            
            # Scale DASS-21 responses
            dass21_X = dass21_scaler.transform([dass21_list])
            dass21_probs = dass21_model.predict_proba(dass21_X)[0]
            print(f"✅ DASS-21 probabilities: {dass21_probs}")
            
        except Exception as e:
            print(f"❌ DASS-21 processing failed: {e}")
            raise ValueError(f"DASS-21 processing failed: {str(e)}")

        # === Process Voice Data (Optional) ===
        voice_probs = None
        if voice_probabilities:
            print("\n🎤 Processing voice probabilities...")
            try:
                voice_probs = validate_and_parse_voice_probs(voice_probabilities)
                voice_probs = np.array(voice_probs)
                print(f"✅ Voice probabilities: {voice_probs}")
            except Exception as e:
                print(f"❌ Voice processing failed: {e}")
                raise ValueError(f"Voice processing failed: {str(e)}")
        else:
            print("\n🎤 No voice probabilities provided, using default uniform distribution")
            voice_probs = np.array([0.33, 0.34, 0.33])  # Default uniform distribution

        # === Fusion ===
        print("\n🔄 Performing fusion...")
        try:
            fusion_input = {
                "phys": physio_probs_avg,
                "text": dass21_probs,
                "voice": voice_probs
            }
            
            fusion_probs = fusion_model.predict_proba(fusion_input)
            fusion_pred = int(fusion_model.predict(fusion_input))
            
            print(f"✅ Fusion probabilities: {fusion_probs}")
            print(f"✅ Fusion prediction: {fusion_pred}")
            
        except Exception as e:
            print(f"❌ Fusion failed: {e}")
            raise ValueError(f"Fusion model failed: {str(e)}")

        # === Explainability ===
        print("\n🔍 Generating explanations...")
        try:
            # Explain physiological prediction
            physio_explanation = xai_explainer.explain_physio_prediction(X_physio)
            
            # Explain DASS-21 prediction
            dass21_explanation = xai_explainer.explain_dass21_prediction(np.array([dass21_list]))
            
            # Explain fusion decision
            fusion_explanation = xai_explainer.explain_fusion_decision(fusion_input, fusion_probs)
            
            print("✅ Explanations generated successfully")
        except Exception as e:
            print(f"⚠ Explanation generation failed: {e}")
            physio_explanation = {"available": False, "error": str(e)}
            dass21_explanation = {"available": False, "error": str(e)}
            fusion_explanation = {"available": False, "error": str(e)}

        # === Prepare Result ===
        result = {
            "success": True,
            "predictions": {
                "physio_probs": physio_probs_avg.tolist(),
                "dass21_probs": dass21_probs.tolist(),
                "voice_probs": voice_probs.tolist() if voice_probabilities else None,
                "fusion_probs": fusion_probs.tolist(),
                "fusion_pred": fusion_pred,
                "prediction_label": ["Low", "Medium", "High"][fusion_pred],
                "confidence": float(np.max(fusion_probs))
            },
            "explanations": {
                "physiological": physio_explanation,
                "questionnaire": dass21_explanation,
                "fusion": fusion_explanation
            },
            "metadata": {
                "physio_windows": X_physio.shape[0],
                "physio_features": X_physio.shape[1],
                "dass21_values": dass21_list,
                "voice_provided": voice_probabilities is not None,
                "modalities_used": ["physiological", "questionnaire", "voice" if voice_probabilities else None]
            }
        }

        # Print result to terminal
        print("\n" + "="*30)
        print("📊 PREDICTION RESULT")
        print("="*30)
        print(json.dumps(result, indent=2))
        print("="*30)

        return JSONResponse(content=result)

    except ValueError as ve:
        error_response = {
            "success": False,
            "error": "Validation Error",
            "message": str(ve),
            "error_type": "validation"
        }
        print(f"❌ Validation Error: {str(ve)}")
        return JSONResponse(content=error_response, status_code=422)
    
    except Exception as e:
        error_response = {
            "success": False,
            "error": "Server Error",
            "message": str(e),
            "error_type": "server"
        }
        print(f"❌ Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content=error_response, status_code=500)

