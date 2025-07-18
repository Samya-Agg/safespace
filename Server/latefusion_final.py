import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class PhysioDominantFusion(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weights=None):
        self.class_weights = class_weights if class_weights else {0:1.0, 1:1.0, 2:1.0}
        
        self.mod_weights = {'phys':0.60, 'text':0.25, 'voice':0.15}
        self.feature_names = ['phys_low', 'phys_medium', 'phys_high', 
                             'text_low', 'text_medium', 'text_high',
                             'voice_low', 'voice_medium', 'voice_high']
        self.is_fitted_ = False
        
    def fit(self, X, y):
        self.is_fitted_ = True
        return self
        
    def _check_inputs(self, mod_probs):
        
        if not mod_probs:
            raise ValueError("No modality probabilities provided")
        for mod in mod_probs:
            if not isinstance(mod_probs[mod], np.ndarray):
                mod_probs[mod] = np.array(mod_probs[mod])
            if mod_probs[mod].shape != (3,):
                raise ValueError(f"{mod} probabilities must be length 3")

    def _convert_to_feature_vector(self, mod_probs):
        features = np.zeros(9)  # 3 modalities × 3 classes
        
        if 'phys' in mod_probs:
            features[0:3] = mod_probs['phys']
        if 'text' in mod_probs:
            features[3:6] = mod_probs['text']
        if 'voice' in mod_probs:
            features[6:9] = mod_probs['voice']
            
        return features

    def predict_proba(self, mod_probs):
        
        self._check_inputs(mod_probs)
        
       
        weighted = []
        for mod in ['phys', 'text', 'voice']:
            if mod in mod_probs:
                conf = np.max(mod_probs[mod])  
                weighted.append(mod_probs[mod] * conf * self.mod_weights[mod])
        
        if not weighted:
            return np.array([0.33, 0.33, 0.33])  # Neutral if no data
        
        fused = sum(weighted)
        return fused / fused.sum()  # Normalize

    def predict_proba_from_features(self, X):
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        results = []
        for row in X:
            mod_probs = {}
            if np.any(row[0:3] != 0):
                mod_probs['phys'] = row[0:3]
            if np.any(row[3:6] != 0):
                mod_probs['text'] = row[3:6]
            if np.any(row[6:9] != 0):
                mod_probs['voice'] = row[6:9]
            
            if mod_probs:
                proba = self.predict_proba(mod_probs)
            else:
                proba = np.array([0.33, 0.33, 0.33])
            results.append(proba)
        
        return np.array(results)

    def predict(self, X, true_label=None):
        
        if isinstance(X, dict):
            
            proba = self.predict_proba(X)
            if true_label is not None:
                proba = proba * self.class_weights.get(true_label, 1.0)
            return np.argmax(proba)
        else:
            
            probas = self.predict_proba_from_features(X)
            return np.argmax(probas, axis=1)

def load_and_validate():
    
    print("Loading data...")
    try:
        voice = pd.read_csv("softmax_voice.csv")
        phys = pd.read_csv("softmax_wesad.csv")
        text = pd.read_csv("softmax_dass.csv")
        
        
        print("Voice columns:", voice.columns.tolist())
        print("Phys columns:", phys.columns.tolist())
        print("Text columns:", text.columns.tolist())
        
        
        return voice, phys, text
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise

def preprocess(voice, phys, text):
    
    print("Preprocessing...")
    
    rename_map = {
        'prob_Low':'low_prob',
        'prob_Medium':'medium_prob',
        'prob_High':'high_prob',
        'Low':'low_prob',
        'Medium':'medium_prob',
        'High':'high_prob',
        'low':'low_prob',
        'medium':'medium_prob',
        'high':'high_prob',
        'Low_prob':'low_prob',
        'Medium_prob':'medium_prob',
        'High_prob':'high_prob',
        'label':'true_label',
        'Label':'true_label',
        'target':'true_label',
        'Target':'true_label'
    }
    for df in [voice, phys, text]:
        df.rename(columns=rename_map, inplace=True)
        if 'SampleID' not in df.columns:
            df.insert(0, 'SampleID', range(len(df)))
        df['true_label'] = df['true_label'].astype(int)  # Ensure int labels
        df.set_index('SampleID', inplace=True)
    
    
    req_cols = {'low_prob','medium_prob','high_prob','true_label'}
    for df, name in zip([voice,phys,text], ['voice','phys','text']):
        if not req_cols.issubset(df.columns):
            print(f"Missing columns in {name} data. Available: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns in {name} data")
    
    return voice, phys, text


