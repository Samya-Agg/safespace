#!/usr/bin/env python3
"""
Test script to verify all explanations are working
"""

import requests
import json
import pandas as pd
import numpy as np
import io

def create_test_data():
    """Create test physiological data"""
    # Create sample physiological data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'ECG': np.random.normal(0, 1, n_samples) + np.sin(np.linspace(0, 10*np.pi, n_samples)),
        'EDA': np.random.normal(0, 0.5, n_samples) + np.abs(np.sin(np.linspace(0, 5*np.pi, n_samples))),
        'EMG': np.random.normal(0, 0.3, n_samples) + np.abs(np.random.normal(0, 0.2, n_samples)),
        'Temp': np.random.normal(37, 0.5, n_samples) + 0.1 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    }
    
    df = pd.DataFrame(data)
    return df

def test_explanations():
    """Test all explanations"""
    print("🧪 Testing All Explanations")
    print("=" * 50)
    
    # Create test data
    test_df = create_test_data()
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    test_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Test data
    test_data = {
        'physiological_file': ('test_data.csv', csv_content, 'text/csv'),
        'dass21_responses': '[1,2,0,3,1,2,0]',  # Sample DASS-21 responses
        'voice_probabilities': '[0.2,0.5,0.3]'  # Sample voice probabilities
    }
    
    try:
        # Make request to your API
        print("📡 Making API request...")
        response = requests.post(
            'http://localhost:8000/predict',
            files={'physiological_file': test_data['physiological_file']},
            data={
                'dass21_responses': test_data['dass21_responses'],
                'voice_probabilities': test_data['voice_probabilities']
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            
            # Check if all explanations are present
            explanations = result.get('explanations', {})
            
            print("\n📊 EXPLANATION STATUS:")
            print("-" * 30)
            
            # Check Physiological explanations
            physio = explanations.get('physiological', {})
            if physio.get('available', False):
                print("✅ Physiological explanations: AVAILABLE")
                print(f"   Method: {physio.get('method', 'Unknown')}")
                print(f"   Features: {len(physio.get('feature_importance', []))}")
                print(f"   Summary: {physio.get('summary', 'No summary')[:100]}...")
            else:
                print("❌ Physiological explanations: NOT AVAILABLE")
                if 'error' in physio:
                    print(f"   Error: {physio['error']}")
            
            # Check Questionnaire explanations
            questionnaire = explanations.get('questionnaire', {})
            if questionnaire.get('available', False):
                print("✅ Questionnaire explanations: AVAILABLE")
                print(f"   Method: {questionnaire.get('method', 'Unknown')}")
                print(f"   Features: {len(questionnaire.get('feature_importance', []))}")
                print(f"   Summary: {questionnaire.get('summary', 'No summary')[:100]}...")
            else:
                print("❌ Questionnaire explanations: NOT AVAILABLE")
                if 'error' in questionnaire:
                    print(f"   Error: {questionnaire['error']}")
            
            # Check Voice explanations
            voice = explanations.get('voice', {})
            if voice.get('available', False):
                print("✅ Voice explanations: AVAILABLE")
                print(f"   Method: {voice.get('method', 'Unknown')}")
                print(f"   Features: {len(voice.get('feature_importance', []))}")
                print(f"   Summary: {voice.get('summary', 'No summary')[:100]}...")
            else:
                print("❌ Voice explanations: NOT AVAILABLE")
                if 'error' in voice:
                    print(f"   Error: {voice['error']}")
            
            # Check Fusion explanations
            fusion = explanations.get('fusion', {})
            if fusion.get('available', False):
                print("✅ Fusion explanations: AVAILABLE")
                print(f"   Method: {fusion.get('method', 'Unknown')}")
                print(f"   Modalities: {len(fusion.get('modality_contributions', []))}")
                print(f"   Summary: {fusion.get('summary', 'No summary')[:100]}...")
            else:
                print("❌ Fusion explanations: NOT AVAILABLE")
                if 'error' in fusion:
                    print(f"   Error: {fusion['error']}")
            
            # Show prediction results
            predictions = result.get('predictions', {})
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Final Prediction: {predictions.get('prediction_label', 'Unknown')}")
            print(f"   Confidence: {predictions.get('confidence', 0):.2%}")
            print(f"   Fusion Probabilities: {[f'{p:.1%}' for p in predictions.get('fusion_probs', [])]}")
            
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure your server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_explanations() 