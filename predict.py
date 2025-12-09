import pickle
import pandas as pd
import numpy as np
import yaml
import argparse
from tensorflow.keras.models import load_model

def load_pipeline(config_path, model_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Load Preprocessor
    with open(cfg['data']['artifacts_path'], 'rb') as f:
        preprocessor = pickle.load(f)
        
    # Load Model
    model = load_model(model_path)
    
    return preprocessor, model

def predict_from_csv(csv_path, model_path):
    preprocessor, model = load_pipeline("configs/config.yaml", model_path)
    
    # Load New Data
    new_data = pd.read_csv(csv_path)
    print(f"Loaded {len(new_data)} rows for inference.")
    
    # Transform (Scaling/Encoding based on training state)
    X_new = preprocessor.transform_new_data(new_data)
    
    # Predict
    probs = model.predict(X_new)
    preds_idx = np.argmax(probs, axis=1)
    
    # Decode labels back to strings (e.g., "Attack", "Normal")
    pred_labels = preprocessor.target_encoder.inverse_transform(preds_idx)
    
    # Save results
    results = new_data.copy()
    results['Predicted_Label'] = pred_labels
    results['Confidence'] = np.max(probs, axis=1)
    
    output_file = "inference_results.csv"
    results.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to new data CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 model file")
    args = parser.parse_args()
    
    predict_from_csv(args.csv, args.model)