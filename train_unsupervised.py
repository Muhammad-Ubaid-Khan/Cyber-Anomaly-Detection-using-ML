import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

print("--- TRAINING UNSUPERVISED MODEL (ISOLATION FOREST) ---")

file_path = "KDDTrain_Cleaned.csv"

try:
    # 1. Load the data
    print("Loading cleaned dataset...")
    df = pd.read_csv(file_path)
    
    # 2. For unsupervised learning, we hide the labels from the AI
    X = df.drop('label', axis=1)
    
    # 3. Initialize Isolation Forest
    # contamination=0.2 means we estimate roughly 20% of the traffic might be anomalous
    print("Setting up Isolation Forest (Unsupervised)...")
    iso_model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42, n_jobs=-1)
    
    # 4. Train the model (Notice we ONLY pass X, we don't pass the y labels!)
    print("Training the AI to find hidden anomalies... (Please wait)")
    iso_model.fit(X)
    print("Training complete!")
    
    # 5. Save the model for Streamlit
    joblib.dump(iso_model, 'iso_forest_model.joblib')
    print("\nSUCCESS! Saved the unsupervised model as 'iso_forest_model.joblib'")

except Exception as e:
    print(f"Error: {e}")