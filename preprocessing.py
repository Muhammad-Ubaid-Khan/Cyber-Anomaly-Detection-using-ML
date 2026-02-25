import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Starting CYBER ANOMALY DETECTION USING MACHINE LEARNING...\n")

file_path = "KDDTrain_filtered.csv" 

try:
    print("[1/4] Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False) 
    
    if 'difficulty_level' in df.columns:
        df = df.drop('difficulty_level', axis=1)
        
    print("[2/4] Converting target labels to 0 (Normal) and 1 (Attack)...")
    # Make sure we convert everything to string first just in case there are mixed types
    df['label'] = df['label'].astype(str).apply(lambda x: 0 if x == 'normal' else 1)
    
    print("[3/4] Encoding remaining text columns to numbers...")
    encoder = LabelEncoder()
    text_cols = ['protocol_type', 'service', 'flag']
    for col in text_cols:
        if df[col].dtype == 'object' or df[col].dtype == 'O':
            # Convert to string first to prevent mixed-type errors during encoding
            df[col] = encoder.fit_transform(df[col].astype(str))
            
    # --- FINAL PHASE: FEATURE SCALING ---
    print("[4/4] Scaling features so large numbers don't dominate small numbers...")
    
    # Separate the Features (X) from the Target Label (y)
    y = df['label'] 
    X = df.drop('label', axis=1) 
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Scale all the features
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Combine the scaled features and the label back into one clean dataset
    df_clean = pd.concat([X_scaled, y], axis=1)
    
    print("\n--- Before Scaling vs After Scaling ---")
    print("Notice how 'src_bytes' values are now small, manageable numbers:")
    print(df_clean[['src_bytes', 'dst_bytes', 'protocol_type']].head())
    
    # Save the cleaned dataset to a new file for your team to use
    output_filename = "KDDTrain_Cleaned.csv"
    df_clean.to_csv(output_filename, index=False)
    
    print(f"\nSUCCESS! Phase 1 Complete.")
    print(f"Your fully preprocessed data has been saved as '{output_filename}'.")

except Exception as e:
    print(f"An error occurred: {e}")