import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical

class IIoTPreprocessor:
    def __init__(self, target_col="class2"):
        self.target_col = target_col
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.feature_cols = None

    def fit_transform(self, df, use_smote=True, test_size=0.2):
        # 1. Clean and Target Setup
        df = df.ffill().bfill().fillna(0)
        
        # Drop other class columns to avoid leakage
        cols_to_drop = [c for c in ["class1", "class2", "class3"] if c != self.target_col]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # 2. Handle Categoricals
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in object_cols:
            if col != self.target_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # 3. Features & Target
        X = df.drop(columns=[self.target_col])
        self.feature_cols = X.columns.tolist()
        y = df[self.target_col]

        # 4. Normalize
        X_scaled = self.scaler.fit_transform(X)

        # 5. Encode Target
        y_enc = self.target_encoder.fit_transform(y)
        
        # 6. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=test_size, stratify=y_enc, random_state=42
        )

        # 7. SMOTE
        if use_smote:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # 8. Categorical Conversion
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)

        return X_train, X_test, y_train_cat, y_test_cat

    def transform_new_data(self, df):
        """Used for Inference on new data"""
        df = df.copy()
        df = df.ffill().bfill().fillna(0)
        
        # Handle Categoricals (using fitted encoders)
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels by mapping to '0' or mode, strictly simplistic here
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
        
        # Ensure correct column order
        df = df[self.feature_cols]
        
        # Normalize
        return self.scaler.transform(df)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)