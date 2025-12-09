import yaml
import os
import pandas as pd
import tensorflow as tf
from src.data_loader import IIoTPreprocessor
from src.model_builder import build_cnn_lstm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime

# 1. Load Config
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 2. Prepare Data
print(f"Loading data from {cfg['data']['raw_path']}...")
df = pd.read_csv(cfg['data']['raw_path'])

preprocessor = IIoTPreprocessor(target_col=cfg['data']['target_col'])
X_train, X_test, y_train, y_test = preprocessor.fit_transform(
    df, 
    use_smote=cfg['data']['use_smote'],
    test_size=cfg['data']['test_size']
)

# Save Preprocessor for Inference
os.makedirs(os.path.dirname(cfg['data']['artifacts_path']), exist_ok=True)
preprocessor.save(cfg['data']['artifacts_path'])
print("✓ Preprocessor saved.")

# 3. Build Model
input_dim = X_train.shape[1]
num_classes = y_train.shape[1]

model = build_cnn_lstm(input_dim, num_classes, cfg['model'])
model.summary()

# 4. Callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(cfg['training']['log_dir'], run_id)
model_path = os.path.join(cfg['training']['model_save_path'], f"model_{run_id}.h5")

callbacks = [
    EarlyStopping(patience=cfg['training']['patience'], restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy'),
    TensorBoard(log_dir=log_path)
]

# 5. Train
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=cfg['training']['epochs'],
    batch_size=cfg['training']['batch_size'],
    callbacks=callbacks,
    verbose=1
)

print(f"Pipeline Complete. Model saved to {model_path}")