IIoT IDS Deep Learning Pipeline
This project implements a hybrid CNN-LSTM Deep Learning model for Intrusion Detection Systems (IDS) in Industrial IoT environments. It is a production-ready MLOps pipeline capable of training, evaluating, and running inference on new data.
All results can be found in the "results" folder. The achieved accuracy is almost 100% due to the big and clean X-IIoTID dataset. 

Architecture and Techniques

This pipeline implements a deep learning model designed to extract complex patterns from IIoT network traffic data:
- 1D CNN Layers: Three convolutional layers (with 32, 64, and 128 filters) are used to extract local feature dependencies from the input vector.
- Bidirectional LSTM: Two Bi-LSTM layers (128 units) process the feature sequence to capture long-range dependencies and contextual relationships.
- Classification Head: A series of Dense layers with Batch Normalization and Dropout leads to a Softmax output for multi-class classification.
- Handling Imbalance: The pipeline applies SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples for minority classes, ensuring the model learns to detect rare attack types effectively.
- Robust Regularization: To prevent overfitting, the model employs L2 Regularization, Dropout (0.3), and Batch Normalization after convolutional and dense layers.

Adaptive Learning: The training loop includes ReduceLROnPlateau to lower the learning rate when progress stalls and Early Stopping to automatically terminate training when validation performance peaks.

Preprocessing: All features are scaled to a [0, 1] range using MinMax Scaling, and categorical features are encoded to ensure numerical compatibility.

Setup
1. run pip install -r requirements.txt
2. Prepare Data: Create a parent folder data/raw/ and ensure your dataset (e.g., X-IIoTID_dataset.csv) is placed in data/raw/ or update the path in configs/config.yaml.
3. run python train.py
Output: Best model saved to models/ and preprocessor state saved to data/processed/.
4. run python predict.py --csv data/raw/new_traffic.csv --model models/model_x.h5
Arguments:
--csv: Path to the new data file.
--model: Path to the specific .h5 model file you want to use.
Output: Generates inference_results.csv containing original data + predicted labels and confidence scores.

Note: Batch size must be changed! I used an Nvidia H200 video card for training.
