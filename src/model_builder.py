import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers

def build_cnn_lstm(input_dim, num_classes, config):
    l2 = regularizers.l2(config['l2_reg'])
    
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),

        # CNN Block
        layers.Conv1D(config['cnn_filters'][0], 3, activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Dropout(config['dropout']),
        
        layers.Conv1D(config['cnn_filters'][1], 3, activation='relu', padding='same', kernel_regularizer=l2),
        layers.MaxPooling1D(2),
        layers.Dropout(config['dropout']),

        # LSTM Block
        layers.Bidirectional(layers.LSTM(config['lstm_units'], return_sequences=True, kernel_regularizer=l2)),
        layers.Dropout(config['dropout']),
        layers.Bidirectional(layers.LSTM(config['lstm_units'], kernel_regularizer=l2)),
        
        # Output
        layers.Dense(64, activation='relu', kernel_regularizer=l2),
        layers.Dropout(config['dropout']),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model