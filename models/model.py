import tensorflow as tf

def get_model():
    # Create a Sequential model
    model = tf.keras.Sequential([
        # Input layer with the specified input shape
        tf.keras.layers.InputLayer(input_shape=(32, 32, 1)),
        
        # First Convolutional layer with 64 filters, a 2x2 kernel, ReLU activation, and padding
        tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"),
        
        # MaxPooling layer with a 2x2 pool size
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional layer with 128 filters, a 2x2 kernel, ReLU activation, and padding
        tf.keras.layers.Conv2D(128, (2, 2), activation="relu", padding="same"),
        
        # MaxPooling layer with a 2x2 pool size
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional layer with 256 filters, a 2x2 kernel, ReLU activation, and padding
        tf.keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same"),
        
        # MaxPooling layer with a 2x2 pool size
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten layer to convert the 3D feature map to 1D
        tf.keras.layers.Flatten(),
        
        # Dense layer with 256 neurons and ReLU activation
        tf.keras.layers.Dense(256, activation='relu'),
        
        # Dense layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Dense layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Dropout layer with a dropout rate of 20% to reduce overfitting
        tf.keras.layers.Dropout(0.2),
        
        # Output layer with 9 neurons (for 9 classes) and softmax activation for multiclass classification
        tf.keras.layers.Dense(9, activation="softmax")
    ])

    # Compile the model using the Adam optimizer, categorical crossentropy loss, and accuracy metric
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the compiled model
    return model
