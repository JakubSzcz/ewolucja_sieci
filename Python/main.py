import os
import numpy as np
import scipy.io
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# load data
dir_path = os.getcwd()
# mat = scipy.io.loadmat(fr"{dir_path}/Matlab/people10000_person.mat")
mat = scipy.io.loadmat("../Matlab/people10000_person.mat")
rssi_values_matrix_complex = np.array(mat['RSSI'], dtype=np.complex_)
rssi_values_matrix_real = np.absolute(rssi_values_matrix_complex, dtype=np.double)

people_position = np.array(mat['positions'], dtype=np.double)[:, :2]

# Normalize input data
scaler = StandardScaler()
rssi_values_matrix_normalized = scaler.fit_transform(rssi_values_matrix_real)

X_train, X_val, y_train, y_val = train_test_split(rssi_values_matrix_normalized, people_position, test_size=0.2,
                                                  random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(2)  # Output layer with 2 units for x and y coordinates
])

model.summary()


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',  # Assuming it's a regression problem
              metrics=['accuracy'])

# Define early stopping to prevent overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model using early stopping and validation data
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1,
                    validation_data=(X_val, y_val))

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Predict on validation set
y_pred = model.predict(X_val)

# Calculate Mean Squared Error
mse = tf.keras.losses.mean_squared_error(y_val, y_pred).numpy()
print("Mean Squared Error:", mse)

# Calculate Mean Absolute Error
mae = tf.keras.losses.mean_absolute_error(y_val, y_pred).numpy()
print("Mean Absolute Error:", mae)

# Save the trained model using the native Keras format
model.save('../models/model.keras')
