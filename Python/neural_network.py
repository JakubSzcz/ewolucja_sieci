import os
import numpy as np
import scipy.io
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# load data
dir_path = os.getcwd()
# mat = scipy.io.loadmat(fr"{dir_path}/Matlab/people10000_person.mat")
mat = scipy.io.loadmat("../Matlab/people1000000_person.mat")
checkpoint_filepath = '../models/neuralNetwork/checkpoint.model_1M.keras'

# extract data
rssi_values_matrix_complex = np.array(mat['RSSI'], dtype=np.complex_)
rssi_values_matrix_real = np.absolute(rssi_values_matrix_complex, dtype=np.double)
people_position = np.array(mat['positions'], dtype=np.double)[:, :2]

# normalize data
scaler = StandardScaler()
rssi_values_matrix_normalized = scaler.fit_transform(rssi_values_matrix_real)

# split sets for training and validation
X_train, X_val, y_train, y_val = train_test_split(rssi_values_matrix_normalized, people_position, test_size=0.2,
                                                  random_state=42)

# define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(2)
])

# print model summary
model.summary()

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['accuracy'])

# define early stopping to prevent overfitting
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

# define model checkpoint
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True)

# train the model
history = model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=1,
                    validation_data=(X_val, y_val), callbacks=[early_stopping_callback, model_checkpoint_callback])

# evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# predict on validation set
y_pred = model.predict(X_val)

# Mean Squared Error
mse = tf.keras.losses.mean_squared_error(y_val, y_pred).numpy()
print("Mean Squared Error:", mse)

# Mean Absolute Error
mae = tf.keras.losses.mean_absolute_error(y_val, y_pred).numpy()
print("Mean Absolute Error:", mae)

# save model
model.save('../models/neuralNetwork/model_1M.keras')
