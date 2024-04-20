import numpy as np
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
mat = scipy.io.loadmat("../Matlab/results.mat")
rssi_values_matrix = np.array(mat['RSSI'], dtype=np.double)
people_position = np.array(mat['pepolePosMatrix'], dtype=np.double)[:, :2]

# implement Keras model
model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(rssi_values_matrix, people_position, epochs=150, batch_size=1, verbose=2)

# evaluate
loss, accuracy = model.evaluate(rssi_values_matrix, people_position, verbose=0)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
