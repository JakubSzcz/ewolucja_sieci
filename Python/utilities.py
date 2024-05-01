import numpy as np
import scipy.io


# constants
ROOM_WIDTH = 5
ROOM_HEIGHT = 5
NUM_CLASSES = 9
SQUARE_WIDTH = ROOM_WIDTH / 3
SQUARE_HEIGHT = ROOM_HEIGHT / 3
DATA_SET_NAME = "../Matlab/people10000_person.mat"
MODEL_NAME_RANDOM_FOREST = '../models/forest/random_forest_estimator.joblib'
MODEL_NAME_RANDOM_NEURAL_NETWORK = ""


# GENERAL PURPOSE FUNCTIONS
def load_data(data_set_name: str) -> (np.array, np.array):
    """
    Loads generated dataset from .mat file

    Parameters
    ----------
    data_set_name : str
      The file location of the .mat file

    Returns
    -------
    tuple
      returns 2 np.arrays: rssi_values_matrix and people_position
    """

    # Load data
    mat = scipy.io.loadmat(data_set_name)
    rssi_values_matrix_complex = np.array(mat['RSSI'], dtype=np.complex_)
    rssi_values_matrix = np.absolute(rssi_values_matrix_complex, dtype=np.double)
    people_position = np.array(mat['positions'], dtype=np.double)[:, :2]
    return rssi_values_matrix, people_position


# RANDOM FOREST FUNCTIONS
def predict_class(rssi_values: np.array, random_forest_model):
    """
    Loads generated dataset from .mat file

    Parameters
    ----------
    rssi_values : nparray
        array with an input rssi absolute values
    random_forest_model :
        Trained RF model on which prediction should take place
    Returns
    -------
    int
      the number of predicted cluster
    """

    # Ensure input is a numpy array
    rssi_values = np.array(rssi_values)
    # Reshape to match the input format expected by the model
    rssi_values = rssi_values.reshape(1, -1)
    # Predict class
    predicted_class = random_forest_model.predict(rssi_values)
    return predicted_class[0]
