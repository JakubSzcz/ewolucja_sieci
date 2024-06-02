import numpy as np
import scipy.io

# constants
ROOM_WIDTH = 5  # width of room
ROOM_HEIGHT = 5  # height of room
NUM_CLASSES = 9  # number of squares that room was divided into, must be equal to the square of the natural number
SQUARE_WIDTH = ROOM_WIDTH / (NUM_CLASSES ** 0.5)  # width of single square
SQUARE_HEIGHT = ROOM_HEIGHT / (NUM_CLASSES ** 0.5)  # height of single square
NUMBER_OF_REFLECTIONS = 3  # reflection coefficient
DATA_SIZE_TO_LOAD = None  # size of data to be loaded from .mat file, if None, load the whole file
LEGEND_ON = False  # flag to print legend on the plot
DATA_SET_TYPE = 'train'  # the type of data to be loaded, possible values: "train", "valid",
KNN_NEIGH = 10
# "furniture_train" and "furniture_valid"


# models paths to change
RF_MODEL_FURNITURE = f"../models/forest/furniturev2.joblib"
RF_MODEL_PATH_K_VAR = f"../models/forest/RF_n=3k,k=var/RF_k={NUMBER_OF_REFLECTIONS}_n=3k.joblib"
RF_MODEL_PATH_CLASS_VAR = f"../models/forest/RF_n=3k,k=3,class=var/RF_class={NUM_CLASSES}_n=3k_k=3.joblib"
MODEL_NAME_KNN = f"../models/knn/knn{KNN_NEIGH}.joblib"

DATA_SET_NAME = f"../dataSets/{DATA_SET_TYPE}/k={NUMBER_OF_REFLECTIONS},n=3k.mat"

# Variables to import DO NOT CHANGE
MODEL_NAME_RANDOM_FOREST = MODEL_NAME_KNN
MODEL_NAME_RANDOM_NEURAL_NETWORK = ""


# GENERAL PURPOSE FUNCTIONS
def load_data(data_set_name: str, data_size: int = DATA_SIZE_TO_LOAD) -> (np.array, np.array):
    """
    Loads generated dataset from .mat file

    Parameters
    ----------
    data_set_name : str
      The file location of the .mat file
    data_size : int = None
      Number of data records to load

    Returns
    -------
    tuple
      returns 2 np.arrays: rssi_values_matrix and people_position
    """

    # Load data
    mat = scipy.io.loadmat(data_set_name)
    rssi_values_matrix_complex = np.array(mat['RSSI'], dtype=np.complex_)

    if data_size:
        assert data_size > 0, "Data size must be greater than 0"
        assert data_size < rssi_values_matrix_complex.shape[0], "Data size must be smaller than rssi values shape"
        rssi_values_matrix = np.absolute(rssi_values_matrix_complex, dtype=np.double)[0:data_size, :]
        people_position = np.array(mat['positions'], dtype=np.double)[0:data_size, :2]
        print(f"[LOG] {str(data_size)} data records loaded.")
    else:
        rssi_values_matrix = np.absolute(rssi_values_matrix_complex, dtype=np.double)
        people_position = np.array(mat['positions'], dtype=np.double)[:, :2]

    return rssi_values_matrix, people_position


# RANDOM FOREST FUNCTIONS
def predict_class(rssi_values: np.array, random_forest_model):
    """
    Predicts cluster number based on the provided rssi_values

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
