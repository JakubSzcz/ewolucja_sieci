import sys
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('../models/model_1M.keras')


def preprocess_input(args):
    """
    Preprocess input arguments by normalizing and applying any required transformations.

    Args:
    - args: A list of input arguments provided via the command line.

    Returns:
    - processed_input: A numpy array containing the preprocessed input data.
    """
    # Perform any required preprocessing here (e.g., date regularization)
    # Placeholder date regularization (example only, replace with your implementation)
    processed_input = [float(arg) for arg in args]  # Convert input arguments to float

    return np.array(processed_input).reshape(1, -1)  # Reshape to (1, 8) array


def predict_position(input_data):
    """
    Predict position using the trained model.

    Args:
    - input_data: A numpy array of shape (1, 8) containing the input data.

    Returns:
    - predicted_position: A numpy array containing the predicted x and y coordinates.
    """
    predicted_position = model.predict(input_data)
    return predicted_position


if __name__ == "__main__":
    # Check if correct number of arguments are provided
    if len(sys.argv) != 9:
        print("Usage: python predict_position.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8")
        sys.exit(1)

    # Preprocess input arguments
    input_data = preprocess_input(sys.argv[1:])

    # Predict position
    predicted_position = predict_position(input_data)

    print("Predicted Position:", predicted_position)
