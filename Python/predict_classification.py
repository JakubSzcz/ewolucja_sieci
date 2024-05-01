import joblib
import matplotlib.pyplot as plt

from utilities import *

# Load the saved Random Forest model
model = joblib.load(MODEL_NAME_RANDOM_FOREST)


def get_class(predicted_class):
    if predicted_class == 0:
        return 2, 0
    if predicted_class == 1:
        return 1, 0
    if predicted_class == 2:
        return 0, 0
    if predicted_class == 3:
        return 2, 1
    if predicted_class == 4:
        return 1, 1
    if predicted_class == 5:
        return 0, 1
    if predicted_class == 6:
        return 2, 2
    if predicted_class == 7:
        return 1, 2
    if predicted_class == 8:
        return 0, 2


# Function to plot the room with highlighted predicted class and save as an image file
def save_room_plot(predicted_class, filename):
    # Create a grid representing the room
    room_grid = np.zeros((3, 3))
    print(predicted_class)
    i, j = get_class(predicted_class)
    room_grid[i, j] = 1

    # Plot the room grid
    plt.figure(figsize=(8, 6))
    plt.imshow(room_grid, cmap='Blues', extent=[0, ROOM_WIDTH, 0, ROOM_HEIGHT], alpha=0.5, vmin=0, vmax=1)

    # Plot grid lines
    for i in range(1, 3):
        plt.axvline(i * SQUARE_WIDTH, color='k', linestyle='--')
        plt.axhline(i * SQUARE_HEIGHT, color='k', linestyle='--')

    # Calculate text position
    text_y = (predicted_class % 3 + 0.5) * SQUARE_WIDTH
    text_x = (predicted_class // 3 + 0.5) * SQUARE_HEIGHT  # Invert the row index to match plot coordinates
    # Plot class label at the center of the highlighted square
    plt.text(text_x, text_y, f'Class {predicted_class}', ha='center', va='center', fontsize=12, color='red')

    # Add title and labels
    plt.title('Room with Highlighted Predicted Class')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.xlim(0, ROOM_WIDTH)
    plt.ylim(0, ROOM_HEIGHT)
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(filename)
    plt.close()


# Function to predict top 3 classes and their probabilities
def predict_top3_classes(input_data):
    # Make prediction probabilities
    predicted_probs = model.predict_proba([input_data])[0]

    # Get indices of top 3 classes based on predicted probabilities
    top3_indices = np.argsort(predicted_probs)[-3:][::-1]

    # Get top 3 classes and their probabilities
    top3_classes = top3_indices
    top3_probs = predicted_probs[top3_indices]

    return top3_classes, top3_probs


# Example usage
if __name__ == "__main__":
    # Wait for input of 8 RSSI values
    rssi_input = input("Enter 8 RSSI values separated by space: ")
    rssi_values = list(map(float, rssi_input.split()))

    if len(rssi_values) != 8:
        print("Error: Input should contain exactly 8 values.")
    else:
        try:
            # Convert input values to floats
            rssi_values = list(map(float, rssi_values))
        except ValueError:
            print("Error: Input values must be numeric.")

    # Predict class
    predicted_class_number = predict_class(rssi_values, model)
    print("Predicted class:", predicted_class_number)
    save_room_plot(predicted_class_number, 'predicted_room_plot.png')
