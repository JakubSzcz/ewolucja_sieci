import numpy as np
import joblib
import matplotlib.pyplot as plt
import scipy.io

from predict_classification import predict_class

# Load the saved Random Forest model
best_rf = joblib.load('../models/forest/random_forest_estimator.joblib')

# Load data
mat = scipy.io.loadmat("../Matlab/people.mat")
rssi_values_matrix_complex = np.array(mat['RSSI'], dtype=np.complex_)
rssi_values_matrix = np.absolute(rssi_values_matrix_complex, dtype=np.double)
people_position = np.array(mat['positions'], dtype=np.double)[:, :2]

# constants
ROOM_WIDTH = 5
ROOM_HEIGHT = 5
NUM_CLASSES = 9
SQUARE_WIDTH = ROOM_WIDTH / 3
SQUARE_HEIGHT = ROOM_HEIGHT / 3


def slot_on_axis(x, axis_len, slots):
    slot_len = axis_len / slots
    counter = 0
    start = 0
    while 1:
        if x <= start + slot_len:
            break
        else:
            counter += 1
            start = start + slot_len

        if counter > slots:
            raise Exception("Error, slot not found")
    return counter


# translate X,Y cords for class number
def cluster_from_position(x, y):
    cluster_col = slot_on_axis(y, ROOM_WIDTH, NUM_CLASSES ** 0.5)
    cluster_row = slot_on_axis(x, ROOM_HEIGHT, NUM_CLASSES ** 0.5)

    cluster_number = (cluster_row * NUM_CLASSES ** 0.5) + cluster_col

    return int(cluster_number)


# translate list cords for class number
def cluster_from_position_from_list(cords_list):
    cluster_col = slot_on_axis(cords_list[1], ROOM_WIDTH, NUM_CLASSES ** 0.5)
    cluster_row = slot_on_axis(cords_list[0], ROOM_HEIGHT, NUM_CLASSES ** 0.5)

    cluster_number = (cluster_row * NUM_CLASSES ** 0.5) + cluster_col

    return int(cluster_number)


# evaluate the model
people_position_cluster = np.apply_along_axis(cluster_from_position_from_list, axis=1, arr=people_position)
people_position_color = list()

for index, rssi_row in enumerate(rssi_values_matrix):
    predicted_cluster = predict_class(rssi_row)

    if predicted_cluster == people_position_cluster[index]:
        people_position_color.append("g")
    else:
        people_position_color.append("r")

# print results
plt.figure(figsize=(8, 6))

# Plotting room grid
for i in range(1, int(NUM_CLASSES ** 0.5)):
    plt.axvline(x=i * SQUARE_WIDTH, color='k', linestyle='--')
    plt.axhline(y=i * SQUARE_HEIGHT, color='k', linestyle='--')

# Plotting each cluster region with its corresponding color
for i in range(NUM_CLASSES):
    cluster_row = i % int(NUM_CLASSES ** 0.5)
    cluster_col = i // int(NUM_CLASSES ** 0.5)
    plt.fill_between([cluster_col * SQUARE_WIDTH, (cluster_col + 1) * SQUARE_WIDTH],
                     cluster_row * SQUARE_HEIGHT, (cluster_row + 1) * SQUARE_HEIGHT,
                     color=f'C{i}', alpha=0.3)

# Plotting people's positions
for index, position in enumerate(people_position):
    plt.scatter(position[0], position[1], color=people_position_color[index])

plt.title('How well model predicted output for each cluster (green- valid prediction)')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.xlim(0, ROOM_WIDTH)
plt.ylim(0, ROOM_HEIGHT)
plt.grid(True)

# Creating custom legend
legend_handles = []
for label in range(NUM_CLASSES):
    legend_handles.append(plt.scatter([], [], color=f'C{label}', marker='o', s=100, label=f'Class {label}'))
plt.legend(handles=legend_handles, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('model_evaluation.png')
plt.close()

# print statistic
print("####################################\nSTATISTICS\n####################################")
print("\tTotal modal accuracy: " +
      str(round(100 * people_position_color.count("g") / len(people_position_color), 2)) + "%.")

# Initialize counts for green predictions and total counts for each label
labels_green_count = dict.fromkeys(range(NUM_CLASSES), 0)
labels_all_count = dict.fromkeys(range(NUM_CLASSES), 0)

# Count green predictions and total counts for each label
for index, cluster in enumerate(people_position_cluster):
    labels_all_count[cluster] += 1
    if people_position_color[index] == "g":
        labels_green_count[cluster] += 1

# Calculate accuracy for each cluster
for label in range(NUM_CLASSES):
    accuracy = (labels_green_count[label] / labels_all_count[label]) * 100 if labels_all_count[label] != 0 else 0
    print("\tAccuracy for cluster " + str(label) + ": " + str(round(accuracy, 2)) + "%.")

