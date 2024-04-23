import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
mat = scipy.io.loadmat("../Matlab/people10000_person.mat")
rssi_values_matrix_complex = np.array(mat['RSSI'], dtype=np.complex_)
rssi_values_matrix = np.absolute(rssi_values_matrix_complex, dtype=np.double)
people_position = np.array(mat['positions'], dtype=np.double)[:, :2]

# Define room dimensions
room_width = 5
room_height = 5

# Define number of classes
num_classes = 9

# Calculate square width and height
square_width = room_width / 3
square_height = room_height / 3

# Calculate row and column indices for each position
row_indices = np.floor(people_position[:, 0] / square_width).astype(int)
col_indices = np.floor(people_position[:, 1] / square_height).astype(int)

# Create class labels based on row and column indices
class_labels = row_indices * 3 + col_indices

# Plotting the room with class labels and people's positions
plt.figure(figsize=(8, 6))

# Plotting room grid
for i in range(1, 3):
    plt.axvline(x=i * square_width, color='k', linestyle='--')
    plt.axhline(y=i * square_height, color='k', linestyle='--')

# Plotting all 9 classes (even if some are not present in the first 100 positions)
for label in range(num_classes):
    plt.scatter([], [], label=f'Class {label}', color=f'C{label}', marker='o', s=100)

# Plotting people's positions
for position, label in zip(people_position[0:100], class_labels[0:100]):
    plt.scatter(position[0], position[1], color=f'C{label}')

# Adding legend and labels
# Adding legend outside the plot
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Room with 9 Classes and People\'s Positions (First 100)')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.xlim(0, room_width)
plt.ylim(0, room_height)
plt.grid(True)

# Save the plot as an image file
plt.tight_layout()
plt.savefig('room_with_classes_and_positions_first_100.png')

# Close the plot
plt.close()

# Flatten the RSSI values matrix
rssi_values_flat = rssi_values_matrix.reshape(rssi_values_matrix.shape[0], -1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(rssi_values_flat, class_labels, test_size=0.2, random_state=42)

## Initialize the Random Forest classifier
clf_rf = RandomForestClassifier(random_state=42)

# Define hyperparameters grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8]
}

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=clf_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best estimator from grid search
best_rf = grid_search.best_estimator_

# Make predictions on the testing set
y_pred_rf = best_rf.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Save the best Random Forest estimator to a file
joblib.dump(best_rf, '../models/forest/random_forest_estimator.joblib')