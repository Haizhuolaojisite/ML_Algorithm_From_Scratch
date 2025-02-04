# This k-Nearest Neighbors tutorial is broken down into 3 parts:
#
# Step 1: Calculate Euclidean Distance.
# Step 2: Get Nearest Neighbors.
# Step 3: Make Predictions.
# calculate the Euclidean distance between two vectors
from math import sqrt


def euclidean_distance(row1, row2):
    '''
    calculate the Euclidean distance between two vectors
    :param row1: e.g. [2.7810836,2.550537003,0]
    :param row2: e.g. [7.627531214,2.759262235,1]
    :return: float
    '''
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    # each train/data row records the last element as its label
    output_values = [row[-1] for row in neighbors]
    # finds the label that appears most frequently among the neighbors
    # The max function is used to find the element in the set that has
    # the highest count in the original output_values list.
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Test distance function
dataset = [[2.7810836,2.550537003,0],
 [1.465489372,2.362125076,0],
 [3.396561688,4.400293529,0],
 [1.38807019,1.850220317,0],
 [3.06407232,3.005305973,0],
 [7.627531214,2.759262235,1],
 [5.332441248,2.088626775,1],
 [6.922596716,1.77106367,1],
 [8.675418651,-0.242068655,1],
 [7.673756466,3.508563011,1]]
prediction = predict_classification(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))