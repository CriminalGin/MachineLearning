import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn; seaborn.set()

# Load the diabetes dataset
boston = datasets.load_boston()

# which feature
i_feature = 0
# Get the feature name
feature_name = boston.feature_names[i_feature]

# Use only one feature
diabetes_X = boston.data[:, np.newaxis, i_feature]

# Split the data into training/testing sets
boston_X_train = diabetes_X[:-20]
boston_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(boston_X_train, boston_y_train)

# Explained variance score: score=1 is perfect prediction
model_score = model.score(boston_X_test, boston_y_test)
