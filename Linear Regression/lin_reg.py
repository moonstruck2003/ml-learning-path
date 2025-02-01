import numpy as np

# Linear Regression
class Linear_Regression:
    # We are giving an instance of the model, so we have to give the instance as "self"
    # learning_rate, no_of_iterations are hyperparameters
    # weight and bias are model parameters, we can't control them directly

    # Initiating the parameters (learning_rate, no_of_iterations):
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # X = features, y = target
    def fit(self, X, y):
        # Number of training examples and number of features
        # m = number of training examples (rows), n = number of features (columns)
        self.m, self.n = X.shape  # Number of rows and columns

        # Initiating the weight and bias
        self.w = np.zeros(self.n)  # Initialize weights to zeros
        self.b = 0  # A model has only one bias, so initialize to 0

        # Store features and target
        self.X = X
        self.y = y  # Use lowercase 'y' consistently

        # Implementing Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        # Predict the target values
        Y_prediction = self.predict(self.X)

        # Calculate the gradient
        dw = -(2 * (self.X.T).dot(self.y - Y_prediction)) / self.m  # Gradient for weights
        db = -(2 * np.sum(self.y - Y_prediction)) / self.m  # Gradient for bias

        # Updating the weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        # Predict the target values using the linear equation
        return X.dot(self.w) + self.b