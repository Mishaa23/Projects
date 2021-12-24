# An attempt at implementing stochastic gradient descent for logistic regression.
# Note that here, the loss function is the negative of the log-likelihood function 
# (assuming that each sample was drawn from a bernoulli distribution
# X ~ Ber(p), where p = sigmoid(w^Tx + b))
#
# Aim is to implement stochastic gradient descent to fit a model. At the end, the model
# is used applied to the pima diabetes dataset, although this is done briefly.

import numpy as np
import pandas as pd
import random as rd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.preprocessing import Normalizer

class logistic_regression:
    def __init__(self, X, y, learn_rate, seed):
        self.X = X 
        self.y = y
        #initialising weights and biases to fixed values
        w_b_seed = rd.seed(1000)
        self.w = np.array([rd.random() for _ in range(X.shape[1])])
        self.b = np.array([rd.random()])
        self.w_best = self.w
        self.b_best = self.b
        self.learn_rate = learn_rate
        self.seed = seed

    def loss(self, y_pred):
        samples = self.X.shape[0]
        return (1/samples)*log_loss(self.y, y_pred)
    
    def sigmoid(self, x, w, b):
        z = x.dot(w) + b
        return 1/(1 + np.exp(-z))

    def train_model(self, iterations):
        #ensuring we get the same training point shuffle sequence each time
        shuffle_seed = np.random.RandomState(self.seed)
        no_points  = self.X.shape[0]
        shuffle_sequence = shuffle_seed.randint(0, no_points, iterations)
        counter = 0
        for point in shuffle_sequence:
            #choose a random training point to update the gradients
            j = point
            #update the weights and biases
            self.w = self.w - (self.learn_rate)*((self.sigmoid(self.X[j], self.w, self.b) - self.y[j])*self.X[j]) 
            self.b = self.b - (self.learn_rate)*(self.sigmoid(self.X[j], self.w, self.b) - self.y[j]) 
            if counter % 200 == 0:
                print(f'Loss for iteration {counter} is:', self.loss(self.sigmoid(self.X, self.w, self.b)))
            counter += 1
    
    def predict_class(self, threshold, input):
        z = self.sigmoid(input, self.w, self.b)
        predictions = np.where(z >= threshold, 1, 0)
        return predictions

def main():
    diabetes = pd.read_csv('/Users/Misha/Desktop/self learning/log_reg_stuff/diabetes.csv', sep = ',')
    diabetes = diabetes.values
    y = diabetes[:, -1]
    X = diabetes[:, :-1]
    normaliser = Normalizer().fit(X)
    X = normaliser.transform(X)
    model = logistic_regression(X, y, 0.01, 42)
    model.train_model(10000)

    #the value of 0.3925 was obtained by trial and error
    predictions = model.predict_class(0.3925, X)
    print("Scratch model prediction: ")
    print(confusion_matrix(y, predictions))

    model_2 = LogisticRegression()
    model_2.fit(X, y)
    preds = model_2.predict(X)
    print("Sklearn model prediction: ")
    print(confusion_matrix(y, preds))



    return None
main()

