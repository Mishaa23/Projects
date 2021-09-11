#Code for a simple neural network with 3 layers, with 2,3,2
#nodes respectively. The aim is to given a set of red and  blue
#points, determine whether a new point is red or blue, based
#on its coordinates.
#The Theory & algorithm implemented here was taken from
# Deep Learning: An Introduction for Applied Mathematicians Catherine by F. Higham & Desmond J. Higham

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#defining the activation funciton, and then its derivative
def sigmoid(a, W, b):
    z = W.dot(a.T).T + b
    return 1/(1+np.exp(-z))

#assume input is a dictionary with weights and biases of the network
def loss(network, N, X, y):
    difference = np.array([])
    for i in range(N):
        a_2 = sigmoid(X[i], network['W_2'], network['b_2'])
        a_3 = sigmoid(a_2, network['W_3'], network['b_3'])
        difference = np.append(difference, LA.norm(y[i] - a_3))
    loss = (1/(2*N))*(LA.norm(difference)**2)
    return loss

def print_points(X, y):
    colours = []
    for i in range(len(X)):
        if y[i,0] == 1:
            colours.append('red')
        else:
            colours.append('blue')
        
    x_coord, y_coord = zip(*X)
    plt.scatter(x_coord, y_coord, color = colours)
    plt.title('Points:')
    plt.show()
    return None

#red and blue points on a page, first 6 are red, last 3 are blue
X = np.array([[1,1],[2,1],[2,2],
                  [1,2],[2,4], [3,4],
                  [3,2], [4,2], [3,1]])

#no. samples
N = len(X) 

#colour of points, [1,0] means red point, [0,1] means blue point
#1st coordinate: how red, 2nd coordinate: how blue 
y = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])

#display the entered points:
print_points(X, y)
print('\n')

#setting random weights and biases
np.random.seed(100)
W = np.array([[1,1], [2,2]])
W_2 = np.random.rand(3,2)
W_3 = np.random.rand(2,3)
b_2= np.random.rand(1,3)
b_3 = np.random.rand(1,2)
network = {'W_2' : W_2, 'W_3': W_3, 'b_2' : b_2, 'b_3' : b_3}
rate = 0.05 # step length, or 'learning rate'
iterations = 1000000

for i in range(iterations):
    j = np.random.randint(8, size = 1)
    #forward pass, computing all layers
    a_2 = sigmoid(X[j], network['W_2'], network['b_2'])
    a_3 = sigmoid(a_2, network['W_3'], network['b_3'])
    #backward pass, computing dC/dz_{j}, where z_{j} is the jth layer
    delta_3 = (a_3*(np.ones(len(a_3)) - a_3))*(a_3 - y[j])
    delta_2 = (a_2*(np.ones(len(a_2)) - a_2))*((network['W_3'].T).dot(delta_3.T)).T
    #gradient descent step
    network['W_3'] = network['W_3'] - rate*((delta_3.T).dot(a_2))
    network['W_2'] = network['W_2'] - rate*((delta_2.T).dot(X[j]))
    network['b_3'] = network['b_3'] - rate*delta_3
    network['b_2'] = network['b_2'] - rate*delta_2
    if i%50000 == 0:
        print("Loss value:", loss(network, N, X, y))
        
#predicting  a colour for a point:
print('\n')
point = np.array([[2.7,1.5]])
print("Chosen point is:", point)
second_layer = sigmoid(point, network['W_2'], network['b_2'])
output = sigmoid(second_layer, network['W_3'], network['b_3'])
print('Value as predicted by network:', output)
new_pred = []
if output[0,0] > output[0,1]:
    print("Chosen point is red")
    new_pred = np.array([[1,0]])
elif output[0,0] < output[0,1]:
    print("Chosen point is blue")
    new_pred = np.array([[0,1]])

X_new = np.append(X, point, axis = 0)
y_new = np.append(y, new_pred, axis = 0)   
print_points(X_new, y_new)
