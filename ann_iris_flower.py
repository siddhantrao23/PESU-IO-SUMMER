#taking the dataset of types of iris flowers to train our ANN and perform some predictions based on this training data
"""Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Virginica"""

import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1
iterations = 10000


class NeuralNetwork:
	def __init__(self, x, y):
		self.input = x
		self.weights = np.random.rand(self.input.shape[1],1)	#hidden layer
		self.bias = np.random.rand(self.input.shape[0],1)
		self.y = y
		self.output = np.zeros(self.y.shape)

	def feedforward(self):
		self.output = sigmoid(np.dot(self.input, self.weights) + self.bias)

	def cost(self):
		squared_err = np.square(self.output - self.y)
		return squared_err

	def backprop(self):
		d_weights = np.dot(self.input.T, (2*(self.y - self.output) * deriv(self.output)))
		d_bias = 2*(self.y - self.output) * deriv(self.output)
		self.weights += learning_rate * d_weights
		self.bias += learning_rate * d_bias

	def train(self):
		for _ in range(iterations):
			self.feedforward()
			self.cost()
			self.backprop()

	def predict(self, arr):
		return np.dot(arr, self.weights)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def deriv(z):
	return sigmoid(z) * (1 - sigmoid(z))

def find_var(string):		#returns 0 or 1 depending on type of flower
	if(string == 'Iris-setosa\n'):
		return 0;
	return 1;


def main():
	train_data = []
	target = []
	with open("iris.data", "r") as f:
		for line in f:
			if(line != '\n'):
				temp_data = line.split(',')
				temp_data[0:4] = [float(x) for x in temp_data[0:4]]				
				
				target.append(find_var(temp_data[4]))
				train_data.append(temp_data[0:4])

	train_data = np.array(train_data)
	target = np.array(target)
	target.shape = (100,1)

	ANN = NeuralNetwork(train_data, target)
	ANN.train()
		
	#confirming with an example
	prediction = ANN.predict(np.array([7.4,2.8,6.1,1.9]))
	
	if (prediction < 0.5):
		print("Iris-setosa")
	else:
		print("Iris-virginica")


if(__name__ == '__main__'):
	main()