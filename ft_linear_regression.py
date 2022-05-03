import sys
import os
import numpy as np
import json
import math
import matplotlib.pyplot as plt

class	ft_linear_regression():

	def __init__(self, model_path=None, verbose=False):
		if model_path is not None:
			try:
				with open(model_path, 'r') as f:
					checkpoint = json.load(f)
					f.close()
				self.load_model(model=checkpoint)
				self.trained = True
				print(f'Model loaded from {model_path}')
			except:
				print("Warning: Loading Untrained Model\n - Train model with model.train([labelled_data])\n - Or specify custom weights")
				print("Initializing weights to zero ...")
				self.weights = [0, 0]
				self.trained = False

	def visualize_trainset(self, set):
		plt.scatter(set.T[0], set.T[1])
		plt.title('Price VS KMs')
		plt.show()
		input()
		return 0

	def __call__(self, x=0):
		'''
			Call function to predict. Defaults x to 0 if no arguments given for value of Discremenant.
		'''
		return self.predict(x)

	def load_model(self, model):
		try:
			self.weights = model['weights']
			self.minimum = model['x_min']
			self.maximum = model['x_max']
		except:
			print("Error in model format, Initializing weights to zero ...")
			self.weights = [0,0]	

	def save_model(self, x_min, x_max):
		model = {
			'weights': [self.weights[0], self.weights[1]],
			'x_min' : x_min,
			'x_max' : x_max
		}
		with open("model.json", 'w') as f:
			json.dump(model, f)
			f.close()
		print("Saved model as \"model.json\".")

	def predict(self, x):
		"""
			linear regression (equation of a flat line) = Theta0 + km * Theta1
		"""
		return self.weights[0] + self.normalize_x(x) * self.weights[1]

	def _predict(self, x):
		return self.weights[0] + self.weights[1] * x

	def train(self, train_set=None, epochs=10000, lr=0.05):
		self.weights = [0,0]
		self.alpha=lr
		dataset = np.genfromtxt(train_set, delimiter=",", skip_header=1)
		transforms = self.visualize_trainset(dataset)
		x = self.scale(dataset.T[0])
		y = dataset.T[1]

		## Start train loop through iterating epochs.
		print(f"Training Model on 2X{len(x)} dataset.")
		print(f"Alpha: {self.alpha} | Theta0: {self.weights[0]} | Theta1: {self.weights[1]} | Epochs: {epochs}")
		self.train_loss = []
		for epoch in range(epochs):
			t0, t1 = self.gradient_descent(x=x, y=y, m=len(dataset))
			print(t0, t1)
			#self.train_loss.append(self.get_loss(self.x, self.y, len(dataset)))
			if self.get_loss(x=x, y=y, t0=t0, t1=t1, m=len(x)) > self.get_loss(x=x, y=y,  t0=self.weights[0], t1=self.weights[1], m=len(x)):
				print(f"Stopping training at epoch {epoch} due to avg loss increasing.")
				break
			self.weights = [t0, t1]
			if epoch % int(epochs / 10) == 0:
				print(f" --  Loss at Epoch {epoch}: {self.get_loss(x=x, y=y, t0=self.weights[0], t1=self.weights[1], m=len(x))}")
		self.save_model(dataset.T[0].min(), dataset.T[0].max())
		self.trained = True
		self.plot_line(x=dataset.T[0], y=dataset.T[1])

	
	def scale(self, x):

		self.minimum = x.min()
		self.maximum = x.max()
		return [(p - self.minimum) / (self.maximum - self.minimum) for p in x]
		
	
	def normalize_x(self, x):
		if self.trained:
			return (x - self.minimum) / (self.maximum - self.minimum)
		return x

	#def normalize_y(self, y):
	#	if self.trained:
	#		return (y * (self.max_y - self.min_y)) + self.min_y
	#	return y
	
	def get_loss(self, x, y, t0, t1, m):
		total_loss = 0
		for km, price in zip(x, y):
			total_loss += math.sqrt((price - (t0 + t1 * km)) ** 2)
		return total_loss / m
	
	def plot_line(self, x, y):
		plt.plot([self.predict(km) for km in x])
		plt.plot(y)
		plt.title('Estimation of a car\'s value based on mileage.')
		plt.show()
	
	def gradient_descent(self, x, y, m):
		mse, intercept, gradient = 0, 0, 0
		for km, price in zip(x, y):
			error = math.sqrt((price - self._predict(km)) ** 2)
			intercept += error
			gradient += error * km
		gradient = gradient / m
		intercept = intercept / m
		return self.weights[0] - intercept * self.alpha, self.weights[1] - gradient * self.alpha