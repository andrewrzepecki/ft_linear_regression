from re import I
import sys
import os
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import pandas as pd

class	ft_linear_regression():

	def __init__(self, model_path=None, verbose=False):
		if model_path is not None:
			try:
				with open(model_path, 'r') as f:
					checkpoint = json.load(f)
					f.close()
				self.load_model(model=checkpoint)
				print(f'Model loaded from {model_path}')
			except:
				print("Warning: Loading Untrained Model\n - Train model with model.train([labelled_data])\n - Or specify custom weights")
				print("Initializing weights to zero ...")
				self.t0 = 0
				self.t1 = 0
				self.input_scaler = None
		else:
			self.t0 = 0
			self.t1 = 0
			self.input_scaler = None

	def __call__(self, x=0):
		''' 
			Call function to predict. Defaults x to 0 if no arguments given for value of Discremenant.
		'''
		return self.predict(x)

	def hypothesis(self, x):
		return self.t0 + x * self.t1

	def predict(self, x):
		x = self.scale_input(x)
		result = self.hypothesis(x)
		if result < 0:
			result = 0
		return result

	def get_loss(self, x, y, t0, t1):
		total_loss = 0
		for i in range(len(x)):
			total_loss += (y[i] - (t0 + x[i] * t1)) ** 2
		return total_loss / len(x)
	
	def scale_input(self, x):
		if self.input_scaler is None:
			print("Warning! Model not trained, weights set to 0 and no input scaling...")
			return x
		else:
			return (x - self.input_scaler['min']) / (self.input_scaler['max'] - self.input_scaler['min'])

	def fit(self, x, y, lr):
		gradient0 = 0
		gradient1 = 0
		n = len(x)

		for i in range(n):
			gradient0 += self.hypothesis(x[i]) - y[i]
			gradient1 += (self.hypothesis(x[i]) - y[i]) * x[i]
		t0 = self.t0 - gradient0 * (1/n) * lr
		t1 = self.t1 - gradient1 * (1/n) * lr
		return t0, t1

	def read_data(self, path):
		try:
			data = pd.read_csv(path)
			y = data['price'].values
			x = data['km'].values
			return x, y
		except Exception:
			print("Error reading data, stopping program.")
			exit()
	
	def train(self, data, epochs, lr):
		x, y = self.read_data(data)
		self.input_scaler = {
			'min' : x.min(),
			'max' : x.max(),
			'mean' : x.mean()
		}
		x = [self.scale_input(x) for x in x]
		history = []
		for epoch in range(epochs):
			t0, t1 = self.fit(x, y, lr)
			current_loss = self.get_loss(x, y, t0, t1)
			print(f"Loss : {current_loss}")
			history.append({
				't0' : t0,
				't1' : t1,
				'loss' : current_loss
			})
			self.t0 = t0
			self.t1 = t1
		# Save model weights with lowest cost.
		self.save_model(history)
	
	def load_model(self, model):
		try:
			self.t0 = model['t0']
			self.t1 = model['t1']
			self.input_scaler = {
				'min' : model['input_min'],
				'max' : model['input_max'],
				'mean' : model['input_mean']
			}
		except:
			print("Error in model format, Initializing weights to zero ...")
			self.t0 = 0
			self.t1 = 0
			self.input_scaler = None	

	def save_model(self, history):
		# Get best epoch
		min_loss = history[0]['loss']
		idx = 0
		for i in range(1, len(history)):
			if history[i]['loss'] < min_loss:
				min_loss = history[i]['loss']
				idx = i
		self.t0 = history[idx]['t0']
		self.t1 = history[idx]['t1']
		model = {
			't0': self.t0,
			't1': self.t1,
			'input_min' : int(self.input_scaler['min']),
			'input_max' : int(self.input_scaler['max']),
			'input_mean' : int(self.input_scaler['mean'])
		}
		print(model)
		with open("model.json", 'w') as f:
			json.dump(model, f)
			f.close()
		print("Saved model as \"model.json\".")