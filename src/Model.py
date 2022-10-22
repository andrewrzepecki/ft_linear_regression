

#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ft_linear_regression.py
@Time    :   2022/10/22 13:30:50
@Author  :   Andrew Rzepecki 
@Version :   1.0
@Contact :   rzepecki.andrew@gmail.com
@License :   (C)Copyright 2022-2023, Andrew Rzepecki
@Desc    :   None
'''

import os
import json
import pickle
import numpy as np
try:
	from Modules import Linear, Standardize, MODULE_MAP
except Exception:
	from src.Modules import Linear, Standardize, MODULE_MAP

LINEAR_REGRESSION_CONFIG = "linear_regression.cfg"

class Model():

	
	def __init__(self, model_path : str = None, verbose : bool = False, config : str = LINEAR_REGRESSION_CONFIG):
		self._modules = []
		if model_path:
			self.load_from_checkpoint(model_path)
		if self._modules != []:
			print(f'Model loaded from {model_path}')
		else:
			self.load_from_config(config)
			print("Loading Untrained Model\n - Train model with model.train([labelled_data])\n - Or specify a model checkpoint")
			print("Initialed weights to zero ...")

	
	def forward(self, x):
		"""
			Forward call to Model modules
		"""
		for module in self._modules:
			x = module(x)
		return x

	
	def __call__(self, x):
		''' 
			Call function to predict. Defaults x to 0 if no arguments given for value of Discremenant.
		'''
		return self.forward(x)

	
	def load_from_config(self, path : str):
		if os.path.exists(path):
			basename = os.path.basename(path)
			self._name_, _ = os.path.splitext(basename)
			with open(path, 'rb') as fd:
				modules = json.load(fd)
				fd.close()
			for module in modules:
				self._modules.append(MODULE_MAP[module['key']]())
		else:
			print("Invalid filepath to model configuration.")

	
	def load_from_checkpoint(self, path : str):
		if os.path.exists(path):
			try:
				with open(path, 'rb') as fd:
					data = pickle.load(fd)
					self._modules = data._modules
					self._name_ = data._name_
					fd.close()
			except Exception as e:
				print("Invalid file format.")
		else:
			print("Invalid filepath to model checkpoint.")

	
	def save(self, name : str = None):
		with open(f"{self._name_ if name is None else name}.ml", 'wb') as fd:
			pickle.dump(self, fd)
			fd.close()
	
	
	def fit(self, x, y, epochs=1000):
		# Fit Data Preprocessing
		self._modules[0].fit(x.T)
		x = np.array([self._modules[0](X) for X in x])
		print(self._modules[0].__dict__)

		# Train trainable layers
		for e in range(epochs):
			for module in self._modules[1:]:
				module.fit(x, y)
				
	'''
	def get_loss(self, x, y, t0, t1):
		total_loss = 0
		for i in range(len(x)):
			total_loss += (y[i] - (t0 + x[i] * t1)) ** 2
		return total_loss / len(x)
	'''