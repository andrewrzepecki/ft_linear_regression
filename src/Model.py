import os
import json
import pickle
import numpy as np
try:
	from Modules import Module, Linear, Standardize, MODULE_MAP
except Exception:
	from src.Modules import Module, Linear, Standardize, MODULE_MAP
try:
	from Maths import mse
except Exception:
	from src.Maths import mse


LINEAR_REGRESSION_CONFIG = "linear_regression_std.cfg"

class Model(Module):
	
	def __init__(self, model_path : str = None, 
					config : str = LINEAR_REGRESSION_CONFIG):
		
		super().__init__()
		self._modules = []
		if model_path:
			self.load_from_checkpoint(model_path)
		if self._modules != []:
			print(f'Model loaded from {model_path}')
		elif self.load_from_config(config):
			print("Loading Untrained Model\n - Train model with model.train([labelled_data])\n - Or specify a model checkpoint")
			print("Initialed weights to zero ...")
		else:
			return None

	
	def forward(self, x):
		"""
			Forward call to Model modules
		"""
		for module in self._modules:
			x = module(x)
		return x

	
	def load_from_config(self, path : str, lr : float = 0.025):
		if os.path.exists(path):
			basename = os.path.basename(path)
			self._name_, _ = os.path.splitext(basename)
			try:
				with open(path, 'rb') as fd:
					modules = json.load(fd)
					fd.close()
				for module in modules:
					self._modules.append(MODULE_MAP[module['key']](dim=module['dim'], trainable=module['trainable']))
				self.input_size = self._modules[0].dim
				return True
			except Exception:
				print("Invalid configuration format.")
				return False
		else:
			print("Invalid filepath to model configuration.")
			return False

	
	def load_from_checkpoint(self, path : str):
		if os.path.exists(path):
			try:
				with open(path, 'rb') as fd:
					data = pickle.load(fd)
					self._modules = data._modules
					self._name_ = data._name_
					self.input_size = data.input_size
					fd.close()
			except Exception as e:
				print("Invalid file format.")
		else:
			print("Invalid filepath to model checkpoint.")

	
	def save(self, name : str = None):
		with open(f"{self._name_ if name is None else name}.ml", 'wb') as fd:
			pickle.dump(self, fd)
			fd.close()
	
	
	def fit(self, x, y, epochs : int = 10000, lr : int = 0.025, optimizer = None):
		
		# Fit Data Preprocessing
		raw = np.array([X for X in x])
		self.lr = lr
		for module in self._modules:
			if not module.trainable:
				module.fit(x.T)
				x = np.array([module(X) for X in x])
		# Train trainable layers
		for e in range(epochs):
			for module in self._modules:
				if module.trainable:
					module.fit(x, y, self.lr)
			
			y_pred = [self.forward(X) for X in raw]
			loss = mse(y, y_pred)
			print(f"Loss at epoch {e}: {loss}")
			if optimizer:
				if optimizer(loss):
					print(f"Early Stopping applied, model has converged at epoch {e}")
					break