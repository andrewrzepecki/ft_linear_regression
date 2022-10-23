import os
from pickletools import optimize
import sys
import argparse
from src.Model import Model
from src.Data import Dataset
from src.Optimize import Optimizer
from src.Visualize import show_hypothesis, show_data

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", type=str, default=False, help="Path to data points for training")
	parser.add_argument("--optimizer", help="Use basic learning rate optimizer and early_stopping", default=argparse.SUPPRESS)
	parser.add_argument("-lr",  "--learning_rate", type=float, default=0.05, help="Learning rate for training")
	parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of Epochs for training")
	parser.add_argument("--verbose", help="Show verbose", default=argparse.SUPPRESS)
	parser.add_argument("--correction", help="Friendly interface for Defense", default=argparse.SUPPRESS)
	args = parser.parse_args()

	return args

def train_model(config : str = "", data : str = "", 
				epochs : int = 10000, lr : float = 0.025, 
				optimizer : bool = False):

	model = Model(config=config)
	if optimizer:
		optimizer = Optimizer(model)
	else:
		optimizer = None
	model.fit(data.X, data.Y, optimizer=optimizer, lr=lr, epochs=epochs)
	
	return model

def change_lr():
	
	while True:
		os.system('cls' if os.name == 'nt' else 'clear')
		x = input('Enter Learning Rate (0.0 - 1.0): ')
		try:
			if float(x) > 0.0 and float(x) < 1.0:
				return float(x)
		except Exception:
			print("Invalid input, please enter a valid float between 0 and 1")
			input()

def change_epochs():
	
	while True:
		os.system('cls' if os.name == 'nt' else 'clear')
		x = input('Enter Number of Epochs: ')
		try:
			if int(x) > 0 and int(x) < 100000000:
				return int(x)
		except Exception:
			print("Invalid input, please enter a valid int")

def user_prompt():
	epochs = 1000
	lr = 0.025
	optimizer = False
	model = None
	data = Dataset("data.csv")
	end = '\033[0m'
	on = '\033[92mOn' + end
	off = '\033[91mOff' + end
	bold = '\033[1m'
	while True:
		print(f"{bold}Hyperparameters:{end} Epochs {bold}{epochs}{end} | Learning Rate {bold}{lr}{end} | Optimizer {on if optimizer else off}")
		print("1. Simple Linear Regression")
		print("2. Simple Linear Regression with Normalization")
		print("3. Simple Linear Regression with Standardization")
		print("4. Multi Variable Linear Regression with Standardization")
		print("5. Change Number of Maximum Epochs")
		print("6. Change Learning Rate (alpha)")
		print("7. Activate/Deactivate Optimizer (Early Stopping & Dynamic Learning Rate)")
		print("8. Visualize Data")
		print("9. Visualize Hypothesis")
		print("10. Exit & Save last model Trained")
		u = input('Select Choice (1-10): ')
		if u in ['1', '2', '3', '4']:
			data = Dataset("data.csv")
			if u == '1':
				config = "linear_regression.cfg"
			elif u == '2':
				config = "linear_regression_norm.cfg"
			elif u == '3':
				config = "linear_regression_std.cfg"
			else:
				config = "linear_regression_multi_std.cfg"
				data = Dataset("data2.csv")
			model = train_model(config=config, epochs=epochs, lr=lr, data=data, optimizer=optimizer) 
			input()
		if u == '5':
			epochs = change_epochs()
		if u == '6':
			lr = change_lr()
		if u == '7':
			optimizer = False if optimizer else True
		if u == '8':
			show_data(model, data)
		if u == '9':
			show_hypothesis(model, data)
		if u == '10':
			if model:
				print(f"Saving model at {model._name_}.ml")
				model.save()
			break
		os.system('cls' if os.name == 'nt' else 'clear')



def main():
	args = parse_args()	
	
	if 'correction' in args:
		user_prompt()
	else:
		train_model(args)


if __name__ == '__main__':
	main()
	