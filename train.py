import os
import sys
import argparse
from src.Model import Model
from src.Data import Dataset
from src.Optimize import Optimizer
from src.Visualize import show_hypothesis, show_data

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", type=str, default='data/data.csv', help="Path to data points for training")
	parser.add_argument("-c", "--config", type=str, default='config/linear_regression_std.cfg', help="Path to model configuration")
	parser.add_argument("--optimizer", help="Use basic learning rate optimizer and early_stopping", default=False, action='store_true')
	parser.add_argument("-lr",  "--learning_rate", type=float, default=0.025, help="Learning rate for training")
	parser.add_argument("-e", "--epochs", type=int, default=2000, help="Number of Epochs for training")
	parser.add_argument("--correction", help="Friendly interface for Defense", default=False, action='store_true')
	args = parser.parse_args()

	return args

def train_model(config : str = "", data : str = "", 
				epochs : int = 2000, lr : float = 0.025, 
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
		x = input('Enter Learning Rate (0.0 - 10.0): ')
		try:
			if float(x) > 0.0 and float(x) < 10.0:
				return float(x)
		except Exception:
			print("Invalid input, please enter a valid float between 0 and 1")

def change_epochs():
	
	while True:
		os.system('cls' if os.name == 'nt' else 'clear')
		x = input('Enter Number of Epochs: ')
		try:
			if int(x) > 0 and int(x) < 50000:
				return int(x)
		except Exception:
			print("Invalid input, please enter a valid int under 50000 and over 0")

def user_prompt():
	epochs = 1000
	lr = 0.025
	optimizer = False
	model = None
	data = Dataset("data/data.csv")
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
		if u.lower() == 'exit':
			break
		if u in ['1', '2', '3', '4']:
			data = Dataset("data/data.csv")
			if u == '1':
				config = "config/linear_regression.cfg"
			elif u == '2':
				config = "config/linear_regression_norm.cfg"
			elif u == '3':
				config = "config/linear_regression_std.cfg"
			else:
				config = "config/linear_regression_multi_std.cfg"
				data = Dataset("data/data2.csv")
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
	
	if args.correction:
		user_prompt()
	else:
		data = Dataset(args.data)
		model = train_model(config=args.config, epochs=args.epochs, lr=args.learning_rate, data=data, optimizer=args.optimizer)
		model.save()


if __name__ == '__main__':
	main()
	