import os
import sys
import argparse
from src.Model import Model
from src.Data import Dataset
from src.Optimize import Optimizer
from src.Visualize import show_hypothesis, show_data, show_derivative


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", type=str, default=False, help="Path to data points for training")
	parser.add_argument("-lr",  "--learning_rate", type=float, default=0.05, help="Learning rate for training")
	parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of Epochs for training")
	parser.add_argument("-v", "--verbose", help="Show verbose")
	args = parser.parse_args()
	
	# Parse args here
	verbose = True if args.verbose else False
	model = Model(config='linear_regression_std.cfg')
	optimizer = Optimizer(model)
	data = Dataset(args.data)

	model.fit(data.X, data.Y, optimizer=optimizer)
	print(model.lr)
	print(model([50000]))


if __name__ == '__main__':
	main()
	