import os
import sys
import argparse
import pandas as pd
import numpy as  np
from ft_linear_regression import ft_linear_regression


def read_weights(path=""):
	if type(path) == str and path != "":
		try:
			return np.genfromtxt(path, delimiter=',')	
		except:
			pass
	print("Invalid path to csv file containing weights")
	return [0,0]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-tr", "--trainset", type=str, default=False, help="Path to data points for training")
	parser.add_argument("-lr",  "--learning_rate", type=float, default=0.05, help="Learning rate for training")
	parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of Epochs for training")
	parser.add_argument("-v", "--verbose", help="Show verbose")
	args = parser.parse_args()
	
	# Parse args here
	verbose = True if args.verbose else False
	model = ft_linear_regression()
	if args.trainset:
		model.train(train_set=args.trainset, epochs=args.epochs, lr=args.learning_rate)
	else:
		print("Error, specify path to dataset for gradient descent.")


if __name__ == '__main__':
	main()
	