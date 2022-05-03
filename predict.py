import os
import sys
import argparse
import pandas as pd
import numpy as  np
from ft_linear_regression import ft_linear_regression


def prompt_user(model):

	while True:
		x = input('''Enter car's mileage:\n''')
		if x.lower() == "end":
			break
		x = int(eval(x))
		if x < 0:
			print("Mileage can't be negative!")
		else:
			print(f"Predicted car value based on mileage: {model.predict(x=x)}")

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
	parser.add_argument("-v", "--verbose", help="Show verbose")
	args = parser.parse_args()
	
	# Parse args here
	verbose = True if args.verbose else False
	model = ft_linear_regression(model_path=args.weights)
	prompt_user(model)

if __name__ == '__main__':
	main()
	