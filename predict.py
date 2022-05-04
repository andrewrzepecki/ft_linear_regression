import os
import sys
import argparse
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
			print(f"Predicted car value based on mileage: {model(x)}")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-w", "--weights", help="Show verbose")
	args = parser.parse_args()
	
	# Parse args here
	model = ft_linear_regression(model_path=args.weights)
	prompt_user(model)

if __name__ == '__main__':
	main()
	