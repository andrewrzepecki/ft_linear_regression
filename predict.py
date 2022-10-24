import os
import sys
import argparse
from src.Model import Model


def prompt_user(model):

	while True:
		x = input('''Enter car's mileage:\n''')
		if x.lower() == "end":
			break
		elif model.input_size == 1:
			x = int(eval(x))
			if x < 0:
				print("A car can't have negative mileage!")
			print(f"Predicted car value based on mileage: {model([x])}")
		elif ' ' in x:
			x = x.split(' ')
			if len(x) == model.input_size:
				x = [int(eval(X)) for X in x]
				print(f"Predicted car value based on mileage: {model(x)}")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", help="Show verbose", default=None)
	args = parser.parse_args()
	
	# Parse args here
	model = Model(model_path=args.model)
	prompt_user(model)

if __name__ == '__main__':
	main()
	