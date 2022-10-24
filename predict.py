import os
import sys
import argparse
from src.Model import Model


def prompt_user(model : Model):
	
	msg = "Enter car's Mileage:\n" if model.input_size == 1 else "Enter car's Mileage & Year:\n"
	while True:
		x = input(msg)
		if x.lower() == "end" or x.lower() == 'exit':
			break
		elif model.input_size == 1 and x.isnumeric():
			x = int(x)
			if x < 0:
				print("A car can't have negative mileage!")
			else:
				result = model([x])
				print(f"Predicted car value based on mileage: {result}")
				if result < 0.0:
					print("The car is in such bad condition that you will need to pay someone to take it!")
		elif ' ' in x:
			x = x.split(' ')
			if len(x) == model.input_size and x[0].isnumeric() and x[1].isnumeric():
				x1 = int(x[0])
				x2 = int(x[1])
				if x1 < 0:
					print("A car can't have negative mileage!")
				elif x2 < 0 or x2 > 2225:
					print("A car's Year should be between 0 and 2225")
				else:
					result = model([x1, x2])
					print(f"Predicted car value based on Mileage & Year: {result}")
					if result < 0.0:
						print("The car is in such bad condition that you will need to pay someone to take it!")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", help="Path to trained model", default=None)
	args = parser.parse_args()
	
	# Parse args here
	model = Model(model_path=args.model)
	prompt_user(model)

if __name__ == '__main__':
	main()
	