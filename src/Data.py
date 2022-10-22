import numpy as np
import pandas as pd


class Dataset():

    def __init__(self, path):
        try:
            df = pd.read_csv(path)
        except Exception:
            print("Error reading data, stopping program.")
            exit()
		
        self.Y = df['price'].values
        self.X = df.loc[:, df.columns != 'price'].to_numpy()
        self.input_size = self.X.shape[1]