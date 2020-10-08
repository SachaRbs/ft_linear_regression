import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def standardize(x):
	return (x - np.mean(x)) / np.std(x)

def destandardize(x, x_ref):
	return x * np.std(x_ref) + np.mean(x_ref)

def plot_data(data, x, y):
	plt.plot(data[:, 0], data[:, 1], 'o')
	plt.plot(x, y)
	plt.xlabel("Km")
	plt.ylabel("Price")
	plt.show()

class LinearRegression():
	def __init__(self, m, alpha, iterations):
		self.alpha = alpha
		self.iterations = iterations
		self.theta = np.zeros((1, 2))
		self.m = m
	
	def fit(self, X, y):
		for i in range(0, self.iterations):
			tmp_theta = np.zeros((1, 2))
			tmp_theta[0, 0] = (self.alpha * np.sum(self.predict(X) - y)) / self.m
			tmp_theta[0, 1] = (self.alpha * np.sum((self.predict(X) - y) * X)) / self.m
			self.theta -= tmp_theta

	def _mse(self, y, y_pred):
		res = (np.sum(y) - np.sum(y_pred))**2 / len(y)
		print("mse : {}".format(res))

	def predict(self, X):
		return self.theta[0, 0] + self.theta[0, 1] * X
	
	def save_theta(self, x, y):
		a = (y[0] - y[1]) / (x[0] - x[1])
		b = a * x[0] * -1 + y[0]
		theta = [[float(b), float(a)]]
		
		with open('../data/theta.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(theta)


def main():
	df = pd.read_csv('../data/data.csv')
	data = np.array(df)
	X = standardize(np.array(df['km']))
	y = standardize(np.array(df['price']))
	m = len(data)
	alpha = 0.3
	iterations = 200
	model = LinearRegression(m, alpha, iterations)
	model.fit(X, y)

	y = model.predict(X)
	y = destandardize(y, data[:, 1])
	X = destandardize(X, data[:, 0])
	model.save_theta(X, y)

	plot_data(data, X, y)

	

if __name__ == "__main__":
	main()
