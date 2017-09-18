

import numpy as np 

class loss_layer(object):
	def __init__(self, name ):
		self.name = name 

		self.grad = None 

	def forward(self, X, y): 
		y = y.reshape(X.shape) 

		loss = np.sum( (X - y)*(X - y)/2.0  ) 

		# self.grad = (X - y) 
		self.grad = (X - y) * (X - y) / 2.0 

		return X, loss  

	def predict(self, X):
		return X 


	def backward(self, grad=None,
				 momentum = 0.9, learning_rate = 0.01, reg_lamdb = 1e-4 ):
		return self.grad 

