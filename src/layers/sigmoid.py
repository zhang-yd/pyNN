

import numpy as np 

class sigmoid(object):
	def __init__(self, name):
		self.name = name 

		self.X = None 
		
		self.y = None 

	def forward(self, X):
		self.X = X 

		self.y = 1.0 / (1.0 + np.exp(-X)) 

		return self.y 

	def backward(self, in_grad, 
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ): 
		sigmoid_x = self.y 

		out_grad = in_grad * sigmoid_x * ( 1. - sigmoid_x ) 

		return out_grad 
