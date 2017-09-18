

import numpy as np 


class tanh(object):
	def __init__(self, name):
		self.name = name 

		self.X = None 

		self.y = None 

	def forward(self, X):
		self.X = X 

		self.y = ( np.exp(X) - np.exp(-X) )/( np.exp(X) + np.exp(-X) ) 

		return self.y 

	def backward(self, in_grad, 
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ):
		tanh_x = self.y 

		out_grad = in_grad * ( 1. - tanh_x * tanh_x )  

		return out_grad 

