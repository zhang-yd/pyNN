
import numpy as np 
import sys 

class relu(object):
	def __init__(self, name, 
				 relu_mode=None, alpha=0.25 ):
		self.name = name 

		self.relu_mode = relu_mode 

		self.alpha = alpha 

		self.X = None 

	def forward(self, X):
		if self.relu_mode is None:
			X[ X < 0 ] = 0 

		elif self.relu_mode is 'leaky':
			X[ X < 0 ] = self.alpha * X[ X < 0 ] 

		else:
			print( 'relu mode is unknown.' )
			sys.exit(0) 

		self.X = X 
		
		return self.X 

	def backward(self, in_grad,
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4):
		assert (in_grad.shape == self.X.shape) 

		out_grad = in_grad 

		out_grad[ self.X < 0 ] = 0 

		return out_grad 


