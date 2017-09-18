
import numpy as np 


class flatten(object):
	def __init__(self, name):
		self.name = name 

		self.dim = None 

	def forward(self, X):

		self.dim = X.shape 

		num_train = X.shape[0] 

		dim = 1 
		for i in range(1, len(X.shape)):
			dim = dim * X.shape[i] 

		y = X.reshape( num_train, dim )  

		return y 

	def backward(self, in_grad, 
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ):

		out_grad = in_grad.reshape( self.dim ) 

		return out_grad 



