
import numpy as np 

class softmax(object):
	def __init__(self, name ):
		self.name = name 

		self.y = None 

		self.grad = None 

	def forward(self, X, Target ):
		####  X (N, C) 
		# x_max = np.reshape(np.max( X, axis=1 ), (self.dim, 1)) ## (N, 1) 

		assert ( X.shape[1] >= np.max(Target) ) 

		num_train = X.shape[0]   

		x_max = np.max( X, axis=1 ).reshape(num_train, 1)  

		prob = np.exp( X - x_max) / np.sum( np.exp( X - x_max ), axis=1, keepdims=True )  ## (N, C) 


		self.y = np.argmax( prob, axis = 1) 



		loss = -np.sum( np.log(prob[ range(num_train), Target ]) ) 

		prob[range(num_train), Target] -= 1 

		self.grad = prob / num_train 


		return self.y, loss 



	def predict(self, X):

		num_train = X.shape[0] 

		x_max = np.max( X, axis=1 ).reshape(num_train, 1) 

		prob = np.exp( X - x_max ) / np.sum( np.exp(X - x_max), axis=1, keepdims=True ) 

		return np.argmax( prob, axis=1 ) 



	def backward(self, in_grad = None,
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ):
		### 
		return self.grad 


