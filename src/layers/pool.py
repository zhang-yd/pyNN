
import numpy as np 

class pool(object):
	def __init__(self, name, kernel_w, kernel_h, 
					pad = 0, stride=1 ):
		self.name = name 

		self.kernel_h = kernel_h 
		self.kernel_w = kernel_w 

		self.pad = pad 
		assert ( stride != 0 ) 
		self.stride = stride 

		self.X = None 



	def forward(self, X):
		N, C, H, W = X.shape 

		self.X = X 

		out_h = (H - self.kernel_h) / self.stride + 1 
		out_w = (W - self.kernel_w) / self.stride + 1 

		y = np.zeros( [N, C, out_h, out_w] ) 

		for i in range(N):
			for j in range(C):
				for k in range(out_h):
					for p in range(out_w):
						y[i, j, k, p] = np.max( X[i, j, k*self.stride:(k*self.stride + out_h), \
								p*self.stride:(p*self.stride + out_w)] ) 

		return y 


	def backward(self, in_grad, 
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ):
		assert ( self.X.shape[0] == in_grad.shape[0] ) 
		assert ( self.X.shape[1] == in_grad.shape[1] ) 

		N, C, H, W = self.X.shape 

		_, _, out_h, out_w = in_grad.shape 

		dx = np.zeros_like( self.X ) 

		for i in range(N):
			for j in range(C):
				for k in range(out_h):
					for p in range(out_w):
						tmp = self.X[i, j, k*self.stride:(k*self.stride + out_h), p*self.stride:(p*self.stride + out_w) ] 
						max_num = np.max( tmp ) 
						dx[i, j, k*self.stride:(k*self.stride+out_h), p*self.stride:(p*self.stride + out_w) ][tmp == max_num ] = \
								in_grad[i, j, k, p] 

		return dx  

