
import numpy as np 

class convolution(object):
	def __init__(self, name, kernel_w, kernel_h, kernel_c, kernel_num, 
				 pad = 0, stride = 0):
		self.name = name 

		self.kernel_w = kernel_w  
		self.kernel_h = kernel_h 
		self.kernel_c = kernel_c 
		self.kernel_num = kernel_num 

		self.pad = pad 

		self.stride = stride 

		self.kernel = np.random.randn( kernel_num, kernel_c, kernel_h, kernel_w ) 

		self.B = np.zeros( kernel_num ) 

		self.X = None  
		self.y = None 

		self.pre_grad = None 
		self.pre_grad_b = None 


	def forward(self, X):
		#### 
		assert len(X.shape) == 4 

		N, C, H, W = X.shape 

		self.X = X 


		out_h = 1 + ( H + 2 * self.pad - self.kernel_h ) / self.stride 

		out_w = 1 + ( W + 2 * self.pad - self.kernel_w ) / self.stride 

		pad_x = np.pad( X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant' ) 


		y = np.zeros( [ N, self.kernel_num, out_h, out_w ] ) 

		for i in range(N):
			for j in range(self.kernel_num):
				for k in range(out_h):
					for p in range(out_w):
						y[i, j, k, p] = np.sum( pad_x[i, :, \
						 k*self.stride:(k * self.stride + self.kernel_h), \
						 p*self.stride:(p * self.stride + self.kernel_w) ] \
						 * self.kernel[j, :, :, :] ) + self.B[j]  

		self.y = y 
		return self.y 


	def backward(self, in_grad, 
				 momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ):
	
		assert ( in_grad.shape == self.y.shape )

		N, C, H, W = self.X.shape 

		dx = np.zeros_like( self.X ) 
		dw = np.zeros_like( self.kernel ) 
		db = np.zeros_like( self.B ) 

		out_h = 1 + ( H + 2 * self.pad - self.kernel_h ) / self.stride 
		out_w = 1 + ( W + 2 * self.pad - self.kernel_w ) / self.stride 

		pad_dx = np.pad( dx, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), 'constant') 
		pad_x = np.pad( self.X, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), 'constant') 

		for i in range(N):
			for j in range(self.kernel_num):
				for k in range(out_h):
					for p in range(out_w):
						pad_dx[i, :, k*self.stride:(k*self.stride + self.kernel_h), \
							p*self.stride:(p*self.stride + self.kernel_w)] += in_grad[i, j, k, p] \
							* self.kernel[j, :, :, :] 
						dw[j, :, :, :] += in_grad[i,j,k,p] * pad_x[i, :, k*self.stride:(k*self.stride + self.kernel_h), \
							p*self.stride:(p*self.stride + self.kernel_w)] 
						db[j] += in_grad[i, j, k, p] 
		dx = pad_dx[:, :, self.pad: (self.pad + H ), \
					self.pad:(self.pad + W )]  

		## 
		if self.pre_grad is not None:
			self.pre_grad = momentum * self.pre_grad - learning_rate * dw 
		else:
			self.pre_grad = -learning_rate * dw 

		self.kernel += self.pre_grad 


		if self.pre_grad_b is not None:
			self.pre_grad_b = momentum * self.pre_grad_b - learning_rate * db 
		else:
			self.pre_grad_b = -learning_rate * db 

		self.B += self.pre_grad_b 

		return dx 

