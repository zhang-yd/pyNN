
import numpy as np 

class fc_layer(object):
	def __init__(self, name, input_num,  output_num, 
				 initial_mode='guass', initial_std=1e-4):
		self.name = name 

		self.y = None 
		self.x = None 

		if initial_mode is 'guass':
			self.W = initial_std * np.random.randn( input_num, output_num ) 
			self.B = np.zeros( output_num ) 
		else:
			self.W = np.zeros( ( input_num, output_num ) ) 
			self.B = np.zeros( output_num )  

		self.pre_dw = np.zeros( (input_num, output_num) ) 
		self.pre_db = np.zeros( output_num ) 

	def forward(self, x): 
		#### 
		#  x(N, C),  W(C, T),  B(1, T),   Y(N, T) 
		#### 
		assert ( x.shape[1] == self.W.shape[0] ) 

		self.y = x.dot( self.W )  +  self.B 

		self.x = x 

		return self.y 

	def backward(self, in_grad, 
				 momentum = 0.9, learning_rate = 0.01, reg_lamdb = 1e-4 ):
		#### 
		# in_grad (N, T) 
		#### 

		dw = np.dot( self.x.T , in_grad )  ## (C, N)*(N, T)=(C, T) 

		db = np.sum( in_grad, axis = 0, keepdims = True )  ## (N, T)->(1, T)  

		
		dw += reg_lamdb * self.W 

		db += reg_lamdb * self.B 

		self.pre_dw = momentum * self.pre_dw - learning_rate * dw 
		self.W += self.pre_dw 

		self.pre_db = momentum * self.pre_db - learning_rate * db  
		self.pre_db = self.pre_db.reshape( self.B.shape ) 
		self.B += self.pre_db 


		out_grad = np.dot( in_grad, self.W.T )  ### (N, T)*(T, C)=(N, C) 

		return out_grad 


