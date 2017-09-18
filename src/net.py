
import numpy as np 
import sys 

from layers.fc_layer import fc_layer 
from layers.softmax import softmax  

from layers.relu import relu 
from layers.sigmoid import sigmoid  
from layers.tanh import tanh 

from layers.convolution import convolution 
from layers.pool import pool 
from layers.loss_layer import loss_layer 
from layers.flatten import flatten 


class net(object):
	def __init__(self, name, structure):
		self.name = name 

		self.layers = [] 

		for elem in structure: 
			self.layers.append( self._construct( elem ) ) 

		self.layers_num = len(structure) 

	def _construct(self, structure):
		if structure[1] == 'fc_layer':
			assert ( len(structure) >= 6 )
			return fc_layer(structure[0], structure[2], structure[3], structure[4], structure[5] )
		elif structure[1] == 'softmax':
			assert ( len(structure) >= 2 ) 
			return softmax(structure[0] ) 
		elif structure[1] == 'convolution':
			assert ( len(structure) >= 8 ) 
			return convolution(structure[0], structure[2], structure[3], structure[4], \
								structure[5], structure[6], structure[7] ) 
		elif structure[1] == 'pool':
			assert ( len(structure) >= 6 ) 
			return pool(structure[0], structure[2], structure[3], structure[4], \
						 structure[5] )
		elif structure[1] == 'sigmoid':  
			return sigmoid(structure[0]) 
		elif structure[1] == 'relu':
			return relu(structure[0]) 
		elif structure[1] == 'tanh':
			return tanh(structure[0]) 
		elif structure[1] == 'loss_layer':
			return loss_layer(structure[0]) 
		elif structure[1] == 'flatten':
			return flatten(structure[0]) 
		else:
			print ('cannot find {} layer'.format( structure[1] )) 
			sys.exit(0) 


	def forward(self, X, y):
		###
		tmp = X 
		for i in range( self.layers_num - 1 ):
			tmp = self.layers[i].forward( tmp )

		tmp, loss = self.layers[-1].forward( tmp, y ) 

		return loss 

	def backward(self, momentum = 0.9, learning_rate=0.01, reg_lamdb=1e-4 ):
		tmp = None  

		for i in range(self.layers_num-1, -1, -1):
			tmp = self.layers[i].backward( tmp, momentum = momentum, \
				learning_rate=learning_rate, reg_lamdb=reg_lamdb )   



	def fit(self, X, y, max_epochs, 
			 batch_size = 0, momentum = 0.9, learning_rate =0.01,
			 reg_lamdb = 1e-4, eps=1e-6, display_epoch = 10): 

		num_train = X.shape[0] 

		cur = 0 
		perm = np.random.permutation( np.arange(num_train) )

		def Get_Batch( X, y, cur ): 
			if batch_size == 0:
				return X, y 
			else:
				batch_x, batch_y = [], [] 
				for i in perm[cur: cur+batch_size]:
					batch_x.append( X[i] ) 
					batch_y.append( y[i] ) 
				cur = ( cur + batch_size ) % num_train 
				return np.array(batch_x), np.array(batch_y), cur 


		learning_rate = self._Get_Learning_rate() 

		epoch = 0  
		while epoch < max_epochs or loss < eps:
			epoch = epoch + 1 
			batch_x, batch_y, cur = Get_Batch( X, y, cur ) 

			loss = self.forward( batch_x, batch_y ) 

			self.backward( momentum, learning_rate, reg_lamdb ) 

			if epoch % display_epoch == 0:
				print('{} th training loss is {}'.format( epoch, loss ) ) 

		print ('trained!')  

	def _Get_Learning_rate(self, lr_policy='default'): 
		return 0.01  

	def predict(self, X):

		tmp = X 
		for i in range( self.layers_num - 1 ):
			tmp = self.layers[i].forward( tmp )

		tmp  = self.layers[-1].predict( tmp ) 

		return tmp 


