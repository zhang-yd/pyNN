
import sys, os 
import numpy as np 
from src.net import net 
from src.utils import * 


def fun(train_file, test_file):
	train_x, train_y = Read_Data(train_file)  

	### Define the network structrue 
	input_dim = train_x.shape[1]  
	
	out_dim = 1 

	struct = [
		['fc1', 'fc_layer', input_dim, out_dim, 'guass', 1e-4 ], 
		['sigmoid', 'sigmoid'] , 
		['loss', 'loss_layer'] 
	]


	### Set the hyper parameters 
	max_epochs = 80000   		### training max epochs 
	momentum = 0.9 				### momentum of gradient descent 
	batch_size = 200 			### the mini batch size 
	reg_lamdb = 1e-4 			### lamdb, the Coefficient of regular item 
	learning_rate = 0.1 		### learning rate 
	display_epoch = 500  		### display the loss every x epochs while training 
	loss_stop_eps = 1e-6 		### eps  

	model_path = './linear_regression.pkl' 			### model path 


	### 
	nerual_network = net('net', struct) 

	### training 
	nerual_network.fit( train_x, train_y,max_epochs,
						 batch_size=batch_size, momentum = momentum, 
						 learning_rate = learning_rate , reg_lamdb = reg_lamdb, 
						 eps = loss_stop_eps, display_epoch = display_epoch ) 

	#### Read the test data 
	test_x, test_y = Read_Data( test_file )  

	#### testing 
	pred = nerual_network.predict( test_x ) 

	### Compute the L2 loss 
	l2_loss = Compute_L2_loss( pred, test_y ) 

	### Save the model 
	Save_Model( nerual_network,  model_path )  

	print( 'testing l2_loss: {}'.format( l2_loss ) )  

	print( 'trained model saved in {}'.format( model_path ) )


if __name__ == '__main__':
	fun('dt2_train.txt', 'dt2_test.txt')  
