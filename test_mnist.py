
import sys, os 
import numpy as np 
from src.net import net 
from src.utils import * 
from skimage import io, transform


def fun(train_file, test_file): 

	# train_file = './mnist_train.txt'  

	train_x, train_y = Read_MNIST(train_file )  

	struct = [ 
		['conv1', 'convolution', 5, 5, 1, 6, 0, 2], 
		['pool1', 'pool', 2, 2, 0, 1], 
		['conv2', 'convolution', 3, 3, 6, 10, 0, 2], 
		['pool2', 'pool', 2, 2, 0, 1], 
		['flatten', 'flatten'], 
		['fc1', 'fc_layer', 160, 80, 'guass', 1e-4], 
		['fc2', 'fc_layer', 80, 10, 'guass', 1e-4], 
		['softmax', 'softmax'] 
	] 

	max_epochs = 1000 
	momentum = 0.9 
	batch_size = 4   
	reg_lamdb = 1e-4 
	learning_rate = 0.01 
	display_epoch = 2   
	loss_stop_eps = 1e-4 
	model_path = './mnist_recog.pkl' 


	a = net('net', struct) 

	a.fit( train_x, train_y, max_epochs, 
			 batch_size=batch_size, momentum = momentum, 
			 learning_rate = learning_rate , reg_lamdb = reg_lamdb, 
			 eps = loss_stop_eps, display_epoch = display_epoch) 


	test_x, test_y = Read_MNIST( test_file )

	pred = a.predict( test_x ) 

	accuracy = Compute_Match(pred, test_y) 

	print('testing accuracy: {}'.format( accuracy ) )  

	Save_Model( a, model_path ) 

	print( 'trained model saved in {}'.format( model_path ) ) 



if __name__ == '__main__':
	train_file = '' 
	test_file = '' 
	fun( train_file, test_file ) 
