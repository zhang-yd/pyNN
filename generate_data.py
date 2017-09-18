

import os, sys 
import numpy as np 


def generate_data1(x_d, x_num, class_num , target_file):
	
	X = np.random.randn( x_num,  x_d )   
	y = np.zeros( x_num ) 

	for i in range(class_num-1):
		X_tmp  = np.random.randn( x_num,  x_d ) 
		X_tmp  = X_tmp * ( i + 2 )* ( i + 2 )  
		X = np.concatenate((X, X_tmp), axis=0) 

		y_tmp  = np.ones( x_num ) * ( i + 1 )
		y = np.concatenate( (y, y_tmp), axis=0) 

	X = np.array(X)  
	y = np.array(y) 

	print X.shape  
	print y.shape 

	with open(target_file, 'w') as f:
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				f.write( str(X[i, j]) + ' ' )  
			f.write( str(int(y[i])) )  
			f.write('\n') 
	print('Done.') 


def generate_data2(x_d, x_num, class_num, target_file):
	X = np.random.randn( x_num, x_d ) 
	y = np.random.random( [x_num] ) * pow(10, 0) 

	for i in range(class_num-1):
		X_tmp = np.random.randn( x_num, x_d ) 
		X_tmp = X_tmp * (i + 2) *(i + 2) 
		X = np.concatenate((X, X_tmp), axis=0) 

		y_tmp = np.random.random( [x_num] ) * pow(10,  i + 1 )  
		y = np.concatenate((y, y_tmp), axis=0) 
	X = np.array(X) 
	y = np.array(y) 

	print 'max', np.max(y) 
	print 'min', np.min(y)  

	print y.shape 

	y = ( y - np.min(y) ) / ( np.max(y) - np.min(y) ) 
	
	print 'max', np.max(y) 
	print 'min', np.min(y) 

	print X.shape 
	print y.shape 

	with open(target_file, 'w') as f:
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				f.write( str(X[i, j]) + ' ' ) 
			f.write( str(y[i]) + '\n' ) 
	print('Done')  





if __name__ == '__main__':
	generate_data1(50, 500,  4, 'dt1_train.txt') 
	generate_data1(50, 50, 4, 'dt1_test.txt') 

	generate_data2(50, 500, 5, 'dt2_train.txt') 
	generate_data2(50, 50, 5, 'dt2_test.txt') 

