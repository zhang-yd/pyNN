
import os, sys 
import numpy as np 
import cPickle as pickle   

def Read_Data(file_name):
	data = [] 
	with open(file_name, 'r') as f:
		for lines in f.readlines():
			line = lines.strip().split() 
			data.append( [float(x) for x in line] ) 
	data = np.array(data) 

	data_x = data[:, :-1] 
	data_y = data[:, -1]  

	return data_x, data_y.T 


def Compute_L2_loss(y, label):


	label = label.reshape(y.shape) 

	loss = np.sum(( y - label ) * ( y - label) ) / 2.0 

	return loss 



def Compute_L1_loss(y, label):

	label = label.reshape( y.shape ) 

	loss = np.sum( abs( y - label ) ) 

	return loss 


def Compute_Match(y, label):
	label = label.reshape( y.shape ) 

	return np.sum( label == y ) * 1.0 / len(label) 




def Save_Model(model, path, save_mode='default'):
	if save_mode is 'default':
		f1 = file(path, 'wb' ) 
		pickle.dump( model, f1, True ) 
		f1.close() 
	elif save_mode is 'other':
		pass 




def Load_Model(path, save_mode='default'):
	if save_mode is 'default':
		
		f1 = file(path, 'rb' )
		a = pickle.load( f1 )
		f1.close() 

	elif save_mode is 'other':
		a = None 

	return a 



def Read_MNIST( file_name ):
	x_path, y = [], [] 

	with open( file_name, 'r') as f:
		for lines in f.readlines():
			line = lines.strip().split() 
			x_path.append( line[0] ) 
			y.append( int(line[1]) ) 

	x = [] 
	for i in range(len(x_path)):  
		try:
			img = io.imread( x_path[i] )  
			img = transform.rescale(img, 1.0)*255.0 
			img = img - [117.0]  
		except:
			img = np.zeros((1, 28, 28)) 
			print '!!!!!!!'   
		x.append( img.reshape(1, 28, 28) ) 

	return np.array(x), np.array(y) 


