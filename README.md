# pyNN


Neural Network implemented with python following the Structure of Caffe 


pyNN Source Code 

--src
  --layers    # This floder contains all the functional layers 
          --convolution.py 
          --fc_layer.py 
          --pool.py
          --flatten.py 
          --...
  --net.py    # The main function is to connect each layer in a series, and establish the network.  
  --utils.py  # The utils functions toolbox, including Read_text_data, Read_Mnist, Load_Model, Save_Model, Compute_loss. 
  



Characteristic
      1， only support stacked and straightforward network 
      2,  only support sgd 
      3,  the implementation of layers is too simple and inefficient 

