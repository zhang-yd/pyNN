# pyNN


Neural Network implemented with python following the Structure of Caffe 


## pyNN Source Code  <br>

-src <br>
---layers    ( This floder contains all the functional layers <br>
---------convolution.py <br>
---------fc_layer.py  <br>
---------pool.py  <br>
---------flatten.py <br>
---------... <br>
---net.py    ( The main function is to connect each layer in a series, and establish the network. <br>
---utils.py  ( The utils functions toolbox, including Read_text_data, Read_Mnist, Load_Model, Save_Model, Compute_loss. <br>



## Characteristic <br>
      * 1， only support stacked and straightforward network  <br>
      * 2,  only support sgd  <br>
      * 3,  the implementation of layers is too simple and inefficient  <br>

