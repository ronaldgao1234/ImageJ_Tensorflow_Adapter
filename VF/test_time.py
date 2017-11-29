import tensorflow as tf
from PIL import Image
from time import time
import numpy as np
from ops import *
import parameters
import os

dict = parameters.import_parameters( 'PARAMETERS.txt' )

NUM_BLOCKS = int( dict['num_blocks'] )
ALPHA = int( dict['alpha'] )
FIRST_CHANNELS = int( dict['first_channels'] )
SCALE = int( dict['scale'] )

input = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )
network_output = build_network( input , NUM_BLOCKS , alpha = ALPHA , first_channels = FIRST_CHANNELS )

with tf.Session() as sess :

    saver = tf.train.Saver()
    model_path = 'Models/best_model'
    saver.restore( sess , model_path )

    os.system('rm -r output_images')
    os.system('mkdir output_images')

    img = Image.open('input.tif')
    inputs = np.array( [ np.array(img) ] ).astype(np.float32)
    img.close()

    start_time = time()

    network_images = sess.run( network_output , feed_dict = { input : inputs } )

    network = np.clip( network_images[0] , 0 , 255 ).astype('uint8')

    Image.fromarray( network ).save( 'output_images/network.bmp' )

print( time() - start_time , "sec" )
