import tensorflow as tf
from PIL import Image
import numpy as np
from ops import *
import parameters
import glob
import os

dict = parameters.import_parameters( 'PARAMETERS.txt' )

NUM_BLOCKS = int( dict['num_blocks'] )
ALPHA = int( dict['alpha'] )
FIRST_CHANNELS = int( dict['first_channels'] )

with tf.Graph().as_default() , tf.device('/cpu:0') :

    input = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )
    network_output = build_network( input , NUM_BLOCKS , alpha = ALPHA , first_channels = FIRST_CHANNELS )

    shape = tf.placeholder( tf.int32 , [2] )
    bicubic_interpolation = tf.image.resize_bicubic( input , shape )

    with tf.Session() as sess :

        saver = tf.train.Saver()
        model_path = 'Models/best_model'
        saver.restore( sess , model_path )

        os.system('rm -r output_images')
        os.system('mkdir output_images')

        for file in glob.glob( 'images/*' ) :

            file_path = 'output_images/' + file.split('\\')[1].split('.')[0]
            os.makedirs( file_path )

            img = Image.open(file)
            inputs = np.array( [ np.array(img) ] ).astype(np.float32)
            img.close()

            inputs = inputs[:,:,:,:3]
            network_images , bicubic_images = sess.run( [ network_output , bicubic_interpolation ] , feed_dict = { input : inputs , shape : ( (inputs.shape[1]*5)//2 , (inputs.shape[2]*5)//2 ) } )

            network = np.clip( network_images[0] , 0 , 255 ).astype('uint8')
            bicubic = np.clip( bicubic_images[0] , 0 , 255 ).astype('uint8')

            Image.fromarray( network ).save( file_path + '/network.tif' )
            Image.fromarray( bicubic ).save( file_path + '/bicubic.tif' )
