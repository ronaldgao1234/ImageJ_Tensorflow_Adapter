import tensorflow as tf
from PIL import Image
from time import time
import numpy as np
from ops import *
import parameters
import glob
import h5py
import sys
import os

dict = parameters.import_parameters( 'PARAMETERS.txt' )

NUM_BLOCKS = int( dict['num_blocks'] )
ALPHA = int( dict['alpha'] )
FIRST_CHANNELS = int( dict['first_channels'] )
SCALE = int( dict['scale'] )

def PSNR( ref , img ) :
    maxi = np.amax( ref )
    mse = np.mean( ( ref - img ) ** 2 )
    return mse , 10 * np.log10( maxi * maxi / mse )

def SSIM( ref , img , sess ) :

    ref = tf.constant( np.array( [ np.array(ref) ] ) )
    img = tf.constant( np.array( [ np.array(img) ] ) )
    ssim = loss_function( ref , img , loss = 'SSIM' )
    return sess.run(ssim)

input = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )
label = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )
network_output = build_network( input , NUM_BLOCKS , FIRST_CHANNELS , ALPHA )

shape = tf.placeholder( tf.int32 , [2] )
bicubic_interpolation = tf.image.resize_bicubic( input , shape )

with tf.Session() as sess :

    saver = tf.train.Saver()
    model_path = 'Models/best_model'
    saver.restore( sess , model_path )

    file = h5py.File( 'data/test/test.h5' )
    inputs = file['inputs'][:].astype(np.float32)
    labels = file['labels'][:].astype(np.float32)
    file.close()

    os.system('rm -r output_images')
    os.system('mkdir output_images')

    sum_bicubic_mse = 0; sum_bicubic_psnr = 0;
    sum_network_mse = 0; sum_network_psnr = 0;
    sum_bicubic_ssim = 0;sum_network_ssim = 0;
    sz = inputs.shape[0]

    for i in range( sz ) :

        network_images , bicubic_images = sess.run( [ network_output , bicubic_interpolation ] , feed_dict = { input : inputs[i:i+1] , shape : ( labels.shape[1] , labels.shape[2] ) } )

        network = np.clip( network_images[0] , 0 , 255 ).astype('uint8')
        bicubic = np.clip( bicubic_images[0] , 0 , 255 ).astype('uint8')

        os.makedirs( 'output_images/' + str(i+1) )
        Image.fromarray( inputs[i].astype('uint8') ).save( 'output_images/' + str(i+1) + '/input.tif'  )
        Image.fromarray( labels[i].astype('uint8') ).save( 'output_images/' + str(i+1) + '/target.tif' )
        Image.fromarray( network ).save( 'output_images/' + str(i+1) + '/network.tif' )
        Image.fromarray( bicubic ).save( 'output_images/' + str(i+1) + '/bicubic.tif' )

        bicubic_mse , bicubic_psnr = PSNR( labels[i] , bicubic )
        bicubic_ssim = SSIM( labels[i] , bicubic.astype(np.float32) , sess )

        sum_bicubic_mse += bicubic_mse
        sum_bicubic_psnr += bicubic_psnr
        sum_bicubic_ssim += bicubic_ssim

        network_mse , network_psnr = PSNR( labels[i] , network )
        network_ssim = SSIM( labels[i] , network.astype(np.float32) , sess )

        sum_network_mse += network_mse
        sum_network_psnr += network_psnr
        sum_network_ssim += network_ssim

        print(i+1 , "bicubic => MSE :" , bicubic_mse , "\tPSNR :" , bicubic_psnr , "\tSSIM :" , bicubic_ssim)
        print(i+1 , "network => MSE :" , network_mse , "\tPSNR :" , network_psnr , "\tSSIM :" , network_ssim)
        print("")
        sys.stdout.flush()

    print( "Average MSE (bicubic)  over" , sz , "images :" , sum_bicubic_mse / float(sz) )
    print( "Average PSNR (bicubic)  over" , sz , "images :" , sum_bicubic_psnr / float(sz) )
    print( "Average SSIM (bicubic)  over" , sz , "images :" , sum_bicubic_ssim / float(sz) )

    print( "\nAverage MSE (network)  over" , sz , "images :" , sum_network_mse / float(sz) )
    print( "Average PSNR (network)  over" , sz , "images :" , sum_network_psnr / float(sz) )
    print( "Average SSIM (network)  over" , sz , "images :" , sum_network_ssim / float(sz) )
