import tensorflow as tf
from time import time
from PIL import Image
import numpy as np
from ops import *
import parameters
import glob
import h5py
import sys

dict = parameters.import_parameters( 'PARAMETERS.txt' )

NUM_EPOCHS = int( dict['num_epochs'] )
LEARNING_RATE = float( dict['learning_rate'] )
TV_WEIGHT = float( dict['tv_weight'] )
NUM_BLOCKS = int( dict['num_blocks'] )
FIRST_CHANNELS = int( dict['first_channels'] )
ALPHA = int( dict['alpha'] )
SCALE = int( dict['scale'] )

def get_results( sess , file_path ) :

    file = h5py.File( file_path )
    inputs = file['inputs'][:].astype(np.float32)
    labels = file['labels'][:].astype(np.float32)
    file.close()

    sa = 0; sb = 0; sc = 0; sz = inputs.shape[0]
    for i in range( sz ) :
        a , b , c = sess.run( [ total_loss , tv_loss , mse ] , feed_dict = { input : inputs[i:i+1] , label : labels[i:i+1] } )
        sa += a; sb += b; sc += c;
    return sa / float(sz) , sb / float(sz) , sc / float(sz)

with tf.variable_scope(tf.get_variable_scope()) as scope:
    for i in range(2):
        input = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )
        label = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )

        network_output = build_network( input , NUM_BLOCKS , FIRST_CHANNELS , ALPHA )
        tf.get_variable_scope().reuse_variables()

tv_loss = total_variation( network_output )
mse  = loss_function( label , network_output , loss = 'l2' )

total_loss = mse + TV_WEIGHT * tv_loss

train_step = tf.train.AdamOptimizer( LEARNING_RATE ).minimize(total_loss)
saver = tf.train.Saver()

results_file = open( 'results.txt' , 'w' )
best_cost = float('inf')
best_index = 0

with tf.Session() as sess :

    sess.run( tf.global_variables_initializer() )
    start_time = time()

    for epoch in range( NUM_EPOCHS ) :
        sum_train_loss = 0
        count = 0
        for file_path in glob.glob('data/train/*') :

            file = h5py.File(file_path , 'r')
            inputs = file['inputs'][:].astype(np.float32)
            labels = file['labels'][:].astype(np.float32)
            file.close()

            _ , train_loss = sess.run( [ train_step , total_loss ] , feed_dict = { input : inputs , label : labels } )
            sum_train_loss += train_loss
            count += 1

            del inputs , labels

        if epoch % 10 == 0 :
            a , b , c = get_results( sess , 'data/test/test.h5' )
            asdf  = "\nTest Results :\n\n"
            asdf += "test_loss  : " + str( '%.2lf' % a ) + '\n'
            asdf += "TV         : " + str( '%.2lf' % b ) + '\n'
            asdf += "MSE        : " + str( '%.2lf' % c ) + '\n'
            asdf += "BEST_INDEX : " + str(  best_index ) + '\n'
            print(asdf)
            results_file.write( asdf + '\n' )
            results_file.flush()

        a , b , c = get_results( sess , 'data/valid/valid.h5' )

        asdf  = "epoch = " + str(epoch + 1) + " train_loss = " + str( '%.2lf' % ( sum_train_loss / count ) ) + " valid_loss = " + str( '%.2lf' % a )
        asdf += " tv_loss = " + str( '%.2lf' % b ) + " MSE = " + str( '%.2lf' % c ) + " time = " + str( int( time() - start_time ) ) + " sec"
        print(asdf)
        sys.stdout.flush()
        results_file.write(asdf + '\n')
        results_file.flush()

        if a < best_cost :
            best_cost = a
            best_index = epoch + 1
            saver.save( sess , 'Models/best_model' )
