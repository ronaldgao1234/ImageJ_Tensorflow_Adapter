import tensorflow as tf
from time import time
from PIL import Image
import numpy as np
import parameters
import glob
import h5py
import ops
import sys

dict = parameters.import_parameters( 'PARAMETERS.txt' )

NUM_EPOCHS = int( dict['num_epochs'] )
LEARNING_RATE = float( dict['learning_rate'] )
TV_WEIGHT = float( dict['tv_weight'] )
NUM_BLOCKS = int( dict['num_blocks'] )
BATCH_SIZE = int( dict['batch_size'] )
FIRST_CHANNELS = int( dict['first_channels'] )
ALPHA = int( dict['alpha'] )
SCALE = int( dict['scale'] )

def average_gradients( tower_grads ) :

    average_grads = []
    for grads_and_vars in zip( *tower_grads ) :
        grads = []
        for g , _ in grads_and_vars :
            grads.append( tf.expand_dims(g,0) )

        grad = tf.concat( 0 , grads )
        grad = tf.reduce_mean( grad , 0 )
        average_grads.append( (grad,grads_and_vars[0][1]) )

    return average_grads

def get_results( sess , file_path ) :

    file = h5py.File( file_path )
    inputs = file['inputs'][:].astype(np.float32)
    labels = file['labels'][:].astype(np.float32)
    file.close()

    sa = 0; sb = 0; sc = 0; sz = inputs.shape[0]
    for i in range( sz ) :
        a , b , c = sess.run( [ total_loss_T , tv_loss_T , mse_T ] , feed_dict = { input : inputs[i:i+1] , label : labels[i:i+1] } )
        sa += a; sb += b; sc += c;
    return sa / float(sz) , sb / float(sz) , sc / float(sz)

with tf.Graph().as_default() , tf.device('/cpu:0') :

    input = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )
    label = tf.placeholder( tf.float32 , [ None , None , None , 3 ] )

    devices = ops.get_available_gpus()
    print(devices)
    NUM_GPU = len(devices)
    optimizer = tf.train.AdamOptimizer( LEARNING_RATE )
    tower_grads = []
    tv_loss_T = 0; mse_T = 0; total_loss_T = 0;
    for i in range(NUM_GPU) :
        with tf.device( devices[i] ) :

            x = i * ( BATCH_SIZE // NUM_GPU )
            y = (i+1) * ( BATCH_SIZE // NUM_GPU )

            network_output = ops.build_network( input[x:y,:,:,:] , NUM_BLOCKS , FIRST_CHANNELS , ALPHA )
            tv_loss = ops.total_variation( network_output )
            mse = ops.loss_function( label[x:y,:,:,:] , network_output , loss = 'l2' )
            total_loss = mse + TV_WEIGHT * tv_loss

            tf.get_variable_scope().reuse_variables()
            gradients = optimizer.compute_gradients( total_loss )
            tower_grads.append(gradients)

    with tf.device( devices[0] ) :
        network_output_T = ops.build_network( input , NUM_BLOCKS , FIRST_CHANNELS , ALPHA )
        tv_loss_T = ops.total_variation( network_output_T )
        mse_T = ops.loss_function( label , network_output_T , loss = 'l2' )
        total_loss_T = mse_T + TV_WEIGHT * tv_loss_T

    avg_grads = average_gradients( tower_grads )
    train_step = optimizer.apply_gradients( avg_grads )

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
