from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
def build_network( input , num_blocks , first_channels , alpha , name = 'Default' ) :

    def conv_layer( input , shp , name , strides = [1,1,1,1] , padding = 'SAME' ) :

        filters = tf.get_variable( name + '/filters' , shp , initializer = tf.truncated_normal_initializer( stddev = 0.05 ) )
        biases = tf.get_variable( name + '/biases' , [ shp[-1] ] , initializer = tf.constant_initializer(0) )
        output = tf.nn.conv2d( input , filters , strides = strides , padding = padding ) + biases
        return output

    def residual_block( input , in_channels , out_channels , name ) :

        conv1 = conv_layer( input , [3,3,in_channels,out_channels] , name + '/conv1' )
        relu1 = tf.nn.relu(conv1)

        conv2 = conv_layer( relu1 , [3,3,out_channels,out_channels] , name + '/conv2' )
        relu2 = tf.nn.relu(conv2)

        if in_channels != out_channels :
            tmp = tf.pad( input , [ [0,0] , [0,0] , [0,0] , [0,out_channels - in_channels] ] , "CONSTANT" )

        output = tmp + relu2

        print(in_channels , '->' , out_channels)

        return output

    current = conv_layer( input , [3,3,3,first_channels] , name + '/first_layer' )
    pre = first_channels
    for i in range(num_blocks) :
        cur = pre + int( ( float(i+1) * alpha ) / num_blocks + 0.5 )
        current = residual_block( current , pre , cur , name + '/residual_block' + str(i) )
        pre = cur

    current = conv_layer( current , [3,3,pre,75] , name + '/last_layer' )
    current = tf.depth_to_space( current , 5 )
    output  = conv_layer( current , [3,3,3,3] , name + '/down_sampling_layer' , strides = (1,2,2,1) )

    return output

def total_variation( images ) :

    tmp = np.array( [ [ [ [-1,0,1] , [-2,0,2] , [-1,0,1] ] ] ] ).astype(np.float32)
    sobel = np.transpose( tmp , (2,3,0,1) )
    sobel_T = np.transpose( sobel , (1,0,2,3) )
    #for RGB channels
    for i in range(3) :
        conv1 = tf.nn.conv2d( images[:,:,:,i:i+1] , tf.constant(  sobel  ) , strides = [1,1,1,1] , padding = 'SAME' )
        conv2 = tf.nn.conv2d( images[:,:,:,i:i+1] , tf.constant( sobel_T ) , strides = [1,1,1,1] , padding = 'SAME' )
        if i > 0 :
            TV += tf.reduce_mean( conv1 * conv1 + conv2 * conv2 )
        else :
            TV  = tf.reduce_mean( conv1 * conv1 + conv2 * conv2 )

    return TV / 3.0

def _tf_fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, size=11, sigma=1.5):

    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = tf.reduce_max(img1)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2

    value = ( (2*mu1_mu2 + C1) * (2*sigma12 + C2) ) / ( (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) )

    return value


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def loss_function( reference , output , color_channels = 3 , loss = 'l2' ) :

    if loss == 'l1' :
        return tf.reduce_mean( tf.abs(reference - output) )
    elif loss == 'l2' :
        return tf.reduce_mean( tf.square( reference - output ) )
    elif loss == 'SSIM' :
        ssim = 0
        for i in range(3) :
            ssim += tf.reduce_mean( tf_ssim( reference[:,:,:,i:i+1] , output[:,:,:,i:i+1] ) )
        return ssim / 3.0
    print("INVALID LOSS FUNCTION")
