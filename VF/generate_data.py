from PIL import Image
import numpy as np
import parameters
import random
import glob
import h5py
import os

dict = parameters.import_parameters( 'PARAMETERS.txt' )

BATCH_SIZE = int( dict['batch_size'] )
IMAGE_SIZE = int( dict['train_image_size'] )
STRIDE = ( IMAGE_SIZE * 2 ) // 5

def crop_images( image1 , image2 , crop_height , crop_width , stride ) :

    image1_height , image1_width , channels = image1.shape

    images1 = []
    images2 = []

    x , y = 0 , 0

    while True :

        y = 0
        while True :
            xx = x
            yy = y
            if x + crop_height > image1_height :
                xx = image1_height - crop_height

            if y + crop_width > image1_width :
                yy = image1_width - crop_width

            images1.append( image1[ xx : xx+crop_height , yy : yy+crop_width , : ] )
            images2.append( image2[ (xx*5)//2 : ((xx+crop_height)*5)//2 , (yy*5)//2 : ((yy+crop_width)*5)//2 , : ] )

            if y + crop_width > image1_width :
                break

            y += stride

        if x + crop_height > image1_height :
            break

        x += stride

    return np.array( images1 ) , np.array( images2 )

other_inp = []
other_tar = []
valid_inp = []
valid_tar = []
test_inp = []
test_tar = []

for file in glob.glob( dict['data_path'] + '/*/*/*.tif' ) :

    image = Image.open( file )
    image_np = np.array( image )
    image.close()

    print(file)

    type = file.split('\\')
    if type[2] == 'target' :
        continue
    file2 = type[0] + '/' + type[1] + '/target/' + type[3]

    image2 = Image.open( file2 )
    image2_np = np.array( image2 )
    image2.close()

    if type[1] == "test" :
        test_inp.append( image_np )
        test_tar.append( image2_np )
    elif type[1] == "valid" :
        valid_inp.append( image_np )
        valid_tar.append( image2_np )
    else :
        a , b = crop_images( image_np , image2_np , IMAGE_SIZE , IMAGE_SIZE , STRIDE )
        other_inp.extend(a)
        other_tar.extend(b)

    del image_np , image2_np

train_inp = []
train_tar = []

A = range( len(other_inp) )
A = random.sample( A , len(A) )

for i in range( len(A) ) :
    train_inp.append( other_inp[ A[i] ] )
    train_tar.append( other_tar[ A[i] ] )

del A , other_inp , other_tar

train_inp = np.array(train_inp)
train_tar = np.array(train_tar)
valid_inp = np.array(valid_inp)
valid_tar = np.array(valid_tar)
test_inp  = np.array(test_inp)
test_tar  = np.array(test_tar)

train_size = train_inp.shape[0]

print("train =>" , train_inp.shape , "->" , train_tar.shape)
print("valid =>" , valid_inp.shape , "->" , valid_tar.shape)
print("test  =>" , test_inp.shape , "->" , test_tar.shape)

x = 0

#os.system('rm -r data')
os.makedirs('data/train')
os.makedirs('data/valid')
os.makedirs('data/test')

while x + BATCH_SIZE <= train_size :

    batch1 = train_tar[x : x + BATCH_SIZE]
    batch2 = train_inp[x : x + BATCH_SIZE]

    x += BATCH_SIZE
    s = x // BATCH_SIZE

    file_path = 'data/train/train_' + str(s//1000) + str( (s//100) % 10 ) + str( (s//10) % 10 ) + str(s%10) + '.h5'

    file = h5py.File( file_path , 'w' )
    file.create_dataset( 'labels' , data = batch1 )
    file.create_dataset( 'inputs' , data = batch2 )
    file.close()

    del batch1 , batch2

del train_tar , train_inp

file = h5py.File( 'data/valid/valid.h5' , 'w' )
file.create_dataset( 'labels' , data = valid_tar )
file.create_dataset( 'inputs' , data = valid_inp )
file.close()

del valid_tar , valid_inp

file = h5py.File( 'data/test/test.h5' , 'w' )
file.create_dataset( 'labels' , data = test_tar )
file.create_dataset( 'inputs' , data = test_inp )
file.close()
