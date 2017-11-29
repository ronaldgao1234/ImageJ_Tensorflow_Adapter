from tensorflow.python.saved_model import builder as saved_model_builder

import tensorflow as tf

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")
feed_dict ={w1:4,w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Run the operation by feeding input
#print(sess.run(w4,feed_dict))
#Prints 24 which is sum of (w1+w2)*b1

#save our model
builder = tf.saved_model.builder.SavedModelBuilder("./Models_const")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save(True)
