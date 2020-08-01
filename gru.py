import tensorflow as tf
##inputs:[-1,16,batchsize,128]
data_x=tf.ones((2,128,16,1),dtype=tf.float32)
data_h=tf.ones((2,128,16,1),dtype=tf.float32)
batchsize=data_x.shape[-2]
input_x=tf.transpose(data_x,(2,0,1,3))#16,2,128,1
input_h=tf.transpose(data_h,(2,0,1,3))#16,2,128,1
dim0,batchsize=int(input_x.shape[0]),int(input_x.shape[1])
input_channel=128
"""
w_ir=tf.placeholder(tf.float32, shape=(128,16))
b_ir=tf.placeholder(tf.float32,shape=(128))

w_hr=tf.placeholder(tf.float32,shape=(128,16))
b_hr=tf.placeholder(tf.float32,shape=(128))

w_iz=tf.placeholder(tf.float32,shape=(128,16))
b_iz=tf.placeholder(tf.float32,shape=(128,16))

w_hz=tf.placeholder(tf.float32,shape=(128,16))
b_hz=tf.placeholder(tf.float32,shape=(128,16))

w_in=tf.placeholder(tf.float32,shape=(128,16))
w_hn=tf.placeholder(tf.float32,shape=(128,16))
b_hn=tf.placeholder(tf.float32,shape=(128))
"""

#16,128,2
w_ir=tf.random_normal((dim0,batchsize,1,input_channel),dtype=tf.float32)
b_ir=tf.ones((128),dtype=tf.float32)

w_hr=tf.random_normal((dim0,batchsize,1,input_channel),dtype=tf.float32)
b_hr=tf.ones((128),dtype=tf.float32)

w_iz=tf.random_normal((dim0,batchsize,1,input_channel),dtype=tf.float32)
b_iz=tf.ones((128),dtype=tf.float32)

w_hz=tf.random_normal((dim0,batchsize,1,input_channel),dtype=tf.float32)
b_hz=tf.ones((128),dtype=tf.float32)

w_in=tf.random_normal((dim0,batchsize,1,input_channel),dtype=tf.float32)
b_in=tf.ones(128,dtype=tf.float32)

w_hn=tf.random_normal((dim0,batchsize,1,input_channel),dtype=tf.float32)
b_hn=tf.ones((128),dtype=tf.float32)


r=tf.sigmoid(tf.matmul(w_ir,input_x)+b_ir+tf.matmul(w_hr,input_h)+b_hr)
z=tf.sigmoid(tf.matmul(w_iz,input_x)+b_iz+tf.matmul(w_hz,input_h)+b_hz)
n=tf.tanh(tf.matmul(w_in,input_x)+b_in+tf.multiply(r,(tf.matmul(w_hn,input_h)+b_hn)))
h2=tf.multiply(tf.subtract(tf.constant(1.0),z),n)+tf.multiply(z,input_h)

with tf.Session() as sess:
    print(sess.run(h2))
