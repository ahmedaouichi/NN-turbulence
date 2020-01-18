import tensorflow as tf
import numpy as np

a = np.ones([10,9])
b = np.ones([10])
c = np.zeros([10,9])

A = tf.constant(a)
B = tf.constant(b, dtype=tf.float32)
C = tf.Variable(c)

C = tf.cast(C, tf.float32)

a = tf.Variable(tf.ones([1,10,9]),dtype = tf.float32) 

sess = tf.InteractiveSession()
for i in range(0,10):
    for j in range(0,9):
        a = a[0,i,j].assign(B[i].eval() * a[0,i,j].eval())
    

#sess = tf.InteractiveSession()

#for ii in range(0,9):
#    x = tf.cast(tf.scalar_mul(B[ii], A[ii,:]), dtype=tf.float32)
#    value1 = tf.cast(tf.Variable(tf.ones([1,9])), dtype=tf.float32)
#    value1 *= 2
#    
#    e = tf.scatter_nd_update(a,[[0,ii]],value1)
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    sess.run(e)
##          
         
    
    
#    my_var = my_var[4:8].assign(tf.zeros(4))
#    

    


#a = tf.Variable(tf.zeros([10,36,36])) 
#value1 = np.random.randn(1,36)
#e = tf.scatter_nd_update(a,[[0,1]],value1)
#init= tf.global_variables_initializer()
#sess.run(init)
#print(a.eval())
#sess.run(e)
    
#import tensorflow as tf
#num = tf.zeros( shape = ( 5, 3 ), dtype = tf.float32 )
## Looping variable
#i = tf.zeros( shape=(), dtype=tf.int32)
## Conditional
#c = lambda i, num: tf.less(i, 2)
#def body(i, num):
#    # Update values
#    updates = tf.ones([1, 3], dtype=tf.float32)
#    num_shape = num.get_shape()
#    num = tf.concat( [ num[ : i ], updates, num[ i + 1 : ] ], axis = 0 )
#    num.set_shape( num_shape )
#    return tf.add(i, tf.ones( shape=(), dtype = tf.int32 ) ), num
#i, num = tf.while_loop( c, body, [ i, num ] )
## Session
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    num_out = sess.run( [ num ] )
#    print(num_out)