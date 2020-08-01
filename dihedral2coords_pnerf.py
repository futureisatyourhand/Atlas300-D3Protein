import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np


pi=3.141592653589793115997963468544
BS = 64
atom_num = 128
classes= 100
num_dimensions=3
num_fragments=6
num_dihedrals=3


def sin_tai(inputs):
    return inputs-tf.math.pow(inputs,3)/6.0+tf.math.pow(inputs,5)/120.-tf.math.pow(inputs,7)/5040.

###########################################################################

def cos_tai(inputs):
    
    return 1.0-tf.math.pow(inputs,2)/2.+tf.math.pow(inputs,4)/24.-tf.math.pow(inputs,6)/720.#+tf.math.pow(inputs,8)/400320.

###########################################################################


def reduce_mean_angle(probs, alphabet):
    #sin=sin_tai(alphabet)
    #cos=cos_tai(alphabet)
    #probs = tf.reshape(probs, (atom_num*BS, classes))
    y_coords=tf.matmul(probs, alphabet)
    x_coords=tf.matmul(probs, alphabet)
    
    angle=tf.math.atan(tf.realdiv(y_coords, x_coords))
    angle = tf.reshape(angle, (atom_num, BS, 3))
    print ("----------", probs.shape, alphabet.shape, angle.shape, angle.name)

    '''
    constant=0.0
    x_less, x_more=tf.cast(tf.math.less(x_coords,constant), dtype=tf.float32), tf.cast(tf.math.greater_equal(x_coords,constant),dtype=tf.float32)
    y_less, y_more=tf.cast(tf.math.less(y_coords,constant),dtype=tf.float32),tf.cast(tf.math.greater_equal(y_coords,constant),dtype=tf.float32)
    
    ####torch.atan2
    pi_2 = pi / 2.
    add_inds = tf.math.multiply(x_less,y_more)
    angle[add_inds] = angle + pi_2 * add_inds
    reduce_inds = tf.math.multiply(x_less,y_less)
    angle[reduce_inds] = angle - pi_2 * reduce_inds
    '''
    return angle

def matmul3(x, y):
    l, m, n=tf.split(x, 3, axis=-2)
    l = tf.reshape(l, (-1, 1, 3))
    m = tf.reshape(m, (-1, 1, 3))
    n = tf.reshape(n, (-1, 1, 3))
    y = tf.reshape(y, (-1, 1, 3))
    l = tf.reduce_sum(l*y, axis=-1)
    m = tf.reduce_sum(m*y, axis=-1)
    n = tf.reduce_sum(n*y, axis=-1)
    z = tf.concat((l, m, n), axis=-1)
    z = tf.reshape(z, (-1, 1, 3, 1))
    return z
    
    
###########################################################################

def extend(coords_a,coords_b,coords_c, point, multi_m):
    #F.normalize(prev_three_coords.c - prev_three_coords.b,axis=-1)
    #bc = tf.nn.l2_normalize(coords_c-coords_b, axis=-1, epsilon=1e-6)
    print('extend:', coords_a.shape, coords_b.shape, coords_c.shape, point.shape)
    bc = coords_c-coords_b
    bc_r = tf.math.reduce_sum(tf.math.pow(bc,2),axis=-1)
    bc_r = tf.reshape(bc_r, (-1,BS,1))
    bc_r = tf.tile(bc_r, (1,1,3))
    bc_r = tf.realdiv(bc, tf.math.maximum(tf.math.sqrt(bc_r),1e-7))
    
    #torch.cross(prev_three_coords.b - prev_three_coords.a, bc)
    ba_reduce = coords_b - coords_a
    l, m, n=tf.split(ba_reduce, 3, axis=-1)
    o, p, q=tf.split(bc_r, 3, axis=-1)
    x=tf.multiply(m,q)-tf.multiply(n,p)
    y=tf.multiply(n,o)-tf.multiply(l,q)
    z=tf.multiply(l,p)-tf.multiply(m,o)
    cross=tf.concat((x,y,z),axis=-1)
    #n = tf.nn.l2_normalize(cross, axis=-1, epsilon=1e-12)
    #cross_reduce = tf.math.reduce_mean(tf.math.pow(cross,2),axis=-1)
    cross_reduce = tf.math.reduce_sum(tf.math.pow(cross,2), axis=-1)
    cross_reduce = tf.reshape(cross_reduce, (-1,BS,1))
    cross_reduce = tf.tile(cross_reduce, (1,1,3))
    n=tf.realdiv(cross,tf.math.maximum(tf.math.sqrt(cross_reduce),1e-7))

    #if multi_m:
    ###torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2,3,0)
    x1,x2,x3=tf.split(n, 3, axis=-1)
    xx=tf.multiply(x2,q)-tf.multiply(x3,p)
    yy=tf.multiply(x3,o)-tf.multiply(x1,q)
    zz=tf.multiply(x1,p)-tf.multiply(x2,o)
    xyz=tf.concat((xx,yy,zz),axis=-1)
    m = tf.stack([bc_r, xyz, n])

    if multi_m:
        m=tf.transpose(m, (1,2,3,0))
    else:
        m=tf.transpose(m, (1,2,3,0))
        ##repeat
        m=tf.tile(m, (point.shape[0],1,1,1))

    #ba=tf.math.reduce_mean(tf.math.pow(ba_reduce,2),axis=-1)*dimension
    mat = matmul3(m, tf.reshape(point, (-1, BS, 3, 1)))
    mat = tf.reshape(mat, (-1, BS, 3))


    #print('shapes', mat.shape, coords_c.shape)
    coord = mat + coords_c
    return coord


def extend1(coords_a,coords_b,coords_c, point, multi_m):
    #F.normalize(prev_three_coords.c - prev_three_coords.b,axis=-1)
    #bc = tf.nn.l2_normalize(coords_c-coords_b, axis=-1, epsilon=1e-6)
    print('extend:', coords_a.shape, coords_b.shape, coords_c.shape, point.shape)
    bc = coords_c-coords_b
    bc_r = tf.math.reduce_sum(tf.math.pow(bc,2),axis=-1)
    bc_r = tf.reshape(bc_r, (BS,1))
    bc_r = tf.tile(bc_r, (1,3))

    print ("========", bc.shape, bc_r.shape)
    bc_r = tf.realdiv(bc, tf.math.maximum(tf.math.sqrt(bc_r),1e-7))
    
 
    #torch.cross(prev_three_coords.b - prev_three_coords.a, bc)
    ba_reduce = coords_b - coords_a
    l, m, n=tf.split(ba_reduce, 3, axis=-1)
    o, p, q=tf.split(bc_r, 3, axis=-1)
    x=tf.multiply(m,q)-tf.multiply(n,p)
    y=tf.multiply(n,o)-tf.multiply(l,q)
    z=tf.multiply(l,p)-tf.multiply(m,o)
    cross=tf.concat((x,y,z),axis=-1)

    #n = tf.nn.l2_normalize(cross, axis=-1, epsilon=1e-12)
    #cross_reduce = tf.math.reduce_mean(tf.math.pow(cross,2),axis=-1)
    cross_reduce = tf.math.reduce_sum(tf.math.pow(cross,2), axis=-1)
    cross_reduce = tf.reshape(cross_reduce, (BS,1))
    cross_reduce = tf.tile(cross_reduce, (1,3))
    n=tf.realdiv(cross,tf.math.maximum(tf.math.sqrt(cross_reduce),1e-7))

    #if multi_m:
    ###torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2,3,0)
    x1,x2,x3=tf.split(n, 3, axis=-1)
    xx=tf.multiply(x2,q)-tf.multiply(x3,p)
    yy=tf.multiply(x3,o)-tf.multiply(x1,q)
    zz=tf.multiply(x1,p)-tf.multiply(x2,o)
    xyz=tf.concat((xx,yy,zz),axis=-1)
    m = tf.stack([bc_r, xyz, n])
    m=tf.transpose(m, (1,2,0))
    ##repeat
    m = tf.tile(m, (point.shape[0],1,1))
    m = tf.reshape(m, list(point.shape) + [3,])

    #ba=tf.math.reduce_mean(tf.math.pow(ba_reduce,2),axis=-1)*dimension
    mat = matmul3(m, tf.expand_dims(point, 3))
    mat = tf.reshape(mat, (-1, BS, 3))
    coords_c = tf.tile(tf.expand_dims(coords_c, 0), (mat.shape[0],1,1))

    #print('shapes', mat.shape, coords_c.shape)
    coord = mat + coords_c
    return coord













##-----dihedral_to_point---
with tf.Session(graph=tf.Graph()) as sess:
    #inputs=tf.constant([[[55,66,77]],[[11,22,33]],[[19.9,10,20]],[[1.23,2.31,3.231]]])##values from reduce_mean_angle
    #data=tf.Variable([[[55,66,77]],[[11,22,33]],[[19.9,10,20]],[[1.23,2.31,3.231]]])##values from reduce_mean_angle

    ##constant
    num_steps = atom_num
    batch_size = BS
    alphabet = tf.constant([[ 0.7771,  0.7035,  0.4772]]*classes)
    #alphabet = tf.tile(alphabet, (classes,1))
    bond_lengths=tf.constant([145.801, 152.326, 132.868])
    bond_angles=tf.constant([2.124, 1.941, 2.028])
    r_cos_theta =tf.constant([[[76.6061, 55.1123, 58.6533]]])
    r_sin_theta=tf.constant([[[124.0541, 142.0065, 119.2212]]])
    r_cos_theta = tf.tile(r_cos_theta, (batch_size,num_steps, 1))
    r_sin_theta = tf.tile(r_sin_theta, (batch_size,num_steps, 1))



    ###################### reduce_mean_angle
    inputs = tf.placeholder(tf.float32, shape=(2, BS, atom_num,  3), name='data')
    x, y = tf.split(inputs, axis=0, num_or_size_splits=2)
    #inputs1 = tf.placeholder(tf.float32, shape=( atom_num, BS, 3), name='data1')
    idiv = x / y
    angle = tf.math.atan(idiv)
    angle = tf.squeeze(angle)
    print ('inputs:', inputs.shape)

    #inputs = reduce_mean_angle(probs, alphabet)


    ######################################################
    point_x = r_cos_theta
    tcos = cos_tai(angle)
    tsin = sin_tai(angle)
    point_y = tcos * r_sin_theta
    point_z = tsin * r_sin_theta

    point=tf.stack([point_x, point_y, point_z])
    print (point.shape)
    point_perm=tf.transpose(point,(2, 3, 1, 0))
    point_final=tf.reshape(point_perm,(num_steps*num_dihedrals,batch_size,num_dimensions))

##-----point_to_coordinate---
    
    a1=tf.constant([[[-0.70710677,1.2247449,0.0]]])
    b1=tf.constant([[[-1.4142135,0.0,0.0]]])
    c1=tf.constant([[[0.0,0.0,0.0]]])
    Triplet_a = tf.tile(a1, (num_fragments, batch_size,1))
    Triplet_b = tf.tile(b1, (num_fragments, batch_size,1))
    Triplet_c = tf.tile(c1, (num_fragments, batch_size,1))

    #for fow in init_matrix:
    #total_num_angles=point_final.shape[0]
    #padding=(num_fragments-total_num_angles%num_fragments)%num_fragments
    #points=tf.pad(point_final,( (0, 0), (0, 0), (0, padding)))
    points=point_final
     
    points=tf.reshape(points,(num_fragments, -1, batch_size,num_dimensions))
    points=tf.transpose(points,(1, 0, 2, 3))
    #points=tf.split(points, axis=0, num_or_size_splits=points.shape[0])


    #coord = extend(Triplet_a,Triplet_b,Triplet_c, points[0], True)



    print('FINISH::', points.name) 
    print(sess.graph_def)
    sess.run(tf.global_variables_initializer())
    #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['add_14'])
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['transpose_1'])
    with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    exit()


   




    print ('Iterate over FRAG_SIZE:', len(points), points[0].shape)
    coords_list = []
    for i in range(len(points)):## Iterate over FRAG_SIZE
        coord = extend(Triplet_a,Triplet_b,Triplet_c, points[i], True)
        coords_list.append(coord) 
        Triplet_a,Triplet_b,Triplet_c=Triplet_b,Triplet_c,coord
        if i > 6:
            break
    
    coords_pretrans=tf.transpose(tf.stack(coords_list),(1,0,2,3))
    # (coords_trans) to be aligned with the next fragment
    print ('coords_pretrans:', coords_pretrans.name)





    ######################################################
    ######################################################


    print ('coords_trans:', coords_pretrans.shape, Triplet_c.shape) 
    coords_pretrans = tf.split(coords_pretrans, axis=0, num_or_size_splits=coords_pretrans.shape[0]) 
    coords_pretrans = [tf.squeeze(t) for t in coords_pretrans]
    coords_trans = coords_pretrans[-1]
    Triplet_a = tf.split(Triplet_a, axis=0, num_or_size_splits=num_fragments) 
    Triplet_b = tf.split(Triplet_b, axis=0, num_or_size_splits=num_fragments) 
    Triplet_c = tf.split(Triplet_c, axis=0, num_or_size_splits=num_fragments) 

    
    print('====', len(coords_pretrans)-1)
    for i in reversed(range(len(coords_pretrans)-1)):
        print("step2 : ", i)
        transformed_coords = extend1(tf.squeeze(Triplet_a[i]), tf.squeeze(Triplet_b[i]), tf.squeeze(Triplet_c[i]), coords_trans, False)
        coords_trans=tf.concat((coords_pretrans[i], transformed_coords), axis=0)



    #coords=tf.pad(coords_trans[:total_num_angles-1], (0, 0, 0, 0, 1, 0))
    coords = coords_trans










