import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np

atom_num = 32
bond_pad_num = 128
max_atom = 32
BS = 8


#(4+5*BS+6)
def gather(atom, bond_index):
    ### gather (bs, atom_num, 128) (bs, 128, 2)
    src, dst = tf.split(bond_index, axis=2, num_or_size_splits=2)
    src_bonds = tf.split(src, axis=0, num_or_size_splits=BS)
    dst_bonds = tf.split(dst, axis=0, num_or_size_splits=BS)
    
    atoms = tf.split(atom, axis=0, num_or_size_splits=BS)
    neighbors = []
    targets = []


    #atom = tf.squeeze(atom[0])
    #index = tf.squeeze(dst_bonds[0])
    #res = tf.gather(atom, index, axis=0)
  
    
    for i in range(BS):
        atom = tf.squeeze(atoms[i])
        dst_bond = tf.squeeze(dst_bonds[i])
        src_bond = tf.squeeze(src_bonds[i])
        neighbors.append( tf.gather(atom, dst_bond, axis=0) )
        targets.append( tf.gather(atom, src_bond, axis=0) )
        #neighbors.append( tf.gather(atoms[i], dst_bonds[i], axis=1) )
        #targets.append( tf.gather(atoms[i], src_bonds[i], axis=1) )
    

    neighbors = tf.reshape( tf.concat(neighbors, axis=0), (-1,128) )
    targets = tf.reshape( tf.concat(targets, axis=0), (-1,128) )
    

    row_num = []
    line = src.shape[1]
    ### here generate the row index offset for flatten src to index
    for i in range(BS):
        for j in range(line):
            row_num.append([i*max_atom])

    row_num = tf.constant(row_num, dtype=tf.float32)
    scatter = row_num + tf.cast(tf.reshape(src, (-1,1)), dtype=tf.float32)
    scatter = tf.cast(scatter, dtype=tf.int32)
    print("gather:, ", atoms[0].shape, dst_bonds[0].shape, neighbors.shape, scatter.shape)
    
    return neighbors, targets, src, scatter
        
      

#5 ops
def index2seg(index, num=max_atom):
    seg_list = []
    index = tf.expand_dims(index, axis=0)
    index = tf.tile(index, (num,1,1,1))
    
    for i in range(num):
        seg_list.append([[[i] for t in range(128)] for s in range(BS)])

    mat = tf.constant(seg_list)
    gt = tf.less(mat, index) 
    ls = tf.less(index, mat) 
    mat = tf.cast(~(ls|gt), tf.float32)	   ## the tf.equal has bug, so we use ~(ls|gt) instead~
    mat = tf.reshape(mat, (num, -1))
    #mat = tf.squeeze(mat)
    print("index2seg:", index.shape, mat.shape)
    return mat
 
   
#9 ops
def scatter_max(bond, indexs, scatter):
    atom_num = indexs.shape[0]
    bond = tf.reshape(bond, (1, -1))
    bonds = tf.tile(bond, (atom_num, 1))
    vals = bonds*indexs - (1.0-indexs)*999.9
    vals = tf.reshape(vals, (atom_num, BS, -1))
    segs = tf.reduce_max(vals, axis=-1)
    segs = tf.transpose(segs, (1,0))
    segs = tf.reshape(segs, (-1, 1))
    res = tf.gather(segs, scatter)
    #(1, 8, 128) (32, 8, 128) (256, 1) (1024, 1) (1024, 1, 1) (256, 128) (8, 128)
    print("scatter:", bond.shape, indexs.shape, segs.shape, scatter.shape, res.shape)
    return tf.reshape(res, (BS, -1)) 



#9 ops
def scatter_sum(bond, indexs, scatter):
    atom_num = indexs.shape[0]
    bond = tf.reshape(bond, (1, -1))
    bonds = tf.tile(bond, (atom_num, 1))
    vals = bonds*indexs
    vals = tf.reshape(vals, (atom_num, BS, -1))
    segs = tf.reduce_sum(vals, axis=-1)
    segs = tf.transpose(segs, (1,0))
    segs = tf.reshape(segs, (-1, 1))
    res = tf.gather(segs, scatter)
    print("scatter:", bond.shape, indexs.shape, segs.shape, scatter.shape, res.shape)
    return tf.reshape(res, (BS, -1)) 


#9 ops
def scatter_sum_only(bond, indexs):
    #(1024, 128), (32,1024)
    atom_num = indexs.shape[0]
    bond = tf.expand_dims(bond, axis=0)
    bonds = tf.tile(bond, (atom_num, 1, 1))
    #bonds = tf.reshape(bonds, (atom_num*BS, -1))

    indexs = tf.expand_dims(indexs, axis=-1)
    indexs = tf.tile(indexs, (1, 1, 128))
    indexs = tf.reshape(indexs, (atom_num, -1, 128))
    vals = bonds*indexs
    vals = tf.reshape(vals, (atom_num, BS, -1, 128))
    
    segs = tf.reduce_sum(vals, axis=2)
    segs = tf.transpose(segs, (1,0,2))
    segs = tf.reshape(segs, (-1, 128))
    print("over:", bond.shape, indexs.shape, vals.shape, segs.shape)
    return segs



#3 ops
def matmul128(a, b):
    a = tf.tile(tf.expand_dims(a, axis=1), (1,128,1))
    b = tf.transpose(b, (0,2,1))
    print ('matmul128:', a.shape, b.shape)
    mat = tf.reduce_sum(a*b, axis=2)
    return mat 

       

#21 ops
def gru(input_x, input_h, prefix):
    ## (256, 128) (128, 128) (128,) (256, 128) (128, 128) (128,)
    wri = tf.constant(np.load(prefix+'gru.weight_ih.npy')[:128].transpose((1,0)), dtype=tf.float32)
    wzi = tf.constant(np.load(prefix+'gru.weight_ih.npy')[128:256].transpose((1,0)), dtype=tf.float32)
    wni = tf.constant(np.load(prefix+'gru.weight_ih.npy')[256:].transpose((1,0)), dtype=tf.float32)

    wrh = tf.constant(np.load(prefix+'gru.weight_hh.npy')[:128].transpose((1,0)), dtype=tf.float32)
    wzh = tf.constant(np.load(prefix+'gru.weight_hh.npy')[128:256].transpose((1,0)), dtype=tf.float32)
    wnh = tf.constant(np.load(prefix+'gru.weight_hh.npy')[256:].transpose((1,0)), dtype=tf.float32)

    br = tf.constant(np.load(prefix+'gru.bias_ih.npy')[:128]+np.load(prefix+'gru.bias_hh.npy')[:128] \
        	, dtype=tf.float32)
    bz = tf.constant(np.load(prefix+'gru.bias_ih.npy')[128:256]+np.load(prefix+'gru.bias_hh.npy')[128:256] \
        	, dtype=tf.float32)
    #bzi = tf.constant(np.load(prefix+'gru.bias_ih.npy')[:128], dtype=tf.float32)
    #bri = tf.constant(np.load(prefix+'gru.bias_ih.npy')[128:256], dtype=tf.float32)
    bni = tf.constant(np.load(prefix+'gru.bias_ih.npy')[256:], dtype=tf.float32)

    #bzh = tf.constant(np.load(prefix+'gru.bias_hh.npy')[:128], dtype=tf.float32)
    #brh = tf.constant(np.load(prefix+'gru.bias_hh.npy')[128:256], dtype=tf.float32)
    bnh = tf.constant(np.load(prefix+'gru.bias_hh.npy')[256:], dtype=tf.float32)


    #input_x = tf.reshape(input_x, (-1,128))   #this is moved to reduce_sum
    input_h = tf.reshape(input_h, (-1,128))
    print(input_x.shape, wzi.shape, bz.shape, input_h.shape, wzh.shape, bz.shape)
    r=tf.sigmoid(tf.matmul(input_x, wri)+tf.matmul(input_h, wrh)+br)
    z=tf.sigmoid(tf.matmul(input_x, wzi)+tf.matmul(input_h, wzh)+bz)
    n=tf.tanh(tf.matmul(input_x, wni) + bni + r*(tf.matmul(input_h, wnh)+bnh))
    h2 = tf.multiply(1-z, n) + tf.multiply(z,input_h)
    return h2






def propagate(atom, bond_index, bond, fingerprint_dim, prefix):
    batch_size, num_atom , atom_dim = atom.shape
    batch_size, num_bond , bond_dim = bond.shape


    ####################################################################################
    neighbor_atom, target_atom, bond_index0, scatter = gather(atom, bond_index)
    print('atom.shape', atom.shape, bond_index.shape, neighbor_atom.shape, target_atom.shape)
    #return neighbor_atom


    ####################################################################################
    W_fc1 = tf.constant(np.load(prefix+'encoder.0.linear.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc1 = tf.constant(np.load(prefix+'encoder.0.linear.bias.npy'), dtype=tf.float32)
    W_bn1 = tf.constant(np.load(prefix+'bn.gamma1.npy'), dtype=tf.float32)
    b_bn1 = tf.constant(np.load(prefix+'bn.beta1.npy'), dtype=tf.float32)
    bond = tf.reshape(bond, (-1, 10))
    out_fc1 = tf.matmul(bond, W_fc1) + b_fc1
    #out_bn1 = tf.nn.fused_batch_norm(out_fc1,tf.constant(1.0),tf.constant(0.0),epsilon=1e-6)
    out_bn1 = out_fc1*W_bn1 + b_bn1
    bond = tf.reshape(tf.nn.relu(out_bn1), (-1, atom_dim, atom_dim)) 
    print('bond.shape', bond.shape, neighbor_atom.shape, out_bn1.shape)
    #return bond


    ####################################################################################
    neighbor = matmul128(neighbor_atom, bond)
    feature_align = tf.concat([target_atom, neighbor], axis=1)
    #return feature_align


    ####################################################################################
    W_fc2 = tf.constant(np.load(prefix+'align.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc2 = tf.constant(np.load(prefix+'align.bias.npy'), dtype=tf.float32)
    out_fc2 = tf.matmul(feature_align, W_fc2) + b_fc2
    #align_score = tf.nn.leaky_relu(out_fc2)
    align_score = tf.nn.relu(out_fc2) - 0.01*tf.nn.relu( out_fc2*(-1.0) )
    #return align_score


    ####################################################################################
    #W_fc3 = weight_variable([fingerprint_dim,fingerprint_dim]) 
    #b_fc3 = bias_variable([fingerprint_dim])
    #W_fc3 = tf.constant(np.ones([fingerprint_dim,fingerprint_dim]), dtype=tf.float32)
    #b_fc3 = tf.constant(np.ones([fingerprint_dim]), dtype=tf.float32)
    W_fc3 = tf.constant(np.load(prefix+'attend.linear.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc3 = tf.constant(np.load(prefix+'attend.linear.bias.npy'), dtype=tf.float32)
    W_bn3 = tf.constant(np.load(prefix+'bn.gamma2.npy'), dtype=tf.float32)
    b_bn3 = tf.constant(np.load(prefix+'bn.beta2.npy'), dtype=tf.float32)
    atend_neigh = tf.matmul(neighbor, W_fc3) + b_fc3
    atend_neigh = atend_neigh*W_bn3 + b_bn3
    #return atend_neigh
  
 
    ####################################################################################
    align_score = tf.reshape(align_score, (-1, bond_pad_num))
    segs = index2seg(bond_index0) 
    #return segs
    
    ########### scatter softmax ##############################
    align_score = align_score - scatter_max(align_score, segs, scatter) 
    align_score = tf.exp(align_score)
    attention_weight = align_score / scatter_sum(align_score, segs, scatter) 
    #return attention_weight

    
    ####################################################################################
    attention_weight = tf.tile(tf.reshape(attention_weight, (-1, 1)) , (1, atom_dim))
    attention = tf.multiply(attention_weight, atend_neigh)
    print("multipy", attention_weight.shape, atend_neigh.shape, attention.shape)
    #return attention


    ########### scatter sum ##############################
    context = scatter_sum_only(attention, segs) 
    context = tf.nn.elu(context)   

    ####################################################################################
    print("gru >>>>>>>>", context.shape, atom.shape)
    update = gru(context, atom, prefix)
    res = tf.reshape(update, (BS, atom_num, 128))
    return res



def superGather(superatom, atom, mol_index, fingerprint_dim, prefix):
    ####################################################################################
    #superatom = tf.expand_dims(superatom, (1))
    atom = tf.reshape(atom, (BS*atom_num, 128))
    superatom_expand = tf.tile(superatom, (atom_num, 1))
    feature_align = tf.concat([superatom_expand, atom], axis=-1)


    ####################################################################################
    W_fc2 = tf.constant(np.load(prefix+'align.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc2 = tf.constant(np.load(prefix+'align.bias.npy'), dtype=tf.float32)
    out_fc2 = tf.matmul(feature_align, W_fc2) + b_fc2
    #align_score = tf.nn.leaky_relu(out_fc2)
    align_score = tf.nn.relu(out_fc2) - 0.01*tf.nn.relu( out_fc2*(-1.0) )


    ########### scatter softmax ##############################
    align_score1 = tf.reshape(align_score, (BS, atom_num, 1))
    scatter_mask_zero = align_score1*mol_index - (1.0-mol_index)*99.9
    scatmax = tf.tile(tf.reduce_max(scatter_mask_zero, axis=1), (1, atom_num))

    x = tf.reshape(align_score, (BS, atom_num))
    x = x - scatmax
    x = tf.exp(x)
    xx = tf.expand_dims(x, axis=-1)
    scatsum = tf.tile(tf.reduce_sum(xx*mol_index, axis=1), (1, atom_num))
    attention_weight = x / scatsum


    ####################################################################################
    W_fc3 = tf.constant(np.load(prefix+'attend.linear.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc3 = tf.constant(np.load(prefix+'attend.linear.bias.npy'), dtype=tf.float32)
    W_bn3 = tf.constant(np.load(prefix+'bn.gamma1.npy'), dtype=tf.float32)
    b_bn3 = tf.constant(np.load(prefix+'bn.beta1.npy'), dtype=tf.float32)
    atend_neigh = tf.matmul(atom, W_fc3) + b_fc3
    atend_neigh = atend_neigh*W_bn3 + b_bn3
    context = tf.reshape(attention_weight, (-1,1))*atend_neigh


    ####################################################################################
    context = tf.reshape(context, (BS, -1, 128))
    mol_index = tf.tile(mol_index, (1,1,128))
    scatsum = tf.reduce_sum(context*mol_index, axis=1)
    context = tf.nn.elu(scatsum)
    update = gru(context, superatom, prefix)

    #return update, attention_weight
    return update

def predict_final(superatom, prefix):

    W_fc1 = tf.constant(np.load(prefix+'0.linear.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc1 = tf.constant(np.load(prefix+'0.linear.bias.npy'), dtype=tf.float32)
    W_bn1 = tf.constant(np.load(prefix+'bn.gamma1.npy'), dtype=tf.float32)
    b_bn1 = tf.constant(np.load(prefix+'bn.beta1.npy'), dtype=tf.float32)
    out_fc1 = tf.matmul(superatom, W_fc1) + b_fc1
    #out_bn1 = tf.nn.fused_batch_norm(out_fc1,tf.constant(1.0),tf.constant(0.0),epsilon=1e-6)
    out_bn1 = out_fc1*W_bn1 + b_bn1
    print('bond.shape', superatom.shape,  out_bn1.shape)
    superatom = tf.nn.relu(out_bn1) 

    W_fc2 = tf.constant(np.load(prefix+'3.weight.npy').transpose((1,0)), dtype=tf.float32)
    b_fc2 = tf.constant(np.load(prefix+'3.bias.npy'), dtype=tf.float32)
    out_fc2 = tf.matmul(superatom, W_fc2) + b_fc2
    return out_fc2



  

 
def run_network():
    #with tf.Graph().as_default():
    with tf.Session(graph=tf.Graph()) as sess:
        #data = tf.placeholder(tf.float32, shape=(BS, 32+5+1, 2, 128), name='data')
        #atom = tf.placeholder(tf.float32, shape=(BS, atom_num, 128), name='atom')
        #bond = tf.placeholder(tf.float32, shape=(BS, bond_pad_num, 10), name='bond')
        #bond_index = tf.placeholder(tf.float32, shape=(BS, bond_pad_num, 2), name='bond_index')
        atom_size = 1*atom_num*128
        bond_size = 1*bond_pad_num*10
        bidx_size = 1*bond_pad_num*2
        midx_size = 1*atom_num
        
        data = tf.placeholder(tf.float32, shape=(BS, atom_size+bond_size+bidx_size+midx_size), name='data')
        atom = tf.reshape(data[:, :atom_size], (BS, atom_num, 128), name='atom')
        bond = tf.reshape(data[:, atom_size : atom_size+bond_size], (BS, bond_pad_num, 10))
        bond_index = tf.reshape(data[:, atom_size+bond_size : atom_size+bond_size+bidx_size], (BS, bond_pad_num, 2))
        mol_index = tf.reshape(data[:, atom_size+bond_size+bidx_size : ], (BS, atom_num,1))

        mol_indexs = tf.tile(mol_index, (1,1,128))
        bond_index1 = tf.cast(bond_index, dtype=tf.int32)


        ## inference is the main function
        atom = propagate(atom, bond_index1, bond, 128, 'parms/propagate0/')
        atom = propagate(atom, bond_index1, bond, 128, 'parms/propagate1/')
        
        atom = tf.reshape(atom*mol_indexs, (BS, atom_num, 128))
        superatom = tf.reduce_sum(atom, axis=1)
        print (mol_index.shape, atom.shape)

        superatom = superGather(superatom, atom, mol_index, 128, 'parms/superGather0/')
        superatom = superGather(superatom, atom, mol_index, 128, 'parms/superGather1/')

        out1 = predict_final(superatom, 'parms/predict/')

        '''
        #index = tf.reshape(bond_index1, (8,2,128))[0][0]
        #out1 = tf.gather(atom[0], index, axis=0)
        #out1 = out1[0]
        #out1 = bond_index[0]
        bond = tf.reduce_max(bond)
        atom = tf.reduce_sum(atom)
        bond_index = tf.reduce_max(bond_index)
        '''


        print('FINISH::',  out1.name, out1.shape)

        sess.run(tf.global_variables_initializer())
        ops = sess.graph.get_operations()
        ops1 = [a.type!='Const' for a in ops]
        print(str(ops).replace('>,','>,\n'))
        print('All ops num: ', len(ops))
        print('Without const num: ', sum(ops1))


        op_name = [out1.name.split(':')[0]]
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=op_name)


        #print(constant_graph)
        with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        exit()



run_network()


