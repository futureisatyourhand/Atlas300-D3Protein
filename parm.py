#coding=utf-8
import os
import caffe
import csv
import numpy as np
# np.set_printoptions(threshold='nan')

##to get all npy

MODEL_FILE = 'IRModel16.prototxt'
PRETRAIN_FILE = 'bs16_iter_1.caffemodel'

net = caffe.Net(MODEL_FILE,caffe.TEST, weights=PRETRAIN_FILE)
root="/home/liqian/chems/ContactPred/IRConPred2/parm/"
p = []

##init to parmeter according ir.py
parm_dict={
    'InceptionResNetBlock1d':'InceptionResNetBlock1d',
    'final_block':'final_block'
 }
conv_dict={
    0:'branch3x3',
    1:'branch5x5',
    2:'branch3x3_dilated',
    3:'branch5x5_dilated'
}
keys=net.params.keys()
count=1
print(net.params.keys())
for param_name in net.params.keys():
    if 'norm' in param_name or 'classification':
        continue
    names=param_name.split('_')
    if count%13==0:
        weights_npy="ir.InceptionResNetBlock_"+str(id)+".reduction1x1.weight.npy"
        bias_npy="ir.InceptionResNetBlock_"+str(id)+".reduction1x1.bias.npy"
    else:
        if 'block' in names[0]:
            id=int(names[0].split('block')[-1])
        else:
            id=int(names[0])
        id/=4
        convs=id%4
        if 'conv' in param_name:
            weight_npy="ir.InceptionResNetBlock_"+str(id)+"."+conv_dict[convs]+".conv.weight.npy"
            bias_npy="ir.InceptionResNetBlock_"+str(id)+"."+conv_dict[convs]+".conv.bias.npy"
        elif 'scale' in param_name:
             weight_npy="ir.InceptionResNetBlock_"+str(id)+"."+conv_dict[convs]+".bn.weight.npy"
             bias_npy="ir.InceptionResNetBlock_"+str(id)+"."+conv_dict[convs]+".bn.bias.npy"
        else:
            weight_npy=bias_npy=""
    if weight_npy!="":
         net.params[param_name][0].data[:]=np.load(root+weight_file)         
    if bias_npy!="":
        net.params[param_name][1].data[:]=np.load(root+bias_file)
    count+=1
net.save('IRModel_new_parameter.caffemodel')
