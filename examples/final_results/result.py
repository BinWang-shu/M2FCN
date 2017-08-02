# -*- coding: utf-8 -*-
import numpy as np
import h5py
import scipy.misc
import Image
import scipy.io
import scipy
import os
from scipy import io
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
test_dir = '/media/data1/JY/neuron/ISBI/test/imgs/'
save_dir_root = '/media/data1/JY/neuron/ISBI/results/edges/hed_new_R/'
net = caffe.Net('./deploy_R.prototxt','./hed_R_iter_26000.caffemodel', caffe.TEST)

# save_dir_1 = save_dir_root + '1/';
# save_dir_2 = save_dir_root + '2/';
# save_dir_3 = save_dir_root + '3/';
# save_dir_4 = save_dir_root + '4/';
# save_dir_5 = save_dir_root + '5/';
save_dir_fuse = save_dir_root + 'fuse/';
save_dir_R = save_dir_root + 'R/';
#save_dir_R2 = save_dir_root + 'R2/';
save_dir_F = save_dir_root + 'F/';

neg_label = 0

# if not os.path.exists(save_dir_1):
#     os.makedirs(save_dir_1)
# if not os.path.exists(save_dir_2):
#     os.makedirs(save_dir_2)
# if not os.path.exists(save_dir_3):
#     os.makedirs(save_dir_3)
# if not os.path.exists(save_dir_4):
#     os.makedirs(save_dir_4)
# if not os.path.exists(save_dir_5):
#     os.makedirs(save_dir_5)
if not os.path.exists(save_dir_fuse):
    os.makedirs(save_dir_fuse)
if not os.path.exists(save_dir_R):
    os.makedirs(save_dir_R)
#if not os.path.exists(save_dir_R2):
#    os.makedirs(save_dir_R2)
if not os.path.exists(save_dir_F):
    os.makedirs(save_dir_F)

imgs = [im for im in os.listdir(test_dir) if '.png' in im]
print imgs
nimgs = len(imgs)
print "totally "+str(nimgs)+"images"
for i in range(nimgs):
    img = imgs[i]
    print img
    img = Image.open(test_dir + img) 
    # img = h5py.File(test_dir + img)
    # img = io.loadmat(test_dir + img)
    # img = img['result'];
    img = np.array(img, dtype=np.float32)
    #if img.shape[0] > 400:
    #    r = np.float(400) / img.shape[0]
    #    print r
    #    img = scipy.misc.imresize(img, r)
    #    img = np.array(img, dtype=np.float32)
    #    print img.shape, '------------------'
    img = img[:,:,::-1]
    img -= np.array((104.00698793,116.66876762,122.67891434))
    img = img.transpose((2,0,1))
    # img = img.transpose((0,2,1))
    net.blobs['data'].reshape(1, *img.shape)
    net.blobs['data'].data[...] = img
    net.forward()

    # out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    # out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    # out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    # out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    # out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    R = net.blobs['sigmoid-fuse_R'].data[0][0,:,:]
    #R2 = net.blobs['sigmoid-fuse_R2'].data[0][0,:,:]
    F = net.blobs['sigmoid-fuse_F'].data[0][0,:,:]

    # scipy.misc.imsave(save_dir_1 + imgs[i][0:-4]+".png",out1/out1.max())
    # scipy.misc.imsave(save_dir_2 + imgs[i][0:-4]+".png",out2/out2.max())
    # scipy.misc.imsave(save_dir_3 + imgs[i][0:-4]+".png",out3/out3.max())
    # scipy.misc.imsave(save_dir_4 + imgs[i][0:-4]+".png",out4/out4.max())
    # scipy.misc.imsave(save_dir_5 + imgs[i][0:-4]+".png",out5/out5.max())
    scipy.misc.imsave(save_dir_fuse + imgs[i][0:-4]+".png",fuse/fuse.max())
    scipy.misc.imsave(save_dir_R + imgs[i][0:-4]+".png",R/R.max())
    #scipy.misc.imsave(save_dir_R2 + imgs[i][0:-4]+".png",R2/R2.max())
    scipy.misc.imsave(save_dir_F + imgs[i][0:-4]+".png",F/F.max())

    print imgs[i]+"   ("+str(i+1)+" of "+str(nimgs)+")    saved"
    
    #xx = dict()
    #conv_blobs = [b for b in net.blobs.keys() if 'conv' in b]
    #for b in conv_blobs:
    #    xx[b] = net.blobs[b].data[0][...]
    #scipy.io.savemat(save_dir_fuse + imgs[i][0:-4],xx,appendmat=True)
