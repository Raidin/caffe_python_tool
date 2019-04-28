caffe_root = '/home/jihunjung/caffe'
image_dir = caffe_root + "working/The Oxford-IIIT Pet Dataset/"
import sys
sys.path.insert(0, caffe_root + 'python')
# MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
# PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

MEAN_FILE = './mean.npy'
MODEL_FILE = caffe_root + '/models/deploy.prototxt'
PRETRAINED = caffe_root + '/models/iter_200000.caffemodel'
FEAT_LAYER = 'fc6wi'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', metavar='Inputs', type=str, default='filenames.npy',
                   help='npy filename containing image filenames')
parser.add_argument('-o', metavar='Outputs', type=str, default='features.npy',
                    help='npy filename wirtes extracted features in')
args = parser.parse_args()

import caffe
import numpy as np
import cv2
import matplotlib.pyplot as plt

def vis_square(data, figure_title):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure(figure_title)
    plt.imshow(data); plt.axis('off')
    # plt.savefig('%s.png'%(figure_title))

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

features = []
# IMAGE_FILES = np.load(args.i)
# N = len(IMAGE_FILES)
net.blobs['data'].reshape(1,3,150,360)
# LOAD_IMAGE = '/home/jihunjung//python/Straight.png'
LOAD_IMAGE = '/home/jihunjung/python/curve.png'

image = caffe.io.load_image(LOAD_IMAGE)
transformed_image = transformer.preprocess('data', image)
# cv2.imshow('image', image)
# key = cv2.waitKey(0)

net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = output['prob'][0]

# weight filter
# filters = net.params['conv_1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))

# layer output
feat = net.blobs['conv_1'].data[0, :16]
vis_square(feat, 'conv_1')
feat = net.blobs['conv_2'].data[0, :16]
vis_square(feat, 'conv_2')
feat = net.blobs['conv_3'].data[0, :16]
vis_square(feat, 'conv_3')
feat = net.blobs['conv_4'].data[0, :16]
vis_square(feat, 'conv_4')


# feat = net.blobs['fc5'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=100)

# feat = net.blobs['fc6'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=100)

# feat = net.blobs['fc7'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=400)

feat = net.blobs['prob'].data[0]
plt.figure('prob',figsize=(15, 3))
plt.plot(feat.flat)
plt.savefig('prob.png')
plt.show()
# print output_prob

# net.blobs['data'].data = transformer.preprocess('data', caffe.io.load_image(LOAD_IMAGE))
# for i in range(N):
#     LOAD_IMAGE = image_dir + IMAGE_FILES[i]
#     net.blobs['data'].data[i] = \
#         transformer.preprocess('data', caffe.io.load_image(LOAD_IMAGE))
# net.forward()
# features = net.blobs[FEAT_LAYER].data
# np.save(args.o, features)