"""
import os
import sys
"""

import os
import caffe
import numpy as np
from numpy import prod, sum
from pprint import pprint # pretty print
import argparse


def print_net_parameters (deploy_file):

    print "Net: " + deploy_file
    net = caffe.Net(deploy_file, caffe.TEST)
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print "Layer-wise parameters: "
    # print net.params['first_conv1'][1].data.shape
    # pprint([(k, v[0].data.shape) for k, v in net.params.items()])
    print " === Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()]))

    # for each layer, show the output shape
    print " === for each layer, show the output shape === "
    with open('{}/deploy_output_shape.txt'.format(output_dir), 'w') as f:
        for layer_name, blob in net.blobs.iteritems():
            info_str = layer_name + '\t' + str(blob.data.shape)
            f.write(info_str + '\n')
            print info_str

'''
    print " === for each layer, show the layer parameters === "
    for layer_name, param in net.params.iteritems():
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
'''

if __name__ == '__main__':

    caffe_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    parser = argparse.ArgumentParser()
    parser.add_argument('--network-path', default='', help='Path to the network prototxt')

    args = parser.parse_args()

    # model_def = args.network_path or '{}/models/WRN_Basic/KITTI/SSD_300x300_0/deploy.prototxt'.format(caffe_root)
    # model_def = args.network_path or '{}/work/mscoco/resnet50.prototxt'.format(caffe_root)
    # model_def = args.network_path or '{}/models/WRN_Inception/KITTI/SSD_300x300_0/deploy.prototxt'.format(caffe_root)
    # model_def = args.network_path or '{}/models/WRN_Inception_l2/KITTI/SSD_300x300_0/deploy.prototxt'.format(caffe_root)
    model_def = args.network_path or '{}/models/VGGNet/KITTI/SSD_300x300_0/deploy.prototxt'.format(caffe_root)
    print_net_parameters(model_def)