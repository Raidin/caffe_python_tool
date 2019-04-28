import os
import caffe
from caffe.draw import *
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def drawNetwork(net, jobname="", filename=""):
	caffe_root= os.environ['HOME'] + "/caffe_ssd"
	# Drawing Network
	save_dir = '{}/models/{}/draw_network'.format(caffe_root, jobname)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	draw_network_store_path = '{}/{}'.format(save_dir,filename)
	draw_net_to_file(net.to_proto(), draw_network_store_path)

if __name__ == '__main__':
	net = caffe.NetSpec()
	drawNetwork(net, '[jobname]', '[path/to/file]')