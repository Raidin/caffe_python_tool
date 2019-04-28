import sys
import os
import subprocess
import stat

caffe_root = os.environ['HOME'] + '/caffe_ssd'

def create_mean_binary():

	lmdb_dir = '/home/jihunjung/data/cifar10/lmdb/cifar10_train_lmdb' # /path/to/lmdb
	# lmdb_dir = '/home/jihunjung/data/cifar10/lmdb/cifar10_train_lmdb'
	create_mean_binary_job_file = './create_mean_binary_job_file.sh'
	output_mean_binary = './train_mean.binaryproto'

	with open(create_mean_binary_job_file, 'w') as f:
		f.write('TOOLS={} \n'.format('{}/build/tools'.format(caffe_root)))
		f.write('GLOG_logtostderr=1  $TOOLS/compute_image_mean \\\n')
		f.write('{} \\\n'.format(lmdb_dir))
		f.write('{} '.format(output_mean_binary))

	os.chmod(create_mean_binary_job_file, stat.S_IRWXU)
	subprocess.call(create_mean_binary_job_file, shell=True)

	os.remove(create_mean_binary_job_file)

if __name__ == '__main__':
	create_mean_binary()