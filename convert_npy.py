import caffe
import numpy as np
import sys

## proto / datum / ndarray conversion
def blobproto_to_array(blob, return_diff = False):
	"""
	Convert a blob proto to an array, In default, we will just return the data,
	unless return_diff is True, in which case we will return the diff.
	"""
	# Read the data into an array
	if return_diff:
		data = np.array(blob.diff)
	else:
		data = np.array(blob.data)

	# Reshape the array
	if blob.HasField('num') or blob.HasField('channels') or blob.HasField('height') or blob.HasField('width'):
		# Use legacy 4D shape
		return data.reshape(blob.channels, blob.height, blob.width)
	else:
		return data.reshape(blob.shape.dim)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('/home/jihunjung/models/train_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = blobproto_to_array(blob)
np.save('./mean.npy', arr)