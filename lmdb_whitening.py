import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2

def main(config):
	lmdb_path = config['lmdb-path']
	lmdb_new_path = config['new-lmdb-path']
	lmdb_map_size = 1e10

	lmdb_env_read = lmdb.open(lmdb_path)
	lmdb_txn_read = lmdb_env_read.begin()
	lmdb_cursor_read = lmdb_txn_read.cursor()
	datum_read = caffe_pb2.Datum()

	lmdb_env_write = lmdb.open(lmdb_new_path, map_size=int(lmdb_map_size))
	lmdb_txn_write = lmdb_env_write.begin(write=True)
	datum_write = caffe_pb2.Datum()
	batch_size = 1000
	item_num = -1

	for key, value in lmdb_cursor_read:
		item_num = item_num + 1
		datum_read.ParseFromString(value)

		label = datum_read.label

		if datum_read.encoded == True:
			arr = np.frombuffer(datum_read.data, dtype='uint8')
			data = cv2.imdecode(arr, cv2.IMREAD_COLOR) # Shape : H x W x C
		else:
			data = caffe.io.datum_to_array(datum_read) # Shape : C x H x W
			data = np.transpose(data, (1,2,0))

		a = np.mean(data)
		print type(a)

		display_img = data = (data - np.mean(data)) / np.std(data)
		data = np.transpose(data, (2,0,1))

		# save in datum
		datum_write = caffe.io.array_to_datum(data, label)
		keystr = '{:0>8d}'.format(item_num)
		lmdb_txn_write.put(keystr, datum_write.SerializeToString())

		if config['image-display']:
			cv2.imshow('cv2', display_img)
			print('- org lmdb :: {},{}'.format(key, label))
			if cv2.waitKey(0) == 27: # esc key --> exit this program...
				break

		# write batch
		if(item_num + 1) % batch_size == 0:
			lmdb_txn_write.commit()
			lmdb_txn_write = lmdb_env_write.begin(write=True)
			print ' - Apply Whitening item :: {}'.format(item_num + 1)

	# write last batch
	if (item_num + 1) % batch_size != 0:
		lmdb_txn_write.commit()
		print ' - Completed applying Whitening item :: {}'.format(item_num + 1)

if __name__ == '__main__':
	# lmdb_path = '/home/jihunjung/gpu_server_8gpu_34_jihunjung/data/cifar10/lmdb/cifar10_train_lmdb'
	lmdb_path = '/home/jihunjung/data/cifar10/lmdb/cifar10_test_lmdb'
	new_lmdb_path = './cifar10_test_lmdb_whitening'

	config = {
		'lmdb-path' : lmdb_path,
		'new-lmdb-path' : new_lmdb_path,
		'image-display' : True,
	}

	main(config)