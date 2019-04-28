import json

from pprint import pprint
import csv

def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


with open('./config.json') as data_file:
    data = json.load(data_file)


keys = data[0].keys()
#print len(keys)
#for k in keys:
#    print(k)
# values = data[0].values();
# print values

# items = data[0].items()
# print items

keyframes = []

for i in range(len(data)):
	#print ' ======================= [ {} ] st ======================= '.format(i)
	#print '[{}]st totla num :: '.format(i), len(data[i]['keyframes'])
	# print '[{}]st color :: '.format(i), data[i]['color']
	# print '[{}]st type :: '.format(i),data[i]['type']
	for j in range(len(data[i]['keyframes'])):
		keyframes.append(data[i]['keyframes'][j])
		#print '[{}]st-{} keyframes | x :: {}, y :: {}, w :: {}, h :: {}, frame :: {}, truncated :: {}, continueInterpolation :: {}'.format(i, j, data[i]['keyframes'][j]['x'], data[i]['keyframes'][j]['y'], data[i]['keyframes'][j]['w'], data[i]['keyframes'][j]['h'], data[i]['keyframes'][j]['frame'], data[i]['keyframes'][j]['truncated'], data[i]['keyframes'][j]['continueInterpolation'])

write_csv('./test.csv',keyframes,',')