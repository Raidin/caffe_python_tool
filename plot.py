#!/usr/bin/env python
import argparse
import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import pandas as pd

caffe_root = os.path.join(os.environ['HOME'] , 'caffe')
plot_path = os.path.join(caffe_root, 'plot')
log_list_dir = os.path.join(plot_path,'log_list')
parse_log_dir = os.path.join(plot_path, 'parse_log')

if not os.path.exists(plot_path):
    os.mkdir(plot_path)
if not os.path.exists(log_list_dir):
    os.mkdir(log_list_dir)
if not os.path.exists(parse_log_dir):
    os.mkdir(parse_log_dir)

class Field:
    class x:
        iters = u"Iteration"
        Seconds = u"Seconds"
    class y:
        TrainingLoss = u"TrainingLoss"
        LearningRate = u"LearningRate"
        TestAccuracy = u"TestAccuracy"
        TestLoss = u"TestLoss"

def get_log_parsing_script(extension='.sh'):
    dirname = os.path.join(caffe_root, 'tools/extra')
    return dirname + '/parse_log' + extension

def csv_reader(filename):
    #table = pd.read_csv(filename, sep=' +', index_col=0, engine='python')
    table = pd.read_csv(filename, index_col=0)
    # print table.columns
    # print table.head(8)
    # print table.info()
    # print table.describe()
    return table

def parsing_log(path_to_log_list, log_type='train'):
    log_output_list = []

    for path_to_log in path_to_log_list:
        tmp_file = os.path.basename(path_to_log + '.' + log_type)
        new_file = os.path.join('{}/parse_log'.format(plot_path), tmp_file)
        os.system('%s %s %s' % (get_log_parsing_script('.py'), path_to_log, './'))
        os.rename(tmp_file, new_file)
        log_output_list.append(new_file)

    # return output_log_list
    return log_output_list

def plot_train_chart(train_list, show_plot=False, job_name=''):
    plt.cla()

    title = job_name + '_{}'.format(Field.y.TrainingLoss)
    list_count = 0

    for log in train_list:
        list_count += 1
        label_name = os.path.basename(log).split('.')[0]

        table = csv_reader(log)
        # table[table>plot_xlim]=plot_xlim
        table.loss.plot(label=label_name+' loss', legend=True)
        table.accuracy_top1.plot(secondary_y=True, label=label_name+' accuracy', legend=True)
        #os.remove(log)

    plt.title(title)
    plt.xlabel(Field.x.iters)
    # plt.ylabel(Field.y.TrainingLoss)
    plt.savefig(os.path.join(plot_path, title+'.png'))

    if show_plot:
        plt.show()
    plt.gcf().clear()

def plot_test_chart(test_list, show_plot=False, job_name=''):

    title = job_name + '_{}'.format(Field.y.TestAccuracy)
    list_count = 0

    for log in test_list:
        list_count += 1
        label_name = os.path.basename(log).split('.')[0]

        try:
            table = csv_reader(log)
            #table[table>plot_xlim]=plot_xlim
            #table.loss.plot(legend=True, label=label_name+' te')
            table.accuracy_top1.plot(lw=1, legend=True, label=label_name+'-top1')
        except:
            # no accuracy
            pass
        #os.remove(log)

    plt.title(title)
    plt.xlabel(Field.x.iters)
    plt.ylabel(Field.y.TestAccuracy)
    plt.savefig(os.path.join(plot_path, title+'.png'))
    if show_plot:
        plt.show()
    plt.gcf().clear()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('job_name', help='Input Current Job Name (ex, googLeNet, etc)')
    parser.add_argument('--train-parsing', default=True, action='store_true', help='Run train parsing')
    parser.add_argument('--test-parsing', default=False, action='store_true', help='Run test parsing')
    parser.add_argument('--show-plot', default=False, action='store_true', help='Display plot')

    args = parser.parse_args()
    filenames = sorted(os.listdir(log_list_dir))

    job_name = args.job_name

    path_to_logs = []
    for filename in filenames:
        path_to_logs.append(os.path.join(log_list_dir, filename))

    for path_to_log in path_to_logs:
        if not os.path.exists(path_to_log):
            print 'Path does not exist: %s' % path_to_log
            sys.exit()
        if not path_to_log.endswith('.log'):
            print 'Log file must end in .log.'

    ## plot_chart accpets multiple path_to_logs
    if args.train_parsing:
        train_list = parsing_log(path_to_log_list=path_to_logs, log_type='train')
        plot_train_chart(train_list=train_list, show_plot=args.show_plot, job_name=job_name)

    if args.test_parsing:
        test_list =  parsing_log(path_to_log_list=path_to_logs, log_type='test')
        plot_test_chart(test_list=test_list, show_plot=args.show_plot, job_name=job_name)