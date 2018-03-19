import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from config.load_configutation import Configuration
from tools import visualization

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--results', help='Path to the folder where the optimization results are stored.')
    argparser.add_argument('--config', help='Path to configuration file. Used to retrieve the range of pixels.')
    args = argparser.parse_args()

    configuration = Configuration(args.config, 'optical_flow_opt')
    cf = configuration.load()

    with open(os.path.join(args.results, 'kitti_ebma_optimization.pkl'), 'r') as fd:
        optimization_results = pickle.load(fd)

    c = ['b', 'g', 'r', 'y', 'm']

    block_size = []
    area_search = []
    msen = []
    pepn = []
    execution_time = []
    for key in optimization_results:
        result = optimization_results[key]
        block_size.append(result['block_size'])
        area_search.append(result['area_search'])
        msen.append(result['msen'])
        pepn.append(result['pepn'])
        execution_time.append(result['execution_time'])

    size_4 = np.ma.masked_not_equal(block_size, 4)*16
    size_8 = np.ma.masked_not_equal(block_size, 8)*8
    size_16 = np.ma.masked_not_equal(block_size, 16)*4
    size_32 = np.ma.masked_not_equal(block_size, 32)*2
    size_64 = np.ma.masked_not_equal(block_size, 64)

    plt.figure(figsize=(50, 10))
    plt.subplot(121)
    plt.scatter(execution_time, msen, s=size_4, marker='o', c=c)
    plt.scatter(execution_time, msen, s=size_8, marker='s', c=c)
    plt.scatter(execution_time, msen, s=size_16, marker='^', c=c)
    plt.scatter(execution_time, msen, s=size_32, marker='p', c=c)
    plt.scatter(execution_time, msen, s=size_64, marker='*', c=c)
    plt.xlabel('Execution time')
    plt.ylabel('MSEN')

    plt.subplot(122)
    plt.scatter(execution_time, pepn, s=size_4, marker='o', c=c)
    plt.scatter(execution_time, pepn, s=size_8, marker='s', c=c)
    plt.scatter(execution_time, pepn, s=size_16, marker='^', c=c)
    plt.scatter(execution_time, pepn, s=size_32, marker='p', c=c)
    plt.scatter(execution_time, pepn, s=size_64, marker='*', c=c)
    plt.xlabel('Execution time')
    plt.ylabel('PEPN')
    plt.show(block=False)
    plt.savefig(os.path.join(args.results, "week4_optical_optimization.png"))
    plt.close()


    print('End')
