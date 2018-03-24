import argparse
import os
import pickle

import numpy as np

from utils.load_configutation import Configuration
from tools import visualization

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('results', help='Path to the folder where the AUC vs pixels results are stored.')
    argparser.add_argument('config', help='Path to configuration file. Used to retrieve the range of pixels.')
    args = argparser.parse_args()

    configuration = Configuration(args.config, 'auc_vs_pixels')
    cf = configuration.load()

    with open(os.path.join(args.results, 'highway_AUC_vs_pixels.pkl'), 'r') as fd:
        auc_highway = pickle.load(fd)

    with open(os.path.join(args.results, 'fall_AUC_vs_pixels.pkl'), 'r') as fd:
        auc_fall = pickle.load(fd)

    with open(os.path.join(args.results, 'traffic_AUC_vs_pixels.pkl'), 'r') as fd:
        auc_traffic = pickle.load(fd)

    # Use the pixels range specified in the configuration file
    pixels_range = np.linspace(cf.P_pixels_range[0], cf.P_pixels_range[1], num=cf.P_pixels_values)

    # Plot AUC vs pixels range
    visualization.plot_auc_vs_pixels(auc_highway, auc_traffic, auc_fall, pixels_range, args.results)
