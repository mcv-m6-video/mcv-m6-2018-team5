from __future__ import print_function

import imp
import os


class Configuration(object):
    def __init__(self, config_path, test_name):

        self.config_path = config_path
        self.test_name = test_name
        self.configuration = None

    def load(self):

        # Get Config path
        print(self.config_path)
        config_path = os.path.join(os.getcwd(), os.path.dirname(self.config_path),
                                   os.path.basename(self.config_path))
        print('Config file loaded: ')
        print(config_path)

        cf = imp.load_source('config', config_path)

        # Save extra parameter
        cf.config_path = config_path
        cf.test_name = self.test_name

        cf.dataset_path = os.path.abspath(cf.dataset_path)
        cf.gt_path = os.path.abspath(cf.gt_path)

        cf.results_path = os.path.abspath(cf.results_path)
        if not os.path.exists(cf.results_path):
            os.makedirs(cf.results_path)

        if cf.save_results:

            # Output folder
            cf.output_folder = os.path.join(os.path.abspath(cf.output_folder), cf.test_name)

            if not os.path.exists(cf.output_folder):
                os.makedirs(cf.output_folder)

            print ('Save results in: ' + cf.output_folder)
            cf.log_file = os.path.join(cf.output_folder, "logfile.log")

        self.configuration = cf
        return cf
