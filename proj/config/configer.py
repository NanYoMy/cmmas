import configparser
import os
import sys


class RegParser:
    def __init__(self, argv='', config_type='all'):
        filename_=None
        nargs_ = len(argv)
        if nargs_ == 3 or nargs_==4:
            if (argv[1] == '-h') or (argv[1] == '-help'):
                self.print_help()
                exit()
            filename_ = argv[1]
        else:
            # filename_ = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config_demo.ini"))
            # print('Reading default config dirutil in: %s.' % filename_)
            # exit(-9999)
            pass

        self.config_file = configparser.ConfigParser()
        if filename_ is not None:
            self.config_file.read(filename_)
        else:
            print('Using defaults due to missing config dirutil.')

        self.config_type = config_type.lower()
        self.config = self.get_defaults()
        self.check_defaults()
        self.print()

    def check_defaults(self):

        for section_key in self.config.keys():
            if section_key in self.config_file:
                for key, value in self.config[section_key].items():
                    if key in self.config_file[section_key] and self.config_file[section_key][key]:
                        if type(value) == str:
                            self.config[section_key][key] = os.path.expanduser(self.config_file[section_key][key])
                        else:
                            self.config[section_key][key] = eval(self.config_file[section_key][key])
                    # else:
                        # print('Default set in [''%s'']: %s = %s' % (section_key, key, value))
            # else:
                # print('Default section set: [''%s'']' % section_key)

    def __getitem__(self, key):
        return self.config[key]

    def print(self):
        print('')
        for section_key, section_value in self.config.items():
                for key, value in section_value.items():
                    print('[''%s'']: %s: %s' % (section_key, key, value))
        print('')

    def get_defaults(self):

        home_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        network = {'network_type': 'local'}

        data = {'dir_moving_image': os.path.join(home_dir, 'data/train/mr_images'),
                'dir_fixed_image': os.path.join(home_dir, 'data/train/us_images'),
                'dir_moving_label': os.path.join(home_dir, 'data/train/mr_labels'),
                'dir_fixed_label': os.path.join(home_dir, 'data/train/us_labels'),
                'ori_train_img':"error",
                'ori_train_lab':'error',
                'structure':'heart',
                'tag':'error',
                'ori_test_img':'error',
                'ori_test_lab':'error',
                'mannual_train_crop_img':"error",
                'mannual_train_crop_lab':'error',
                'mannual_test_crop_img':'error'}

        loss = {'similarity_type': 'dice',
                'similarity_scales': [0, 1, 2, 4, 8, 16],
                'regulariser_type': 'bending',
                'regulariser_weight': 0.5,
                'consistent_weight':0.01}

        train = {'total_iterations': int(1e5),
                 'learning_rate': 1e-5,
                 'minibatch_size': 2,
                 'freq_info_print': 100,
                 'freq_model_save': 500,
                 'file_model_save': os.path.join(home_dir, 'data/model.ckpt')}

        inference = {'file_model_saved': train['file_model_save'],
                     'dir_moving_image': os.path.join(home_dir, 'data/test/mr_images'),
                     'dir_fixed_image': os.path.join(home_dir, 'data/test/us_images'),
                     'dir_save': os.path.join(home_dir, 'data/'),
                     'dir_moving_label': '',
                     'dir_fixed_label': '',
                     'ori_test_img':'error',
                     'ori_test_lab':'error'}

        if self.config_type == 'training':
            config = {'Data': data, 'Network': network, 'Loss': loss, 'Train': train}
        elif self.config_type == 'inference':
            config = {'Network': network, 'Inference': inference}
        else:
            config = {'Data': data, 'Network': network, 'Loss': loss, 'Train': train, 'Inference': inference}


        return config

    @staticmethod
    def print_help():
        print('\n'.join([
            '',
            '************************************************************',
            '  Weakly-Supervised CNNs for Multimodal Image Registration',
            '      2018 Yipeng Hu <yipeng.hu@ucl.ac.uk> ',
            '  LabelReg package is licensed under: ',
            '      http://www.apache.org/licenses/LICENSE-2.0',
            '************************************************************',
            '',
            'Training script:',
            '   python3 training_20.py myConfig.ini',
            '',
            'Inference script:',
            '   python3 _inference_test_20.py myConfig.ini',
            '',
            'Options in config dirutil myConfig.ini:',
            '   network_type:       {local, global, composite}',
            '   similarity_type:    {dice, cross-entropy, mean-squared, jaccard}',
            '   regulariser_type:   {bending, gradient-l2, gradient-l1}',
            'See other parameters in the template config dirutil config_demo.ini.',
            ''
        ]))

# reg_config = RegParser(sys.argv, 'all')

class VoteNetParser(RegParser):

    def __init__(self, argv='', config_type='all'):
        filename_=None
        nargs_ = len(argv)
        if nargs_ == 3 or nargs_==4:
            if (argv[1] == '-h') or (argv[1] == '-help'):
                self.print_help()
                exit()
            filename_ = argv[2]
        else:
            # filename_ = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config_demo.ini"))
            # print('Reading default config dirutil in: %s.' % filename_)
            # exit(-9999)
            print("")

        self.config_file = configparser.ConfigParser()
        if filename_ is not None:
            self.config_file.read(filename_)
        else:
            print('Using defaults due to missing config dirutil.')

        self.config_type = config_type.lower()
        self.config = self.get_defaults()
        self.check_defaults()
        self.print()
    def get_defaults(self):

        home_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        network = {'network_type': 'local'}

        generator={
            'output_train_dir':"error",
            'output_test_dir':"error",
            'reg_model': "error"
        }

        data = {'dir_atlas_image': "error",
                'dir_target_image': "error",
                'dir_atlas_label': "error",
                'dir_target_label': "error",
                'structure':'myo',
                'tag':'error'}

        loss = {'similarity_type': 'dice',
                'similarity_scales': [0, 1, 2, 4, 8, 16],
                'regulariser_weight': 0.5
                }

        train = {'total_iterations': int(1e5),
                 'learning_rate': 1e-5,
                 'minibatch_size': 2,
                 'freq_info_print': 100,
                 'freq_model_save': 500,
                 'file_model_save': os.path.join(home_dir, 'data/model.ckpt')}

        inference = {'file_model_saved': train['file_model_save'],
                     'dir_atlas_image': "",
                     'dir_target_image': "",
                     'dir_save': "",
                     'dir_atlas_label': '',
                     'dir_target_label': '',
                     'fusion_out':'',
                     'forward_process_start_dir':''}

        if self.config_type == 'training':
            config = {'Data': data, 'Network': network, 'Loss': loss, 'Train': train}
        elif self.config_type == 'inference':
            config = {'Network': network, 'Inference': inference}
        elif self.config_type=='generator':
            config = {'Generator':generator}
        else:
            config = {'Data': data, 'Network': network, 'Loss': loss,'Generator':generator, 'Train': train, 'Inference': inference}

        return config

# vote_config = VoteNetParser(sys.argv, 'all')

# if vote_config["Generator"]['reg_model']!= reg_config['Train']['file_model_save']:
#     print("error!!!!! the model didn't match")
#     exit(-998)

def get_reg_config():
    global reg_config
    if reg_config==None:
        reg_config=RegParser(sys.argv, 'all')
    return reg_config

def get_vote_config():
    global vote_config
    if vote_config==None:
        vote_config=VoteNetParser(sys.argv, 'all')
    return vote_config