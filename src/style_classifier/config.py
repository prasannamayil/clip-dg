'''
Couple of things to note: base_directory, data_root, beta
'''

# config.py imports
import os
import argparse
from datetime import datetime
import time as tm
import torchvision.transforms as transforms
import pickle
import torch
import wandb
import yaml
import numpy as np
import random
from .utils import *


# Some util funcs
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args_train():
    parser = argparse.ArgumentParser()
    # Cluster configuration

    # Dataset Configuration
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', dest='dataset', nargs='+',
                        type=str, help='List of values separated by space')
    parser.add_argument("--data_root", type=str,
                        default="",
                        help="[E] root location of dataset")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--train_transform', default=None, help='used later in the routine, do not specify here')
    parser.add_argument('--test_transform', default=None, help='used later in the routine, do not specify here')
    parser.add_argument('--image_folder', type=boolean_string, default=False)

    # Model Configuration
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrained', type=boolean_string, default=False)
    parser.add_argument('--freeze_encoder', type=boolean_string,
                        default=False, help='train only linear readout for clip models')

    # Training Configuration
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', default='step', choices=['step', 'cosine'])
    parser.add_argument('--step_size', type=int, default=12)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--number_of_gpus', type=int, default=1)
    parser.add_argument('--test_num_workers', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=0.9)

    # Training Control
    parser.add_argument('--resume', '-r', type=boolean_string, default=False, help='resume from checkpoint')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--base_directory', type=str,
                        default="",
                        help="directory to save everything")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_split_size', type=float, default=0.75, help="split size of train")
    parser.add_argument('--val_split_size', type=float, default=0.125, help="split size of validation")
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='sample size to collect stats from test and val datasets, if -1 then whole set')
    parser.add_argument('--save', type=bool, default=True, help='marker for saving')
    parser.add_argument('--patience_es', type=int, default=5, help="Early stopping patience epochs")
    parser.add_argument('--save_after_epoch', type=int, default=50, help="save checkpoints after n epochs")
    parser.add_argument('--early_stopping',  type=boolean_string, default=False, help='to activate early stopping')

    # Fourier-related Configuration
    parser.add_argument("--fourier",  type=boolean_string, default=False, help='to train only on the amplitude spectrum')
    parser.add_argument("--shift_images",  type=boolean_string, default=False, help='To invert FFT shift images')
    parser.add_argument("--shift_spectrum", type=boolean_string, default=False, help='to FFT shift spectrum')

    args = parser.parse_args()

    # number of gpus
    args.number_of_gpus = torch.cuda.device_count()

    # open configs and paths
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_config_folder = "../config"

    label_mapping_name = "label_mapping.yaml"
    label_data_path = os.path.join(current_script_dir, relative_path_to_config_folder, label_mapping_name)
    with open(label_data_path, "r") as ymlfile:
        label_mapping = yaml.safe_load(ymlfile)

    relative_path_to_paths_folder = "../paths"
    paths_file_name = "paths.yaml"
    paths_file_path = os.path.join(current_script_dir, relative_path_to_paths_folder, paths_file_name)
    with open(paths_file_path, "r") as ymlfile:
        paths = yaml.safe_load(ymlfile)

    # Get label mapping and folder locs and fixing some args
    if not args.image_folder:
        args.data_root = [paths['datasets'][dataset] for dataset in args.dataset]
        args.label_mapping = label_mapping['label_mapping']
        args.label_mapping_dirs = {}
        for dataset in args.dataset:
            args.label_mapping_dirs[paths['datasets'][dataset]] = args.label_mapping[dataset]
    else:
        args.dataset = args.dataset[0]
        args.data_root = paths['datasets'][args.dataset]
        args.num_classes = len(os.listdir(args.data_root))   # change num_classes from 2 to whatever, usually 3
        args.label_mapping = label_mapping['label_mapping']  # this is useless
        args.label_mapping_dirs = {}  # this is useless too

    # Create new directory (putting this in a loop because multiple jobs sometime start and dir names are same)
    while True:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")

        args.name = args.dataset if args.name is None else args.name
        args.directory = args.base_directory + args.name + '_' + args.arch + '_' + \
                         str(args.freeze_encoder) + '_' + date_time + "/"
        args.directory_net = args.directory + "checkpoints/"
        try:
            if (not os.path.exists(args.directory_net)):
                os.makedirs(args.directory_net)
            break
        except:
            rn = random.uniform(2, 100)  # 2 to 100 seconds
            tm.sleep(rn)
            continue

    # Save text file about experiment's essential data
    lines = ["Experiment: Style classifier", "Name: " + args.name, "Datetime: " + date_time,
             "dataset: " + ', '.join(map(str, args.dataset)),
             "Model: " + str(args.arch), "lr: " + str(args.lr),
             "bs: " + str(args.batch_size),
             "Seed: " + str(args.seed), "Train split size: " + str(args.train_split_size),
             "Val split size: " + str(args.val_split_size), "Epochs: " + str(args.epochs)
             ]
    # print args
    print(args)
    # Save summary
    with open(args.directory+'summary.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    # Save args
    args_dict = vars(args)
    with open(args.directory + 'args.pickle', 'wb') as handle:
        pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return args


def get_args_eval():
    '''
    Job no. is the most important argument through which we get everything else
    '''

    # Basic eval args
    parser = argparse.ArgumentParser()

    # input args
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--directory", type=str,
                        default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--models_eval', type=str)
    parser.add_argument('--eval_datasets', dest='eval_datasets', nargs='+',
                        type=str, help='List of values separated by space')

    # other args
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--save_preds_number', type=int, default=1000000)
    parser.add_argument('--test_transform', default=None, help='used later in the routine, do not specify here')

    # Create args and add some extra things
    ap = parser.parse_args()
    args = vars(ap)

    # Get training args
    with open(args['directory'] + '/args.pickle', 'rb') as handle:
        training_args = pickle.load(handle)

    # number of gpus
    args['number_of_gpus'] = torch.cuda.device_count()

    # open paths
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    relative_path_to_paths_folder = "../paths"
    paths_file_name = "paths.yaml"
    paths_file_path = os.path.join(current_script_dir, relative_path_to_paths_folder, paths_file_name)
    with open(paths_file_path, "r") as ymlfile:
        paths = yaml.safe_load(ymlfile)

    # open configs and paths
    relative_path_to_config_folder = "../config"

    # label mapping for datasets (in case it's specified)
    label_mapping_name = "label_mapping.yaml"
    label_data_path = os.path.join(current_script_dir, relative_path_to_config_folder, label_mapping_name)
    with open(label_data_path, "r") as ymlfile:
        label_mapping = yaml.safe_load(ymlfile)

    # binary test labels for some binary datasets
    if training_args['num_classes'] == 2:
        label_mapping_name = "binary_labels_mapping.yaml"
        binary_labels_path = os.path.join(current_script_dir, relative_path_to_config_folder, label_mapping_name)
        with open(binary_labels_path, "r") as ymlfile:
            binary_labels = yaml.safe_load(ymlfile)
        for key in binary_labels.keys():
            if key in training_args['dataset']:
                args['binary_labels_eval'] = True
                args['binary_labels_mapping'] = binary_labels[key]
    else:
        args['binary_labels_eval'] = False
        args['binary_labels_mapping'] = None

    # Get label mapping and folder locs
    args['base_directory'] = training_args['base_directory']
    args['dataset'] = training_args['dataset']
    # args['data_root'] = [paths['datasets'][dataset] for dataset in training_args['dataset']]
    args['label_mapping'] = label_mapping['label_mapping']

    # Add nonetype if there is no label_mapping (this invokes loader to use its own labels)
    for dataset in args['eval_datasets']:
        if dataset not in args['label_mapping'].keys():
            args['label_mapping'][dataset] = None

    args["datasets_paths"] = {dataset: paths["datasets"][dataset] for dataset in args["eval_datasets"]}

    if args['models_eval'] == 'best':
        args["models_eval"] = ['best']
    elif args['models_eval'] == 'all':
        for chkpt in os.listdir(args["directory"]+'/checkpoints/'):
            args["models_eval"].append(chkpt.split('.pth')[0])
    else:
        raise ValueError(f"args[models_eval] has unknown value {args['models_eval']}")

    # Save pickle
    with open(args['directory'] + '/eval_args.pickle', 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\ntrain args:\n")
    print(training_args)
    print("eval args:\n")
    print(args)
    return args, training_args