import os
import yaml
import numpy as np
import shutil
from argparse import ArgumentParser


# data root
def main(args):
    dataset_name = args.dataset_name
    save_dir_root = ''
    precision_level = 0.98

    # get classifier roots and dataset names
    classifier_root = {}
    classifier_root['stylistic'] = ''
    classifier_root['natural'] =''

    test_sets = {}
    test_sets['natural'] = ['imagenet-a', 'imagenet-v2', 'imagenet-val', 'dn-real', 'objectnet-subsample', 'dn-infograph']
    test_sets['stylistic'] = ['imagenet-sketch', 'imagenet-r', 'dn-sketch', 'dn-painting', 'dn-clipart', 'dn-quickdraw']

    classname = {}
    classname['natural'] = [0]
    classname['stylistic'] = [1]

    # get paths of all datasets
    paths_file_path = ''
    with open(paths_file_path, "r") as ymlfile:
        paths = yaml.safe_load(ymlfile)

    # data folder
    data_root = paths['datasets'][dataset_name]

    # get all sorted file paths
    # using exactly the same code as loader

    file_paths = []
    folder_paths = []
    for root, dirs, files in sorted(os.walk(data_root, followlinks=True)):
        for file in sorted(files):
            if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.JPEG')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                folder_paths.append(root)

    # clean folder_paths
    folder_paths = [folder_path.split('/')[-1] for folder_path in folder_paths]

    # get meta and preds
    for key in test_sets.keys():
        if dataset_name in test_sets[key]:
            dataset_type = key
            break

    meta = np.load(classifier_root[dataset_type] + dataset_name + '_' + str(precision_level) + '/meta.npy')
    preds = np.load(classifier_root[dataset_type] + dataset_name + '_' + str(precision_level) + '/preds.npy')

    # copy datapoints
    for cls in classname[dataset_type]:
        # get subset of paths that we want to copy
        locs = np.where(preds == cls)[0]
        file_paths_cls = [file_paths[meta[loc]] for loc in locs]
        folder_paths_cls = [folder_paths[meta[loc]] for loc in locs]

        # Destination folder
        for file, folder in zip(file_paths_cls, folder_paths_cls):
            destination_folder = save_dir_root + '/' + dataset_name + '_' + str(cls) + '/' + folder + '/'
            os.makedirs(destination_folder, exist_ok=True)

            shutil.copy(file, destination_folder)
        print(f"total {len(locs)} images saved for {dataset_name}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str,
        help='output folder')
    args = parser.parse_args()
    main(args)