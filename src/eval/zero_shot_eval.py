from utils import *
from classnames import *
from templates import *
import torchvision.datasets as datasets
import argparse
import open_clip
import numpy as np
import torch
import pickle
import yaml
import os
from tqdm import tqdm


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# Calculates accuracy of a clip model given dataloader
def calculate_cnn_accuracy(dataloader, dataset_name, model, device):
    labels = []
    predictions = []

    avg_acc = 0
    for img, label in tqdm(dataloader):
        img = img.to(device, non_blocking=True)
        label = label.to(device)

        # compute predictions if imagenet-r or 200 take only 200 logits
        if (dataset_name == 'imagenet-r') or (dataset_name == 'imagenet-200'):
            out = model(img)[:, imagenet_r_mask]
        elif (dataset_name == 'imagenet-a'):
            out = model(img)[:, imagenet_a_mask]
        elif (dataset_name == 'objectnet-subsample'):
            out = model(img)[:, objectnet_subsample_mask]
        else:
            out = model(img)

        out = out.argmax(1)
        acc_batch = ((label == out).sum()).float()
        avg_acc += acc_batch.item()

        # Append predictions and labels
        labels.append(label.cpu().detach())

    # Compute acc
    labels = torch.cat(labels).numpy()
    acc = avg_acc / len(labels)

    return acc


# Calculates accuracy of a clip model given dataloader
def calculate_accuracy(dataloader, model, class_matrix, device):
    with torch.no_grad():
        labels = []
        predictions = []

        for img, label in tqdm(dataloader):
            img = img.to(device, non_blocking=True)
            img_emb = encode_image(model, img)

            # compute predictions
            pred = torch.argmax(img_emb @ class_matrix.T, dim=-1)

            # Append predictions and labels
            labels.append(label.cpu().detach())
            predictions.append(pred.cpu().detach())

        # Compute acc
        predictions = torch.cat(predictions).numpy()
        labels = torch.cat(labels).numpy()
        acc = np.sum(predictions == labels) / len(labels)

    return acc, predictions, labels


def get_dataset(dataset_name, batch_size, num_workers, transform, dataset_directories):
    '''Add subset idcs to only test on a subset of the eval dataset
    '''
    datadir = dataset_directories[dataset_name]

    dataset = datasets.ImageFolder(datadir, transform=transform)

    # get the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size,
                                             pin_memory=False, num_workers=num_workers)
    return dataloader


# main function
def main(args):
    # model args
    model_name = args['model_name']
    checkpoints_path = args['checkpoints_path']
    checkpoints_epochs = args['checkpoints_epochs']
    pretrained = args['pretrained']
    jit = args['jit']
    cnn = args['cnn']

    # data args
    dataset_names = args['dataset_names']
    batch_size = args['batch_size']
    num_workers = args['num_workers']

    # save args
    save_dir = args['save_dir']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(save_dir + 'accuracy.pickle'):  # resuming zero shot eval
        with open(save_dir + 'accuracy.pickle', 'rb') as handle:
            acc_dict = pickle.load(handle)
    else:
        acc_dict = {}

    model, transform = get_model_and_transform(cnn, model_name, pretrained, device, jit)

    # For each epoch in a list of epochs (models of different epochs)
    for epoch in checkpoints_epochs:
        if epoch not in list(acc_dict.keys()):  # resuming zero shot eval
            acc_dict[epoch] = {}


        # Load checkpoint if model weights are given
        if not pretrained:
            model.train()  # change mode before load (maybe unnecessary)
            checkpoint_path = checkpoints_path + 'epoch_' + str(epoch) + '.pt'
            checkpoint = torch.load(checkpoint_path)
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        model.eval()
        print("Model loaded")

        # Compute accuracy on each new dataset
        for dataset_name in dataset_names:
            if dataset_name in list(acc_dict[epoch].keys()):  # resuming zero shot eval
                continue
            else:
                # Get loader
                dataloader = get_dataset(dataset_name, batch_size, num_workers, transform, args["datasets_paths"])
                print(f"got dataloader for {dataset_name}")
                if cnn:
                    acc_dict[epoch][dataset_name] = calculate_cnn_accuracy(dataloader, dataset_name, model, device)

                else:
                    # For generating an appropriately sized class matrix
                    if ('imagenet' in dataset_name) or ('objectnet' in dataset_name):
                        if dataset_name in masks.keys():
                            mask = masks[dataset_name]
                            classnames = []
                            for i in range(len(imagenet_classnames)):
                                if mask[i]:
                                    classnames.append(imagenet_classnames[i])
                        else:
                            classnames = imagenet_classnames
                    elif 'dn' in dataset_name:
                        if dataset_name in masks.keys():
                            mask = masks[dataset_name]
                            classnames = []
                            for i in range(len(domainnet_classnames_sorted)):
                                if mask[i]:
                                    classnames.append(domainnet_classnames_sorted[i])
                        else:
                            classnames = domainnet_classnames_sorted
                    else:
                        raise ValueError(f"wrong dataset name: {dataset_name}")

                    class_matrix = get_class_matrix(model, classnames, openai_imagenet_template)

                    # Compute and store accuracy
                    acc_dict[epoch][dataset_name], predictions, labels = calculate_accuracy(dataloader, model, class_matrix, device)

                    # make eval data dir
                    if epoch == 32:
                        save_dir_eval = save_dir+'/'+dataset_name+'/'
                        os.makedirs(save_dir_eval, exist_ok=True)
                        np.save(save_dir_eval+'preds.npy', predictions)
                        np.save(save_dir_eval+'labels.npy', labels)

                with open(save_dir + 'accuracy.pickle', 'wb') as handle:
                    pickle.dump(acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(f"Epoch: {epoch}, Dataset: {dataset_name}, with acc {acc_dict[epoch][dataset_name]} done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dir args
    parser.add_argument("-b", "--base_save_dir", type=str,
                        default='', help='saving accuracies')
    parser.add_argument("-n", "--name", type=str,
                        default=None, help='directory')

    # model args
    parser.add_argument("-m", "--model_name", type=str, default='ViT-B-32', help='model_name')
    parser.add_argument("-p", "--pretrained", type=boolean_string,
                        default=False,
                        help='using an already pretrained model')  # set to True if using a pretrained model(VERY IMP)

    parser.add_argument("--cnn", type=boolean_string,
                        default=False,
                        help='using an already pretrained model')  # set to True if using a CNN else set to False

    parser.add_argument("-c", "--checkpoints_path", type=str,
                        default='', help='model_name')

    # data args
    parser.add_argument("--batch_size", type=int, default=500, help='batch size')
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for the data loader')
    parser.add_argument("--dataset", type=str, default=None, help='dataset trained on')

    ap = parser.parse_args()
    args = vars(ap)

    # get paths and dataset names
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    relative_path_to_paths_folder = "../paths"
    paths_file_name = "paths.yaml"
    paths_file_path = os.path.join(current_script_dir, relative_path_to_paths_folder, paths_file_name)
    with open(paths_file_path, "r") as ymlfile:
        paths = yaml.safe_load(ymlfile)

    args['dataset_names'] = ['imagenet-a', 'objectnet-subsample', 'imagenet-r', 'imagenet-sketch',
                             'imagenet-v2', 'imagenet-val', 'dn-real', 'dn-sketch', 'dn-quickdraw', 'dn-infograph',
                             'dn-clipart', 'dn-painting', 'imagenet-a_clean', 'objectnet-subsample_clean',
                             'imagenet-r_clean', 'imagenet-sketch_clean', 'imagenet-v2_clean',
                             'imagenet-val_clean', 'dn-real_clean', 'dn-sketch_clean', 'dn-quickdraw_clean',
                             'dn-infograph_0_clean', 'dn-infograph_1_clean',
                             'dn-clipart_clean', 'dn-painting_clean']

    args["datasets_paths"] = {dataset: paths["datasets"][dataset] for dataset in args["dataset_names"]}

    # This is only for CLIP models
    if args['pretrained']:
        pretrained_data = args['model_name'].split('__')[1]
        if pretrained_data == 'openai':
            args['jit'] = True  # jit = True needed for some openai models
        else:
            args['jit'] = False

        args['save_dir'] = args['base_save_dir'] + args['model_name'] + '/'  # model_name has also the data it's trained on
        if args['name'] is not None:
            args['save_dir'] = args['save_dir'] + args['dataset'] + '/' + args['name'] + '/'

        args['checkpoints_epochs'] = [-1]  # model epochs to be evaluated (Last/Best epoch)
    else:
        args['jit'] = False

        args['save_dir'] = args['base_save_dir'] + args['model_name'] + '_' + args['checkpoints_path'].split('/')[-3]+'/'

        args['checkpoints_epochs'] = [i for i in range(1, len(os.listdir(args['checkpoints_path']))+1)]
        args['checkpoints_epochs'].reverse()   # start from last epoch

    # Create save directory
    os.makedirs(args['save_dir'], exist_ok=True)

    # run  main
    print(args)
    main(args)
    print("All done")
