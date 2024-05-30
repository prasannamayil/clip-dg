import torch
import pickle
import glob
import numpy as np
import torch.optim as optim
from time import time
import argparse
import shutil
import yaml
import torch.nn.functional as F
import os


def setup_device(required_gpus):
    """
    Configures the device for training based on the specified number of GPUs.

    Parameters:
    - required_gpus (int): The number of GPUs to be used for training.

    Returns:
    - device (torch.device): The selected device ('cuda:0' if using GPUs, 'cpu' otherwise).
    - list_ids (list): A list of GPU IDs to be utilized. Empty if training on CPU.

    Example:
    ```python
    device, list_ids = setup_device(2)
    model.to(device)
    ```
    """
    actual_gpus = torch.cuda.device_count()

    # Check if GPUs are available and adjust required_gpus accordingly
    if required_gpus > 0 and actual_gpus == 0:
        print("Warning: There's no GPU available on this machine, training will be performed on CPU.")
        required_gpus = 0

    # Adjust required_gpus if the specified number exceeds the available GPUs
    if required_gpus > actual_gpus:
        print("Warning: The number of GPUs configured to use is {}, but only {} are available on this machine.".format(
            required_gpus, actual_gpus))
        required_gpus = actual_gpus

    # Select the appropriate device
    device = torch.device('cuda:0' if required_gpus > 0 else 'cpu')

    # Generate a list of GPU IDs to be utilized
    list_ids = list(range(required_gpus))

    return device, list_ids


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, filename=None, best=False):
    """
    Saves the model, optimizer, and learning rate scheduler states to a specified file.

    Parameters:
    - save_dir (str): The directory where the model checkpoint will be saved.
    - epoch (int): The current epoch during training.
    - model (torch.nn.Module): The model to be saved.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    - device_ids (list): List of GPU IDs used for training.
    - filename (str, optional): The name of the file to save the checkpoint. Default is 'current.pth'.
    - best (bool, optional): If True, also saves the model as the best checkpoint. Default is False.

    Example:
    ```python
    save_model('checkpoints/', 5, my_model, my_optimizer, my_scheduler, [0, 1], filename='checkpoint_epoch_5', best=True)
    ```

    This function saves the model, optimizer, and learning rate scheduler states to a specified file in the given directory.
    If 'best' is set to True, it also saves the model as the best checkpoint separately.
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }

    # Set default filename if not provided
    if filename is None:
        filename = str(save_dir + 'current.pth')
    else:
        filename = str(save_dir + filename + '.pth')

    # Save the checkpoint
    torch.save(state, filename)

    # Save the model as the best checkpoint if 'best' is True
    if best:
        best_filename = str(save_dir + 'best.pth')
        torch.save(state, best_filename)


def get_optimizer(net, args):
    """
    Initializes and returns an optimizer based on the specified optimizer type and arguments.
    """
    try:
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError("Unsupported optimizer type. Supported types: 'SGD', 'Adam', 'AdamW'.")
    except:
        if args['optimizer'] == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=args['lr'],
                                  momentum=args['momentum'],
                                  weight_decay=args['weight_decay'])
        elif args['optimizer'] == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        elif args['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        else:
            raise ValueError("Unsupported optimizer type. Supported types: 'SGD', 'Adam', 'AdamW'.")
    return optimizer


def get_scheduler(optimizer, args):
    """
    Initializes and returns an optimizer based on the specified optimizer type and arguments.
    """
    try:
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        else:
            raise ValueError("Unsupported scheduler type. Supported types: 'step', 'cosine'.")
    except:
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
        else:
            raise ValueError("Unsupported scheduler type. Supported types: 'step', 'cosine'.")
    return scheduler


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def one_epoch(data_loader, net, device, criterion, mode, sample_size, optimizer=None,
              num_batches_print=100, fourier=False, shift_images=False, shift_spectrum=False,
              labels_exp=None, save_preds=False, save_preds_number=100000, save_dir=None,
              binary_labels=False, binary_mapping=None):
    """
    Doing one epoch of training/testing/validating
    """
    # Initialize variables that accumulate stats
    avg_accuracy = 0
    avg_loss = 0
    total_images = 0
    start_time = time()
    start_time_save = time()

    # Change it to train mode if we want to train
    if mode == 'train':
        net.train()
    else:
        net.eval()

    # prediction vars
    j = 0
    preds_stack = []
    labels_stack = []
    meta_stack = []
    logits_stack = []
    prob_stack = []
    for i, data in enumerate(data_loader):

        # Break out of the function if we exceed the given sample_size (-1 = whole dataset) and it is not train
        if (sample_size != -1) and (total_images >= sample_size) and (mode != 'train'):
            break

        if len(data) == 2:
            images, labels = data
        else:
            images, labels, metadata = data

        # if explicit labels are given then discard data labels
        if labels_exp is not None:
            with torch.no_grad():
                labels = torch.from_numpy(np.array([labels_exp for i in range(len(labels))])).to(dtype=labels.dtype)

        # if binary labels then change it accordingly
        if binary_labels:
            with torch.no_grad():
                mapped_labels = torch.tensor([binary_mapping[label.item()] for label in labels])
                labels = mapped_labels.to(dtype=labels.dtype)

        images = images.to(device)

        with torch.no_grad():
            if fourier:
                # Inspired from https://github.com/YanchaoYang/FDA
                # if you want to do inverse shift to images before FFT
                if shift_images:
                    images = torch.fft.ifftshift(images, dim=(-2, -1))

                # Apply FFT to the shifted image
                images = torch.fft.fft2(images,  dim=(-2, -1))

                # Compute amplitude spectrum
                images = torch.abs(images)

                # Shift the zero frequency component to the center
                if shift_spectrum:
                    images = torch.fft.fftshift(images, dim=(-2, -1))

        labels = labels.to(device)

        # Get training ready
        out = net(images)
        loss = criterion(out, labels)

        with torch.no_grad():
            _, predictions = torch.max(out, 1)
            acc = (predictions == labels).sum()

            # get only out logits of predicted class
            logits = F.softmax(out, dim=-1)
            prob = logits[torch.arange(out.size(0)), :]
            logits = logits[torch.arange(out.size(0)), predictions]

        total_images += len(images)

        # Saving predictions
        if save_preds:
            predictions = predictions.cpu().detach()
            labels = labels.cpu().detach()
            logits = logits.cpu().detach()
            prob = prob.cpu().detach()

            preds_stack.append(predictions)
            labels_stack.append(labels)
            meta_stack.append(metadata)
            logits_stack.append(logits)
            prob_stack.append(prob)

            if (total_images % save_preds_number) == 0:
                preds_stack = torch.cat(preds_stack).numpy()
                labels_stack = torch.cat(labels_stack).numpy()
                logits_stack = torch.cat(logits_stack).numpy()
                prob_stack = torch.cat(prob_stack).numpy()

                # meta_stack = torch.cat(meta_stack).numpy()
                meta_stack = np.concatenate(meta_stack)

                np.save(save_dir + 'preds_'+str(j)+'.npy', preds_stack)
                np.save(save_dir + 'labels_'+str(j)+'.npy', labels_stack)
                np.save(save_dir + 'meta_'+str(j)+'.npy', meta_stack)
                np.save(save_dir + 'logits_'+str(j)+'.npy', logits_stack)
                np.save(save_dir + 'prob_'+str(j)+'.npy', prob_stack)

                print(f"time taken for {save_preds_number} images: {time() - start_time_save}, "
                      f"total_images done:{total_images}")
                start_time_save = time()

                preds_stack = []
                labels_stack = []
                logits_stack = []
                meta_stack = []
                prob_stack = []
                j += 1

        # Training model on that particular batch of images if it is training
        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Losses, Accuracies
        avg_loss += loss.item() * len(images)
        avg_accuracy += acc.item()

        # Print stats for 1000 batches
        if (i + 1) % num_batches_print == 0:
            print(
                "i = {} Accuracy = {} Loss = {}".format(i + 1, avg_accuracy / total_images, avg_loss / total_images))
            print(f"time taken for {num_batches_print} batches: {time() - start_time}")
            start_time = time()

    if save_preds:
        preds_stack = torch.cat(preds_stack).numpy()
        labels_stack = torch.cat(labels_stack).numpy()
        logits_stack = torch.cat(logits_stack).numpy()
        prob_stack = torch.cat(prob_stack).numpy()
        meta_stack = np.concatenate(meta_stack)

        np.save(save_dir + 'preds_' + str(j) + '.npy', preds_stack)
        np.save(save_dir + 'labels_' + str(j) + '.npy', labels_stack)
        np.save(save_dir + 'meta_' + str(j) + '.npy', meta_stack)
        np.save(save_dir + 'logits_' + str(j) + '.npy', logits_stack)
        np.save(save_dir + 'prob_' + str(j) + '.npy', prob_stack)

        # combine all chunks
        combine_and_delete_numpy_files(save_dir)
    # return stats
    return avg_accuracy / total_images, avg_loss / total_images


# Convert dict args to argparse
def dict_to_args(args_dict):
    """
    Converts dict to args. Only handles types: int, float, str, bool
    """
    parser = argparse.ArgumentParser(description='Command line arguments from a dictionary')

    # Iterate over the dictionary and add each key as an argument
    for key, value in args_dict.items():
        # Determine the appropriate type for the argument (str, int, float, bool, etc.)
        # You can customize this based on your dictionary's values.
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', help=f'Set {key} to True')
        elif isinstance(value, int):
            parser.add_argument(f'--{key}', type=int, default=value, help=f'Specify {key} (default: {value})')
        elif isinstance(value, float):
            parser.add_argument(f'--{key}', type=float, default=value, help=f'Specify {key} (default: {value})')
        elif isinstance(value, str):
            parser.add_argument(f'--{key}', type=str, default=value, help=f'Specify {key} (default: {value})')
        else:
            continue

    # Parse the command-line arguments
    args = parser.parse_args([])
    return args


# Copy yaml file
def copy_yaml_file(source_path, destination_path):
    try:
        # Read YAML file
        with open(source_path, 'r') as source_file:
            yaml_data = yaml.safe_load(source_file)

        # Write YAML data to the destination file
        with open(destination_path, 'w') as destination_file:
            yaml.dump(yaml_data, destination_file, default_flow_style=False)

        print(f"YAML file copied from {source_path} to {destination_path}")
    except Exception as e:
        print(f"Error: {e}")


# Combine chunks of preds, labels, logits
def combine_and_delete_numpy_files(folder_path, prefix_list=['preds', 'labels', 'logits', 'meta', 'prob']):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    for prefix in prefix_list:
        file_chunks = [file for file in file_list if file.startswith(prefix+'_') and file.endswith('.npy')]

        if not file_chunks:
            print("No files found to combine.")
            return

        # Sort the files based on the index j
        file_chunks.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Initialize an empty list to hold arrays
        arrays = []

        # Load arrays from files and append to the list
        for file in file_chunks:
            array = np.load(os.path.join(folder_path, file))
            arrays.append(array)

            # Delete the file after loading
            os.remove(os.path.join(folder_path, file))

        # Concatenate arrays along the first axis
        combined_array = np.concatenate(arrays, axis=0)

        # Save the combined array to 'preds.npy'
        np.save(os.path.join(folder_path, prefix+'.npy'), combined_array)