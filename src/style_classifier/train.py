import wandb
import numpy as np
import torch
import torch.nn as nn
import sys
import random
import os

from .config import *
from .model import *
from .utils import *
# Add the parent directory to the sys.path

from data.loaders import *


if __name__ == '__main__':
    args = get_args_train()

    wandb.init(
        project="",
        notes="",
        tags=['', str(args.arch), str(args.fourier), str(args.shift_images), str(args.shift_spectrum)],
        config=args)

    device, device_ids = setup_device(args.number_of_gpus)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    print('==> Building model and transforms..')
    net = get_model_and_transforms(args)  # potentially model is updated in args.
    net = net.to(device)

    # Data
    trainloader, vallooader, testloader = dataloader(args)
    loaders = {'train': trainloader, 'val': vallooader,  'test': testloader}  # No test

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.directory_net), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.directory_net+'current.pth')
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net, args)
    scheduler = get_scheduler(optimizer, args)

    # Parallelization of the network
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.to(device)

    # current and best accuracies
    results = {}
    for stat in ['loss', 'accuracy']:
        results[stat] = {}
        for mode in ['train', 'val', 'test']:
            results[stat][mode] = []

    current_acc = {'train': 0, 'val': 0, 'test': 0}
    best_acc = {'train': 0, 'val': 0, 'test': 0}
    current_loss = {'train': 0.0, 'val': 0.0, 'test': 0.0}
    best_loss = {}
    best_loss['train']: 0.0
    best_loss['val'] = 100000.0  # some big number
    best_loss['test']: 0.0

    # Training process
    print("Starting training")
    start_time_epoch = time()
    patience_es = 0
    best_epoch = 0

    for e in range(start_epoch, start_epoch + args.epochs):
        for mode in ['train', 'test', 'val']:
            avg_accuracy, avg_loss = one_epoch(loaders[mode], net, device,
                                               criterion, mode, args.sample_size, optimizer=optimizer,
                                               fourier=args.fourier, shift_images=args.shift_images,
                                               shift_spectrum=args.shift_spectrum)
            # Update stats and results
            current_acc[mode] = avg_accuracy
            current_loss[mode] = avg_loss
            results['loss'][mode].append(avg_loss)
            results['accuracy'][mode].append(avg_accuracy)

            wandb.log({mode: {"acc": avg_accuracy, "loss": avg_loss}})

            # Store best stats
            if mode == 'val':
                if best_loss['val'] >= current_loss['val']:
                    best_epoch = e

                    for m in ['train', 'test', 'val']:
                        best_loss[m] = current_loss[m]
                        best_acc[m] = current_acc[m]

                        # Save best stats to wandb
                        wandb.run.summary["best_"+m+"_accuracy"] = best_acc[m]
                    wandb.run.summary["best_epoch"] = e

                    patience_es = 0
                    best = True
                else:
                    patience_es += 1
                    best = False
                if args.save:
                    save_model(args.directory_net, e, net,
                               optimizer, scheduler, device_ids, best=best)  # Current or best

                    # save weights every nth epoch or the 5th epoch
                    if ((e+1) % args.save_after_epoch == 0) or ((e+1) == 5):
                        save_model(args.directory_net, e, net,
                                   optimizer, scheduler, device_ids, filename=str(e+1), best=False)

        # Printing stats after an epoch
        wandb.run.summary["current_epoch"] = e+1
        rough_epoch_time = time() - start_time_epoch
        start_time_epoch = time()
        print(f"epoch = {e + 1}, train acc (Top 1) = {current_acc['train']},"
              f"val acc (Top 1) = {current_acc['val']}, test acc (Top 1) = {current_acc['test']}"
              f"train loss = {current_loss['train']}, "
              f"val loss = {current_loss['val']}, \n"
              f"best epoch = {best_epoch},"
              f"best train acc (Top 1) = {best_acc['train']},"
              f"best val acc (Top 1) = {best_acc['val']}, best test acc (Top 1) = {best_acc['test']}, \n"
              f"epoch time = {rough_epoch_time}"
              )

        # Step learning rate
        wandb.log({'lr': optimizer.param_groups[0]["lr"]})
        scheduler.step()

        # Write results onto summary.txt
        summary_stats = ["Train Accuracy Current: " + str(current_acc['train']),
                         "Val Accuracy Current: " + str(current_acc['val']),
                         "Test Accuracy Current: " + str(current_acc['test']),
                         "Current Epoch: " + str(e),
                         "Train Accuracy Best: " + str(best_acc['train']),
                         "Val Accuracy Best: " + str(best_acc['val']),
                         "Val Accuracy Best: " + str(best_acc['test']),
                         "Best Epoch: " + str(best_epoch),
                         "Rough Epoch Time: " + str(rough_epoch_time) + " s"]

        if args.save:
            # Save args
            args.resume_epoch = e
            args.best_epoch = best_epoch
            args_dict = vars(args)
            with open(args.directory + 'args.pickle', 'wb') as handle:
                pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save results
            with open(args.directory + 'results.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save best acc
            with open(args.directory + 'best_acc.pickle', 'wb') as handle:
                pickle.dump(best_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save summary
            with open(args.directory + 'summary.txt', 'a') as f:
                f.write('\n')
                for line in summary_stats:
                    f.write(line)
                    f.write('\n')

            # Early Stopping
            if (patience_es >= args.patience_es) and args.early_stopping:
                print(f"Early stopping of training, waited for {patience_es} epochs")
                break