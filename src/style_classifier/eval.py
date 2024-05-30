import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import wandb

from data.loaders import *
from .config import *
from .model import *
from .utils import *
from .madry_model import *
from .centroid_model import *


def main(args, training_args):
    # Get devices ready
    device, device_ids = setup_device(args['number_of_gpus'])

    # Get models, optimizer, criterion, and scheduler
    models = {}
    optimizers = {}
    for model_key in args["models_eval"]:
        if 'centroids.pt' in os.listdir(training_args["directory"]+"/checkpoints/"):
            PATH = training_args["directory"] + "/checkpoints/centroids.pt"
            centroids = torch.load(PATH)
            models[model_key] = CentroidModel(training_args["backbone"], centroids)
            models[model_key].eval()
            training_args['test_transform'] = models[model_key].transform
        else:
            try:
                PATH = args["directory"]+"/checkpoints/"+model_key+".pth"
                checkpoint = torch.load(PATH)

                models[model_key] = get_model_and_transforms(training_args)
                models[model_key].load_state_dict(checkpoint['state_dict'])
                models[model_key].eval()

                optimizers[model_key] = get_optimizer(models[model_key], training_args)
                optimizers[model_key].load_state_dict(checkpoint['optimizer'])
            except:
                PATH = args["directory"]+"/checkpoints/"+model_key+".pt"
                checkpoint = torch.load(PATH)
                model_name = training_args['arch'].split('__')[0]
                pretrained = training_args['arch'].split('__')[1]
                model = construct_clip_model(training_args['num_classes'], model_name, pretrained)
                model.load_state_dict(checkpoint)
                models[model_key] = model
                training_args['test_transform'] = model.model.preprocessor

    # get dataset(s) and transforms
    args['test_transform'] = training_args['test_transform']
    print(args['test_transform'])
    dl = {}
    for dataset in args["eval_datasets"]:
        args['test_data_root'] = args['datasets_paths'][dataset]
        # need to update test transform
        if 'laion' in dataset:
            dl[dataset] = get_wds_dataloader_test(args, meta_data=True)
        else:
            dl[dataset] = dataloader_test_IMAGEFOLDER(args)

    criterion = nn.CrossEntropyLoss()  # the loss function

    # Parallelization of the network
    if len(device_ids) > 1:
        for model_key in args["models_eval"]:
            models[model_key] = torch.nn.DataParallel(models[model_key], device_ids=device_ids)
    for model_key in args["models_eval"]:
        models[model_key] = models[model_key].to(device)

    # Summary stats
    # best_acc = {'top1': {}, 'top5': {}}

    # if accuracy exists then load it instead of overwriting it
    if os.path.exists(args['directory'] + '/accuracy.pickle'):  # resuming zero shot eval
        with open(args['directory'] + '/accuracy.pickle', 'rb') as handle:
            best_acc = pickle.load(handle)
    else:
        best_acc = {}

    # Evaluation process
    print("Starting Evaluation")

    for dataset in args["eval_datasets"]:
        best_acc[dataset] = {}
        for model_key in args["models_eval"]:
            print(f"evaluating {model_key} on {dataset}")
            save_dir_eval = args['directory']+'/'+dataset+'/'
            os.makedirs(save_dir_eval, exist_ok=True)

            avg_accuracy_1, _ = one_epoch(dl[dataset], models[model_key], device,
                                          criterion, 'eval', -1, optimizer=None, save_preds=True,
                                          save_preds_number=args['save_preds_number'], save_dir=save_dir_eval,
                                          labels_exp=args['label_mapping'][dataset],
                                          binary_labels=args['binary_labels_eval'],
                                          binary_mapping=args['binary_labels_mapping'])

            # Update stats
            best_acc[dataset][model_key] = avg_accuracy_1

            wandb.run.summary[dataset + "_" + model_key + "_accuracy_1"] = avg_accuracy_1
            # wandb.run.summary[dataset + "_" + model_key + "_accuracy_5"] = avg_accuracy_5

            # pickle and save results
            with open(args['directory'] + '/accuracy.pickle', 'wb') as handle:
                pickle.dump(best_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Write results onto summary.txt
    summary_stats = [model_key + " model acc on " + dataset + " = " + str(best_acc[dataset][model_key])
                     for dataset in args["eval_datasets"] for model_key in args["models_eval"]]

    # summary_stats = ['training dataset = ' + args['training_dataset']] + \
    #                 [model_key + " model acc on " + dataset + " = " + str(best_acc['top1'][dataset][model_key])
    #                  for dataset in args["eval_datasets"] for model_key in args["models_eval"]] + \
    #                 [model_key + " model acc on " + dataset + " = " + str(best_acc['top5'][dataset][model_key])
    #                  for dataset in args["eval_datasets"] for model_key in args["models_eval"]]

    with open(args['directory'] + '/eval_summary.txt', 'a') as f:
        f.write('\n')
        for line in summary_stats:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    args, training_args = get_args_eval()
    print(args)
    wandb.init(
        project="",
        notes="style classifier",
        config=args,
        settings=wandb.Settings(start_method='fork'))
    main(args, training_args)