model_name=  # name of model
checkpoints_dir=  # location where the model's checkpoints re
cnn= # is the model a CNN
pretrained=  # is the model pretrained

python3 zero_shot_eval.py -m "$model_name" -c "$checkpoints_dir" --cnn "$cnn"  -p "$pretrained"
