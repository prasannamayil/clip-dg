import torch
import torch.utils.data
import torchvision
#from torchvision.transforms import v2 as transforms
from torchvision import transforms
# import lucent.optvis.hooks
# import lucent.modelzoo.util
import open_clip
from typing import List


def get_data_loader(
        path: str,
        batch_size: int = 256,
        transform=None,
        shuffle: bool = True
):
    """Get a dataloader for the annotated dataset."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    dataset = torchvision.datasets.ImageFolder(
        path,
        transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def get_model_and_transform(model_name: str) -> torch.nn.Module:
    """Load a pre-trained PyTorch model."""
    if model_name.startswith("torch://"):
        model_name = model_name[len("torch://"):]
        model_cls = getattr(torchvision.models, model_name)
        weights_name = f"{model_name.upper()}_Weights"
        model_kwargs = {}
        if hasattr(torchvision.models, weights_name):
            weights = getattr(torchvision.models, weights_name).DEFAULT
            model_kwargs["weights"] = weights
        else:
            print("Could not detect correct weights for model_name")
            model_kwargs["pretrained"] = True

        model = model_cls(**model_kwargs)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return model, transform
    elif model_name.startswith("open_clip://"):
        model_name = model_name[len("open_clip://"):]
        if "#" not in model_name:
            print(
                "Attention: Did not detect information about training data in model "
                "name. Using potentially untrained model.")
            training_data = None
        else:
            model_name, training_data = model_name.split("#")
        model, _, test_transform = open_clip.create_model_and_transforms(
            model_name,
            pretrained=training_data,
            precision=training_data,
            # openai models use half precision
            jit=True if training_data == 'openai' else False
        )

        if hasattr(model, "visual"):
            model = model.visual

        return model, test_transform
    elif model_name.startswith("torch_hub://"):
        model_name = model_name[len("torch_hub://"):]
        model = torch.hub.load(*model_name.split("#"))

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return model, transform
    else:
        raise NotImplementedError("Only torch models are implemented yet.")


# def prepare_intermediate_features_model_and_transform(
#         model_name: str,
#         layer_names: List[str],
#         device: str
# ) -> lucent.optvis.hooks.ModelHook:
#     model, transform = get_model_and_transform(model_name)
#
#     model = model.to(device)
#
#     layer_names = [l for l in layer_names if l != "output"]
#
#     available_layer_names = lucent.modelzoo.util.get_model_layers(model)
#     for layer_name in layer_names:
#         if layer_name not in available_layer_names:
#             raise RuntimeError(
#                 f"Invalid layer name. Valid names are: `{available_layer_names}`")
#
#     hook = lucent.optvis.hooks.ModelHook(model, layer_names=layer_names)
#     return transform, model, hook
