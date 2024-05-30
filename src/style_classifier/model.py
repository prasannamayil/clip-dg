# model.py Imports
import torch
import torch.nn as nn
import torchvision.models as models
import open_clip
from torchvision import transforms
from typing import Sequence


class ImageEncoderWrapper(nn.Module):
    def __init__(self, encoder_model, output_size, image_size, dino=False, normalize=True):
        super(ImageEncoderWrapper, self).__init__()
        self.encoder = encoder_model
        self.normalize = normalize
        self.dino = dino

        # get input size
        with torch.no_grad():
            # Dummy input
            dummy_input = torch.randn(1, 3, image_size,
                                      image_size)  # Assuming input size of (batch_size, channels, height, width)
            # Pass the dummy input through the encoder and get size
            if not self.dino:
                dummy_output = self.encoder.encode_image(dummy_input)
            else:
                dummy_output = self.encoder(dummy_input)

            input_size = dummy_output.size()[-1]
            print(input_size)

        # init readout
        self.linear_readout = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Pass input through the image encoder
        if not self.dino:
            x = self.encoder.encode_image(x)
        else:
            x = self.encoder(x)
        # Normalize
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        # Pass through linear readout layer
        output = self.linear_readout(x)
        return output

    def freeze(self, encoder=False, linear_readout=False):
        if encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if linear_readout:
            for param in self.linear_readout.parameters():
                param.requires_grad = False


class SimpleLinearNet(nn.Module):
    def __init__(self, num_classes, image_size=224, num_channels=3):
        super(SimpleLinearNet, self).__init__()
        self.size = num_channels * image_size * image_size
        self.fc = nn.Linear(self.size, num_classes, bias=True)  # Input size is 3*224*224, output size is 2

    def forward(self, x):
        x = x.view(-1, self.size)  # Flatten the input images
        x = self.fc(x)  # Pass through linear layer
        return x


# model func
def get_model_and_transforms(args):
    """ Returns the model we want to train on and updates transforms
    """
    try:
        model_name = args.arch
        num_classes = args.num_classes
        pretrained = args.pretrained
        freeze_encoder = args.freeze_encoder

        # initializing transforms
        args.train_transform = None
        args.test_transform = None
    except:
        model_name = args['arch']
        num_classes = args['num_classes']
        pretrained = args['pretrained']
        freeze_encoder = args['freeze_encoder'] if 'freeze_encoder' in args.keys() else False

        # initializing transforms
        args['train_transform'] = None
        args['test_transform'] = None

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(512, num_classes, bias=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(2048, num_classes, bias=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(2048, num_classes, bias=True)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(2048, num_classes, bias=True)
    elif model_name == 'linear':
        # Create an instance of the network
        model = SimpleLinearNet(num_classes)
    elif 'laion' in model_name:
        # get open_clip image encoder and transforms
        if not pretrained:
            pretrained_data = ''
        else:
            pretrained_data = model_name.split('__')[1]

        encoder, train_transform, test_transform = open_clip.create_model_and_transforms(
            model_name.split('__')[0],
            pretrained=pretrained_data,
            precision='fp16' if pretrained_data == 'openai' else 'fp32',  # openai models use half precision
            jit=True if pretrained_data == 'openai' else False  # openai models use half precision
            )
        # wrap it with a linear head
        image_size = 240 if 'plus' in model_name else 224
        model = ImageEncoderWrapper(encoder, num_classes, image_size)

        # freezing
        model.freeze(encoder=freeze_encoder)

        # update transforms
        try:
            args.train_transform = train_transform
            args.test_transform = test_transform
        except:
            args['train_transform'] = train_transform
            args['test_transform'] = test_transform


    elif 'dino' in model_name:
        encoder = torch.hub.load('facebookresearch/dSino:main', 'dino_vits16')
        image_size = 224  #TODO unsure if 224 is the right size
        model = ImageEncoderWrapper(encoder, num_classes, image_size, dino=True, normalize=False)

        # freezing
        model.freeze(encoder=freeze_encoder)

        # update transforms
        train_transform = make_classification_train_transform()
        test_transform = make_classification_eval_transform()
        try:
            args.train_transform = train_transform
            args.test_transform = test_transform
        except:
            args['train_transform'] = train_transform
            args['test_transform'] = test_transform
    else:
        raise ValueError(f"Given model (model key = {model_name} doesn't exist")

    return model


# Dinov2 transforms

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)
