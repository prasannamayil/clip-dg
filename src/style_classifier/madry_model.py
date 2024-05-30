import torch
import torch.nn as nn
import open_clip
import torchvision


class CLIPClassifier(torch.nn.Module):
    def __init__(self, model_name, pretrained, num_classes, class_names=None, templates=None):
        super().__init__()
        model, _, self.preprocessor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.cuda()
        self.image_encoder = model.visual
        if class_names is None:
            if isinstance(self.image_encoder, open_clip.transformer.VisionTransformer):
                num_features, = self.image_encoder.ln_post.normalized_shape
                self.image_encoder.proj = None
            else:
                num_features = self.image_encoder.output_dim
            dummy = torch.nn.Linear(num_features, num_classes)
            self.classification_head = ClassificationHead(False, dummy.weight, dummy.bias)
        else:
            weights = get_zero_shot_weights(model, model_name, class_names, templates=templates)
            self.classification_head = ClassificationHead(True, weights)

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.classification_head(x)
        return x


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(ch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class ResizeWrapper(torch.nn.Module):
    def __init__(self, model, size):
        super().__init__()
        self.resize = torchvision.transforms.Resize(size,
                                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.center_crop = torchvision.transforms.CenterCrop(size)
        self.model = model

    def forward(self, x):
        x = self.resize(x)
        x = self.center_crop(x)
        return self.model(x)


def construct_clip_model(num_classes, model_name, pretrained, class_names=None, templates=None, features_only=False):
    model = CLIPClassifier(model_name, pretrained, num_classes, class_names=class_names, templates=templates)
    size = model.preprocessor.transforms[0].size
    if features_only:
        model = model.image_encoder
    model = ResizeWrapper(model, size)
    return model