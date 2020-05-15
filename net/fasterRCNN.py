from torchvision.models.detection.faster_rcnn import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import torchvision
import logging


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Dataset')


def fasterRCNN_ResNet50_fpn(num_classes=72,
                            anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
                            aspect_ratios=((0.5, 1.0, 2.0),)*5,
                            image_mean=None, image_std=None):
    """
    Generate a Faster R-CNN with ResNet50 backbone

    :param num_classes: number of output classes of the model (including the background)
    :param anchor_sizes: List[Tuple[int],...], anchor sizes
    :param aspect_ratios: List[Tuple[int],...], aspect ratios

    For both anchor_sizes and aspect_ratios, you should specify as many tuples as the number of feature maps
    that you want to extract the RoIs from, anchor_sizes and aspect_ratios should have the same number of elements,
    aka. len(anchor_sizes) == len(aspect_ratios), e.g. sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0)), which will generate
    1x3 anchors per spatial location with 1 size and 3 different ratios for 5 feature maps. ResNet 50 with fpn backbone
    outputs 5 feature maps so the length of anchor_sizes and the aspect_ratios must be 5.
    If input size is 200x200, the output feature maps are {layer1:50x50, layer2:25x25, layer3:13x13, layer4:7x7,
    pool:4x4}.

    :param image_mean: List[int,int,int], mean values for RGB 3 channels
    :param image_std: List[int,int,int], std. values for RGB 3 channels
    Mean and std. values are used for normalization, calculated from the training dataset
    :return:
    """
    if len(anchor_sizes) != 5 or len(aspect_ratios) != 5:
        LOG.warning("The length of anchor_sizes and the aspect_ratios must be 5!")
        raise ValueError("The length of anchor_sizes and the aspect_ratios must be 5!")
    args = {'rpn_anchor_generator': AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios),
            'image_mean': image_mean,
            'image_std': image_std}

    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes, pretrained_backbone=False, **args)
    return model


def test():
    backbone = torchvision.models.mobilenet_v2(pretrained=False).features

    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

    # model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator)

    model.eval().cuda()

    x = [torch.rand(3, 300, 400).float().cuda(), torch.rand(3, 500, 400).float().cuda()]

    predictions = model(x)
    print(len(predictions))
    print(predictions[0].keys())


if __name__ == '__main__':
    test()
