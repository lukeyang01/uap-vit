# Notes:
# Models should output pre-softmax activiations for DeepFool.
# Models should be pre-trained on ImageNet-1K (so output tensors of shape 1000).
# For info on pre-trained models, see pytorch.org/vision/stable/models.html.


import itertools
import numpy as np
import torchvision.models as tvm
import data
import uap

MODELS = (
    ("GoogLeNet", tvm.GoogLeNet_Weights.IMAGENET1K_V1, lambda w: tvm.googlenet(weights=w)),
    # ("ResNet", tvm.ResNet152_Weights.IMAGENET1K_V2, lambda w: tvm.resnet152(weights=w)),
    # ("VGG16", tvm.VGG16_Weights.IMAGENET1K_V1, lambda w: tvm.vgg16(weights=w)),
    # ("VGG19", tvm.VGG19_Weights.IMAGENET1K_V1, lambda w: tvm.vgg19(weights=w))
)

XI = (1000, )
P = (2, )

TRAINING_PATH = "imagenet_data/ILSVRC/Data/CLS-LOC/train"
TRAINING_SIZE = 1 # Number of images to train on.

VALIDATION_PATH = "imagenet_data/ILSVRC/Data/CLS-LOC/val"
VALIDATION_SIZE = 100 # Number of images to validate on.

SAVE_PATH = "compute_uaps/"


for ((name, weights, model_func), xi, p) in itertools.product(MODELS, XI, P):
    summary = f"uap_xi_{xi}_p_{p}_{name}"
    print(f"Fooling Model {summary}")
    
    print("Initializing Model")
    clf = model_func(weights)
    clf.eval()

    print("Loading Data")
    transforms = weights.transforms()
    train = data.load_random_subset(TRAINING_PATH, TRAINING_SIZE, transforms)
    # valid = data.load_random_subset(VALIDATION_PATH, VALIDATION_SIZE, transforms)

    print("Computing UAP")
    cb = lambda (i, v): data.save_uap(v, f"{summary}_{i}.pt")
    v = uap.compute_uap(train, clf, cb=cb, max_iter=5, xi=xi, p=p)

    print("Saving UAP")
    data.save_uap(v, f"{summary}.pt")

    # print("Computing Fooling Rate")
    # rate = uap.compute_fooling_rate()
    # print(f"(Model; Fooling Rate) = ({model.__name__}; {rate}")
