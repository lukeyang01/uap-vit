import torchvision.models as tvm
import data
import uap


MODELS = (
    # ("GoogLeNet", tvm.GoogLeNet_Weights.IMAGENET1K_V1, lambda w: tvm.googlenet(weights=w)),
    # ("ResNet", tvm.ResNet152_Weights.IMAGENET1K_V2, lambda w: tvm.resnet152(weights=w)),
    # ("VGG16", tvm.VGG16_Weights.IMAGENET1K_V1, lambda w: tvm.vgg16(weights=w)),
    ("VGG19", tvm.VGG19_Weights.IMAGENET1K_V1, lambda w: tvm.vgg19(weights=w)),
)

UAP_PATHS = (
    # "results/uap_x_500_xi_30_p_2_GoogLeNet.pt",
    # "results/uap_x_500_xi_30_p_2_ResNet.pt",
    "results/uap_x_500_xi_30_p_2_VGG16.pt",
    # "results/uap_x_500_xi_30_p_2_VGG19.pt"
)

VALIDATION_PATH = "imagenet_data/ILSVRC/Data/CLS-LOC/val"
VALIDATION_SIZE = 1000 # Number of images to validate on.


for (name, weights, model_func) in MODELS:
    for path in UAP_PATHS:
        clf = model_func(weights)
        clf.eval()

        transforms = weights.transforms()
        validate = data.load_random_subset(VALIDATION_PATH, VALIDATION_SIZE, transforms)

        v = data.load_uap(path)
        fr = uap.compute_fooling_rate(validate, clf, v)
        
        print(f"UAP {path} | Model {name} | Fooling Rate {fr}")
