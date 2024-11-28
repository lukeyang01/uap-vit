# UAP Replication

This is where we're recreating the results of Moosavi-Dezfooli et al.'s original UAP paper [M16] (as opposed to extending them to ViTs). Precisely, as stated in our project proposal, "we plan to attack DNNs pre-trained on the ImageNet dataset, including the VGG family of networks and the ILSVRC-winning ResNet family of networks. These networks are precisely the ones uses in [M16]."

## Setup

1. Run extract-imagenet-subset.ipynb as a Kaggle notebook. Download and unzip the output.
2. Run reformat_validation_data.py.
3. Run 

## Experiment 1

From [M16]:

> In a first experiment, we assess the estimated universal perturbations for different recent deep neural networks on the ILSVRC 2012 validation set (50,000 images), and report the fooling ratio [...] Results are reported for p=2 and p=\infty, where we respectively set \xi = 2000 and \xi = 10.

> Results are listed in Table 1. Each result is reported on the set X, which is used to compute the perturbation, as well as on the validation set (that is not used...) [...] The above universal perturbations are computed for a set X of 10,000 images from the training set.

### Replication

The authors do not mention how many epochs they run DeepFool for. I doubt they set \delta and 



## Experiment 2

> We show in Fig. 6 the fooling rates obtained on the validation set for different sizes of X for GoogLeNet.