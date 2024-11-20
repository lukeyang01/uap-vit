from model.vit import ViT
from train_common import *
from utils import config
from attacks import uap_sgd
from dataset import get_train_val_test_loaders
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from torchvision.models import vit_b_16
import argparse
import cv2

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', default='test_img.png', type=str)
    parser.add_argument('--pert_weights', default='saved_pert.npy', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # model = vit_b_16()

    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("vit.batch_size"),
    )

    model = ViT(num_blocks=2,
                   num_heads=2,
                   num_hidden=16,
                   num_patches=16)


    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imread uses bgr format by default
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))

    # label_index = np.argmax(model(img))
    # print(label_index)
    # label = tr_loader[label_index]
    # print(label)

    nb_epoch = 5
    eps = 10 / 255
    beta = 10
    uap, losses = uap_sgd(model, tr_loader, nb_epoch, eps, beta)

    # visualize UAP
    plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))    
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.savefig("uap_image.png", format='png', bbox_inches='tight')
    
    uap = uap.reshape((64, 64, 3))
    uap_img = img + np.array(uap*255)

    # visualize side by side
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    

    plt.subplot(1, 2, 2)
    plt.imshow((uap_img).astype(np.uint8), interpolation='none')

    # plt.show()
    # plot loss
    # plt.subplot(2, 2, 2)
    # plt.plot(losses)

    plt.axis('off')  # Turn off axes for a cleaner image
    plt.savefig("uap_sidebyside.png", format='png', bbox_inches='tight')
    plt.show()