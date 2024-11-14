from model.vit import ViT
from train_common import *
from utils import config
from attacks import uap_sgd
from dataset import get_train_val_test_loaders
from matplotlib import pyplot as plt

if __name__ == "__main__":

    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("vit.batch_size"),
    )

    model = ViT(num_blocks=2,
                   num_heads=2,
                   num_hidden=16,
                   num_patches=16)
    
    model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))

    nb_epoch = 5
    eps = 10 / 255
    beta = 10
    uap, losses = uap_sgd(model, tr_loader, nb_epoch, eps, beta)    

    # visualize UAP
    plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))    
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.savefig("uap_image.png", format='png', bbox_inches='tight')

    # plot loss
    # plt.plot(losses)
    # plt.show()