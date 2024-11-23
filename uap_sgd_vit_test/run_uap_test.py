from model.vit import ViT
from train_common import *
from utils import config
from attacks import uap_sgd
from dataset import get_train_val_test_loaders
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from torchvision.models import vit_b_16, ViT_B_16_Weights
import argparse
import cv2
from imagenet_dataset import ImageNet100Dataset

import time

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', default='test_img.png', type=str)
    parser.add_argument('--pert_weights', default='', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Example transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (e.g., for models like ResNet)
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                            std=[0.229, 0.224, 0.225])
    ])

    # Paths
    root_dir = "data/Imagenet100"
    labels_file = os.path.join(root_dir, "Labels.json")

    # Train dataset and dataloader
    train_dataset = ImageNet100Dataset(root_dir=root_dir, subset="train.X1", labels_file=labels_file, transform=transform)
    tr_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    # Validation dataset and dataloader
    # val_dataset = ImageNet100Dataset(root_dir=root_dir, subset="val.X", labels_file=labels_file, transform=transform)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imread uses bgr format by default
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Calculate UAP
    if os.path.isfile(args.pert_weights) == 0:
        print("Computing Perturbation weights...")
        # model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))
        nb_epoch = 5
        eps = 10 / 255
        beta = 10

        # start_time = time.time()
    
        uap, losses = uap_sgd(model, tr_loader, nb_epoch, eps, beta)

        # print("Finished training, took ", time.time()-start_time, "seconds to run.")

        # visualize UAP
        plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))    
        plt.axis('off')  # Turn off axes for a cleaner image
        plt.savefig("uap_image.png", format='png', bbox_inches='tight')

        np.save(os.path.join('data', 'imagenet_data.npy'), uap)
    else:
        print("Found saved perturbation weights, loading...")
        uap = np.load(args.pert_weights)

    transformed_uap = np.array(uap.reshape((224, 224, 3))) * 255
    uap_img = img + np.array(transformed_uap)

    # Find labels for both images
    transformed_image = transforms.ToTensor()(img).reshape(1, 3, 224, 224)
    img_prediction = model(transformed_image).squeeze(0).softmax(0)
    img_class_id = img_prediction.argmax().item()
    img_score = img_prediction[img_class_id].item()
    img_category_name = weights.meta["categories"][img_class_id]
    print(f"ORIG: {img_category_name}: {100 * img_score:.1f}%")

    uap_tensor = transforms.ToTensor()(uap_img).reshape(1, 3, 224, 224)
    uap_prediction = model(uap_tensor).squeeze(0).softmax(0)
    uap_class_id = uap_prediction.argmax().item()
    uap_score = uap_prediction[uap_class_id].item()
    uap_category_name = weights.meta["categories"][uap_class_id]
    print(f"UAP: {uap_category_name}: {100 * uap_score:.1f}%")




    # visualize side by side
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title(img_category_name)
    plt.imshow(img)
    plt.axis('off')  # Turn off axes for a cleaner image
    
    plt.subplot(1, 2, 2)
    plt.title(uap_category_name)
    plt.imshow((uap_img).astype(np.uint8), interpolation='none')
    plt.axis('off')  # Turn off axes for a cleaner image

    # plt.show()
    # plot loss
    # plt.subplot(2, 2, 2)
    # plt.plot(losses)

    plt.savefig("uap_sidebyside.png", format='png', bbox_inches='tight')
    plt.show()