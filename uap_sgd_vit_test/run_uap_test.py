from model.vit import ViT
from train_common import *
from utils import config
from attacks import uap_sgd
from dataset import get_train_val_test_loaders
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
import torchvision.models as m
# from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights, vit_l_16, ViT_L_16_Weights, vit_l_32, ViT_L_32_Weights
import argparse
import cv2
from imagenet_dataset import ImageNet100Dataset, ImageNet100ValidationDataset

import time

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default='vit_b_16', type=str)
    parser.add_argument('-i', '--input_image', default='test_img.png', type=str)
    parser.add_argument('-w', '--pert_weights', default='', type=str)
    parser.add_argument('--validate', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    weights = None
    model = None

    if args.model == 'vit_b_16':
        print("Loading vit_b_16...")
        weights = m.ViT_B_16_Weights.DEFAULT
        model = m.vit_b_16(weights=weights)
    elif args.model == 'vit_b_32':
        print("Loading vit_b_32...")
        weights = m.ViT_B_32_Weights.DEFAULT
        model = m.vit_b_32(weights=weights)
    elif args.model == 'vit_l_16':
        print("Loading vit_l_16...")
        weights = m.ViT_L_16_Weights.DEFAULT
        model = m.vit_l_16(weights=weights)
    elif args.model == 'vit_l_32':
        print("Loading vit_l_32...")
        weights = m.ViT_L_32_Weights.DEFAULT
        model = m.vit_l_32(weights=weights)
    elif args.model == 'resnet18':
        print("Loading resnet18...")
        weights = m.ResNet18_Weights.DEFAULT
        model = m.resnet18(weights=weights)
    elif args.model == 'alexnet':
        print("Loading alexnet...")
        weights = m.AlexNet_Weights.DEFAULT
        model = m.alexnet(weights=weights)
    elif args.model == 'vgg16':
        print("Loading vgg16...")
        weights = m.VGG16_Weights.DEFAULT
        model = m.vgg16(weights=weights)

    # Example transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (e.g., for models like ResNet)
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                            std=[0.229, 0.224, 0.225])
    ])

    validate_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (e.g., for models like ResNet)
        transforms.ToTensor(),          # Convert image to tensor
    ])

    # Paths
    root_dir = "data/Imagenet100"
    labels_file = os.path.join(root_dir, "Labels.json")

    # Train dataset and dataloader
    train_dataset = ImageNet100Dataset(root_dir=root_dir, subset="train.X6", labels_file=labels_file, transform=transform)
    tr_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    # Validation dataset and dataloader
    val_dataset = ImageNet100ValidationDataset(root_dir=root_dir, subset="val.X", labels_file=labels_file)
    # val_dataset = ImageNet100ValidationSet(root_dir="data/ILSVRC2012_img_val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imread uses bgr format by default
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Calculate UAP
    eps = 10 / 255
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

        np.save(args.pert_weights, uap)
    else:
        print("Found saved perturbation weights, loading...")
        uap = np.load(args.pert_weights)

    # Apply UAP to original img
    uap_filter = np.transpose(((uap / eps) + 1) / 2, (1, 2, 0))
    uap_img = img + np.array(uap_filter)

    def extract_label(input_batch):
        prediction = model(input_batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        img_score = prediction[class_id].item()
        img_category_name = weights.meta["categories"][class_id]
        return (img_category_name, img_score)

    # Find labels for both images
    transformed_image = transforms.ToTensor()(img).reshape(1, 3, 224, 224)
    img_category_name, img_score = extract_label(transformed_image)
    print(f"ORIG: {img_category_name}: {100 * img_score:.1f}%")

    transformed_uap = transforms.ToTensor()(uap_img).reshape(1, 3, 224, 224)
    uap_category_name, uap_score = extract_label(transformed_uap)
    print(f"UAP: {uap_category_name}: {100 * uap_score:.1f}%")

    # Calculate fooling rate of validation set
    if args.validate:
        print("Validating...")
        score = 0
        total = len(val_loader)
        for i, (X, _) in enumerate(val_loader):
            print(f"{i}/{total} Done...")

            x_np = X[0].detach().numpy()
            x_img = x_np + np.array(uap_filter)

            X_trans = transforms.ToTensor()(x_np).reshape(1, 3, 224, 224)
            uap_X = transforms.ToTensor()(x_img).reshape(1, 3, 224, 224)

            orig_label, _ = extract_label(X_trans)
            uap_label, _ = extract_label(uap_X)

            if orig_label != uap_label:
                score += 1
            
        fooling_rate = score / total
        print(f"Fooling Rate: {100*fooling_rate:.1f}%")

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