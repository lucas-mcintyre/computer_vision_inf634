import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import RGBA2RGB


class ResNet_CAM(nn.Module):
    """
    Network wrapper for CAM visualization.
    This implementation was taken and adapted from
    https://colab.research.google.com/github/ecs-vlc/fmix/blob/master/notebooks/grad_cam.ipynb.
    """

    def __init__(self, net, layer_k):
        super(ResNet_CAM, self).__init__()
        self.resnet = net

        for param in net.parameters():
            param.requires_grad = True

        convs = nn.Sequential(*list(net.children())[:-2])  # remove the avg pool and the fc layer
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(*list(net.children())[-1:])

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.first_part_conv(x)
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view((1, -1))
        x = self.linear(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.first_part_conv(x)


def superimpose_heatmap(heatmap, img, inv_norm):
    """
    Auxiliary function for plotting the heatmap on top of the image.
    This implementation was taken and adapted from
    https://colab.research.google.com/github/ecs-vlc/fmix/blob/master/notebooks/grad_cam.ipynb.
    """
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    superimposed_img = torch.Tensor(cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)) * 0.006 + inv_norm(
        img[0]).permute(1, 2, 0)

    return superimposed_img


def get_grad_cam(net, img, inv_norm):
    """
    Auxiliary function for computing the Grad-CAM.
    This implementation was taken and adapted from
    https://colab.research.google.com/github/ecs-vlc/fmix/blob/master/notebooks/grad_cam.ipynb.
    """
    net.eval()
    pred = net(img)
    pred[:, pred.argmax(dim=1)].backward()
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = net.get_activations(img).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    return torch.Tensor(superimpose_heatmap(heatmap, img, inv_norm).permute(2, 0, 1))


def plot_grad_cam(layer_k, n_imgs, model, test_loader, device):
    """
    Plot Grad-CAM for a given layer and a given number of images.
    This implementation was taken and adapted from
    https://colab.research.google.com/github/ecs-vlc/fmix/blob/master/notebooks/grad_cam.ipynb.
    """
    inv_norm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]), ])
    model.train()
    baseline_cam_net = ResNet_CAM(model, layer_k)
    baseline_cam_net.to("cpu")

    imgs = torch.Tensor(2, n_imgs, 3, 224, 224)
    it = iter(test_loader)
    for i in range(0, n_imgs):
        img, _, _ = next(it)
        img = img[0:1]
        img.to(device)
        imgs[0][i] = inv_norm(img[0])
        imgs[1][i] = get_grad_cam(baseline_cam_net, img, inv_norm)

    torchvision.utils.save_image(imgs.view(-1, 3, 224, 224), "gradcam_at_layer" + str(layer_k) + ".png", nrow=n_imgs,
                                 pad_value=1)


def plot_predictions(preds, labels, test_set, layer_k, n_imgs, model, device):
    """
    Plot the predictions of the model for a given layer and a given number of images, as well as the Grad-CAMs for the
    respective images.
    """
    # plot examples for each case
    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        RGBA2RGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inv_norm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]), ])

    model.train()
    baseline_cam_net = ResNet_CAM(model, layer_k)
    baseline_cam_net.to("cpu")

    # loop through predictions and labels
    for i in range(len(preds)):
        if labels[i] == 1:
            if preds[i] < 0.5:
                false_negatives.append(i)
            else:
                true_positives.append(i)
        else:
            if preds[i] >= 0.5:
                false_positives.append(i)
            else:
                true_negatives.append(i)

    # plot 5 false positive examples
    fig, ax = plt.subplots(8, n_imgs, figsize=(n_imgs * 3, 8 * 2))
    for i in range(n_imgs):
        idx = false_positives[i]
        image = test_set[idx][0]
        image = image.to(device)
        image = image.permute(1, 2, 0)
        ax[0, i].imshow(image.cpu().numpy())
        ax[0, i].set_title(f"False Positive ({preds[idx, 0]:.2f})")
        ax[1, i].imshow(
            get_grad_cam(baseline_cam_net, transform(image.permute(2, 0, 1)).unsqueeze(0),
                         inv_norm).cpu().numpy().transpose(1, 2,
                                                           0))
        ax[1, i].set_title(f"Grad-CAM")

        idx = false_negatives[i]
        image = test_set[idx][0]
        image = image.to(device)
        image = image.permute(1, 2, 0)
        ax[2, i].imshow(image.cpu().numpy())
        ax[2, i].set_title(f"False Negative ({preds[idx, 0]:.2f})")
        ax[3, i].imshow(
            get_grad_cam(baseline_cam_net, transform(image.permute(2, 0, 1)).unsqueeze(0),
                         inv_norm).cpu().numpy().transpose(1, 2,
                                                           0))
        ax[3, i].set_title(f"Grad-CAM")

        idx = true_positives[i]
        image = test_set[idx][0]
        image = image.to(device)
        image = image.permute(1, 2, 0)
        ax[4, i].imshow(image.cpu().numpy())
        ax[4, i].set_title(f"True Positive ({preds[idx, 0]:.2f})")
        ax[5, i].imshow(
            get_grad_cam(baseline_cam_net, transform(image.permute(2, 0, 1)).unsqueeze(0),
                         inv_norm).cpu().numpy().transpose(1, 2,
                                                           0))
        ax[5, i].set_title(f"Grad-CAM")

        idx = true_negatives[i]
        image = test_set[idx][0]
        image = image.to(device)
        image = image.permute(1, 2, 0)
        ax[6, i].imshow(image.cpu().numpy())
        ax[6, i].set_title(f"True Negative ({preds[idx, 0]:.2f})")
        ax[7, i].imshow(
            get_grad_cam(baseline_cam_net, transform(image.permute(2, 0, 1)).unsqueeze(0),
                         inv_norm).cpu().numpy().transpose(1, 2,
                                                           0))
        ax[7, i].set_title(f"Grad-CAM")
        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
        ax[3, i].axis("off")
        ax[4, i].axis("off")
        ax[5, i].axis("off")
        ax[6, i].axis("off")
        ax[7, i].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def plot_grad_cam_for_one_image(model, test_set, id, k_list, device):
    # plot grad cam for one image for different layers
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        RGBA2RGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inv_norm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]), ])

    model.train()
    image = test_set[id][0]
    image = image.to(device)
    image = image.permute(1, 2, 0)
    fig, ax = plt.subplots(1, len(k_list) + 1, figsize=(len(k_list) * 3, 3))
    ax[0].imshow(image.cpu().numpy())
    ax[0].set_title(f"Original Image")
    for i, k in enumerate(k_list):
        baseline_cam_net = ResNet_CAM(model, k)
        baseline_cam_net.to(device)
        ax[i + 1].imshow(
            get_grad_cam(baseline_cam_net, transform(image.permute(2, 0, 1)).unsqueeze(0),
                         inv_norm).cpu().numpy().transpose(1, 2,
                                                           0))
        ax[i + 1].set_title(f"Grad-CAM for layer {k}")
        ax[i + 1].axis("off")
    plt.tight_layout()
    plt.show()
    return fig
