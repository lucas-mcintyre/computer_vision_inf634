import matplotlib.pyplot as plt
from data import get_dataloaders
from torchvision.io import read_image
import os
from torchvision import transforms


def evaluate_predictions(preds, labels, debug=False):
    """
    Evaluate predictions and labels and return accuracy, precision, recall, and F1 score

    :param preds: predictions on test set
    :param labels: labels on test set
    :param debug: If True, print out all values for easy copy-paste
    :return: accuracy, precision, recall, and F1 score
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(preds)):
        if labels[i] == 1:
            if preds[i] >= 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if preds[i] >= 0.5:
                fp += 1
            else:
                tn += 1

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("True Positive Rate: {:.4f}".format(tpr))
    print("False Positive Rate: {:.4f}".format(fpr))
    print("True Negative Rate: {:.4f}".format(tnr))
    print("False Negative Rate: {:.4f}".format(fnr))
    print()
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("Accuracy: {:.4f}".format(accuracy))
    print("F1: {:.4f}".format(f1))
    if debug:
        print(tpr)
        print(fpr)
        print(tnr)
        print(fnr)
        print(precision)
        print(recall)
        print(accuracy)
        print(f1)
    return tpr, fpr, tnr, fnr


def plot_false_predictions(preds, labels, test_set, device):
    """
    Plot 5 false positive and 5 false negative examples from the test set.

    :param preds: Predictions on test set
    :param labels: Labels on test set
    :param test_set: Dataset object for test set
    :param device: Device to use
    :return: Figure object
    """
    # plot 5 examples for each, false positive and false negative cases
    # initialize lists to store false positive and false negative examples
    false_positives = []
    false_negatives = []

    # loop through predictions and labels
    for i in range(len(preds)):
        if labels[i] == 1:
            if preds[i] < 0.5:
                false_negatives.append(i)
        else:
            if preds[i] >= 0.5:
                false_positives.append(i)

    # plot 5 false positive examples
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        idx = false_positives[i]
        image = test_set[idx][0]
        image = image.to(device)
        image = image.permute(1, 2, 0)
        ax[0, i].imshow(image.cpu().numpy())
        ax[0, i].set_title(f"Predicted: {preds[idx, 0]:.2f} - True Label: 0")
        ax[0, i].axis("off")

    for i in range(5):
        idx = false_negatives[i]
        image = test_set[idx][0]
        image = image.to(device)
        image = image.permute(1, 2, 0)
        ax[1, i].imshow(image.cpu().numpy())
        ax[1, i].set_title(f"Predicted: {preds[idx, 0]:.2f} - True Label: 1")
        ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def plot_augmentation_samples(num_samples, root_dir, metadata_csv):
    """
    Plots a number of samples from the dataset with and without data augmentation.

    :param num_samples: number of samples to plot
    :param root_dir: root directory of dataset
    :param metadata_csv: path to metadata csv
    :return: Figure with samples
    """
    train, val, test = get_dataloaders(batch_size=num_samples, root_dir=root_dir, metadata_csv=metadata_csv,
                                       use_augmentation=True)
    fig, axs = plt.subplots(2, num_samples, figsize=(20, 4))
    (images, labels, ids) = next(iter(train))

    inv_norm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]), ])

    for i in range(num_samples):
        axs[0, i].imshow(inv_norm(images[i]).permute(1, 2, 0) / 256)
        axs[0, i].set_title("Augmented")
        axs[0, i].axis("off")

        axs[1, i].imshow(read_image(os.path.join(root_dir, ids[i])).permute(1, 2, 0) / 256)
        axs[1, i].set_title("Original")
        axs[1, i].axis("off")

    plt.show()
    return fig
