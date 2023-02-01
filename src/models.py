import torchvision.models as models
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from tqdm import tqdm


def load_model(model_name, neuron_list, dropout, freeze=True):
    """
    Loads model from torchvision.models and replaces the last layer with a linear layer with num_classes outputs

    :param model_name: name of model to load
    :param neuron_list: list of neurons in each layer that will be added to the model
    :param dropout: dropout rate
    :return: model
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == "efficientnet-b0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "efficientnet-b1":
        model = models.efficientnet_b1(pretrained=True)
    elif model_name == "efficientnet-b2":
        model = models.efficientnet_b2(pretrained=True)
    elif model_name == "efficientnet-b3":
        model = models.efficientnet_b3(pretrained=True)
    elif model_name == "efficientnet-b4":
        model = models.efficientnet_b4(pretrained=True)
    elif model_name == "efficientnet-b5":
        model = models.efficientnet_b5(pretrained=True)
    elif model_name == "efficientnet-b6":
        model = models.efficientnet_b6(pretrained=True)
    elif model_name == "efficientnet-b7":
        model = models.efficientnet_b7(pretrained=True)
    else:
        raise Exception("Invalid model name, got {}".format(model_name))

    # freeze model weights
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # replace last layer
    if model_name.startswith("efficientnet"):
        num_ftrs = model.classifier[1].in_features
    else:
        num_ftrs = model.fc.in_features

    # add layers
    layers = []
    for i in range(len(neuron_list)):
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        if i == 0:
            layers.append(torch.nn.Linear(num_ftrs, neuron_list[i]))
        else:
            layers.append(torch.nn.Linear(neuron_list[i - 1], neuron_list[i]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(neuron_list[-1], 1))
    layers.append(torch.nn.Sigmoid())
    if model_name.startswith("efficientnet"):
        model.classifier = torch.nn.Sequential(*layers)
    else:
        model.fc = torch.nn.Sequential(*layers)
    return model


def unfreeze_model(model):
    """
    Unfreezes model weights

    :param model: model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def freeze_model(model):
    """
    Freezes all layers of model except the last module

    :param model: model to freeze
    """
    # freeze all layers except last module
    for name, param in model.named_parameters():
        if not (name.startswith("fc") or name.startswith("classifier")):
            param.requires_grad = False


def perform_training(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
        use_scheduler=True,
        scheduler_patience=5,
        auto_lr_find=False,
        lr_find_start=1e-7,
        lr_find_end=1,
        lr_find_num=100,
        lr_find_step_mode="exp"):
    """
    Performs training and validation of model

    :param model: model to train
    :param train_loader: train data loader
    :param val_loader: validation data loader
    :param criterion: loss function
    :param optimizer: optimizer
    :param num_epochs: number of epochs to train for
    :param device: device to train on
    :param use_scheduler: whether to use learning rate scheduler
    :return: list of training losses, list of validation losses
    """
    # set model to training mode
    model.train()

    if auto_lr_find:
        # initialize lr finder
        lr_finder = LRFinder(model, optimizer, criterion, device=device)

        # custom data loader iterator

        class CustomTrainIter(TrainDataLoaderIter):
            def inputs_labels_from_batch(self, batch_data):
                return batch_data[0], batch_data[1].unsqueeze(1).float()

        class CustomValIter(ValDataLoaderIter):
            def inputs_labels_from_batch(self, batch_data):
                return batch_data[0], batch_data[1].unsqueeze(1).float()

        custom_train_iter = CustomTrainIter(train_loader)
        custom_val_iter = CustomValIter(val_loader)

        lr_finder.range_test(custom_train_iter, custom_val_iter, start_lr=lr_find_start, end_lr=lr_find_end,
                             num_iter=lr_find_num, step_mode=lr_find_step_mode)
        lr_finder.plot()
        lr_finder.reset()

        for g in optimizer.param_groups:
            g['lr'] = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'])]

    # initialize lr scheduler
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience)

    # create lists to store training and validation loss
    train_losses = []
    val_losses = []

    # start training loop
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # initialize running loss
        running_loss = 0.0
        val_running_loss = 0.0

        # training loop
        for images, labels, _ in tqdm(train_loader):
            # move images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            # forward pass
            outputs = model(images)
            # calculate loss
            loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

            # update running loss
            running_loss += loss.item() * images.size(0)

        # validation loop
        model.eval()
        with torch.no_grad():
            for images, labels, _ in val_loader:
                # move images and labels to device
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1)
                labels = labels.float()

                # forward pass
                outputs = model(images)
                # calculate loss
                loss = criterion(outputs, labels)

                # update running loss
                val_running_loss += loss.item() * images.size(0)
        model.train()

        # calculate average loss
        epoch_loss = running_loss / len(train_loader.dataset)
        val_epoch_loss = val_running_loss / len(val_loader.dataset)

        # update learning rate
        if use_scheduler:
            scheduler.step(val_epoch_loss)

        # store loss
        train_losses.append(epoch_loss)
        val_losses.append(val_epoch_loss)

        print("Training Loss: {:.4f} \tValidation Loss: {:.4f}".format(epoch_loss, val_epoch_loss))

    return train_losses, val_losses


def perform_testing(model, test_loader, device):
    """
    Performs testing of model

    :param model: model to test
    :param test_loader: test data loader
    :param device: device to test on
    :return: list of test predictions, list of real labels
    """
    # set model to evaluation mode
    model.eval()

    # initialize lists to store predictions and labels
    preds = []
    label_list = []

    # disable gradient calculation
    with torch.no_grad():
        # loop over test data
        for images, labels, _ in test_loader:
            # move images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            # forward pass
            outputs = model(images)
            # store predictions
            preds.append(outputs.cpu().numpy())
            label_list.append(labels.cpu().numpy())

    # concatenate predictions and labels
    preds = np.concatenate(preds)
    labels = np.concatenate(label_list)

    return preds, labels


def save_model(model, model_name, save_path):
    """
    Saves model to disk

    :param model: model to save
    :param model_name: name of model
    :param save_path: path to save model to
    """
    # save model
    torch.save(model.state_dict(), save_path + model_name + ".pth")

    print("Model saved to {}".format(save_path + model_name + ".pth"))


def load_model_from_saved(model_name, neuron_list, dropout, save_path, model_name_save, device):
    """
    Loads model from disk (might need to move to device after loading)

    :param model_name: name of model
    :param num_classes: number of classes
    :param save_path: path to load model from
    :return: loaded model
    """
    # create model
    model = load_model(model_name, neuron_list, dropout)

    # load model weights
    model.load_state_dict(torch.load(save_path + model_name_save + ".pth", map_location=device))

    return model
