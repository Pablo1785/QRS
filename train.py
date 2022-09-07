import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

from common import *
from dataset import ArrhythmiaDataset

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from torch.utils.tensorboard import SummaryWriter


def run(moving_average_range, use_classes_from_manual_labels, subset_from_manual_labels, include_manual_labels,
    include_raw_signal, include_derivative):

    MOVING_AVERAGE_RANGE = moving_average_range
    USE_CLASSES_FROM_MANUAL_LABELS = use_classes_from_manual_labels
    SUBSET_FROM_MANUAL_LABELS = subset_from_manual_labels
    INCLUDE_MANUAL_LABELS = include_manual_labels
    INCLUDE_RAW_SIGNAL = include_raw_signal
    INCLUDE_DERIVATIVE = include_derivative

    RECORD_DIR_PATH = 'data/mit-bih-arrhythmia-database-1.0.0'
    WINDOW_SIZE = 540

    CLASSES = ['N', 'L', 'R', 'a', 'V', 'J', 'F'] if USE_CLASSES_FROM_MANUAL_LABELS else ['N', 'L', 'R', 'A', 'a', 'V', 'j', 'J', 'E', 'f', 'F', '[', '!', ']', '/', 'x', '|', 'Q']

    batch_size = 256
    n_epoch = 300

    RUN_NAME = ''
    if INCLUDE_RAW_SIGNAL:
        RUN_NAME += 'raw_signal'

    if MOVING_AVERAGE_RANGE:
        if RUN_NAME:
            RUN_NAME += '_and_'
        RUN_NAME += f'moving_average-{MOVING_AVERAGE_RANGE}'

    if INCLUDE_DERIVATIVE:
        if RUN_NAME:
            RUN_NAME += '_and_'
        RUN_NAME += 'derivative'

    if INCLUDE_MANUAL_LABELS:
        if RUN_NAME:
            RUN_NAME += '_and_'
        RUN_NAME += '11_points'

    if RUN_NAME:
        RUN_NAME += '_and_'

        if SUBSET_FROM_MANUAL_LABELS:
            RUN_NAME += 'reduced_dataset'
        else:
            RUN_NAME += 'full_dataset'

    if RUN_NAME:
        RUN_NAME += '_and_'

        if USE_CLASSES_FROM_MANUAL_LABELS:
            RUN_NAME += 'reduced_labels'
        else:
            RUN_NAME += 'all_labels'

    CHECKPOINT_PATH = f'models/{RUN_NAME} - checkpoint.pt'
    ACCURACY_MOVING_AVERAGE_SIZE = 30  # moving average for accuracy to check if performance degraded


    # TODO: S, e - need some preprocessing, dimensions seem to be wrong in one of these
    # TODO: Q - of course, quite confusing, this is the most confused beat in confusion matrices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Randomness seed
    random_seed = 1 # or any of your favorite number
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    dataset = ArrhythmiaDataset(RECORD_DIR_PATH, WINDOW_SIZE, only_include_labels = CLASSES, moving_average_range = MOVING_AVERAGE_RANGE, include_manual_labels = INCLUDE_MANUAL_LABELS, subset_from_manual_labels = SUBSET_FROM_MANUAL_LABELS, include_raw_signal =
    INCLUDE_RAW_SIGNAL, include_derivative = INCLUDE_DERIVATIVE)

    print(dataset.data.shape)
    print(len(dataset.labels))

    labels, counts = torch.unique(dataset.labels_encoded, dim = 0, return_counts = True)

    for label, count in zip(labels, counts):
        print(f'{dataset.get_label_from_tensor(label)}: {count}')

    # Drop some Normal beats to balance classes
    normal_beat_mask = np.array(dataset.labels) == 'N'

    new_labels = []
    for idx, l in enumerate(normal_beat_mask):
        # Leave 10% samples in (currently theres 75k samples, while other popular classes are at about 8k)
        if l and random.uniform(0, 1) < 0.1:
            normal_beat_mask[idx] = False
        if not normal_beat_mask[idx]:
            new_labels.append(dataset.labels[idx])

    new_data = dataset.data[normal_beat_mask == False]
    dataset.data = new_data
    dataset.labels = new_labels
    dataset.encode_labels()

    def show_class_count(dataset: ArrhythmiaDataset):
        print(dataset.data.shape)
        print(len(dataset.labels))
        labels, counts = torch.unique(dataset.labels_encoded, dim = 0, return_counts = True)

        for label, count in zip(labels, counts):
            print(f'{dataset.get_label_from_tensor(label)}: {count}')

    show_class_count(dataset)

    num_channels = 1 if len(dataset.data.shape) == 2 else dataset.data.shape[1]
    def collate_fn(batch):

        # A data tuple has the form:
        # waveform, one-hot-encoded_label

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, label in batch:
            tensors += [waveform]
            targets += [label]

        # Group the list of tensors into a batched tensor
        tensors = torch.stack(tensors)

        if num_channels == 1:
            # Introduce an empty axis as the single channel
            tensors = tensors[:, None, :]
        else:
            tensors = tensors[:, :]
        targets = torch.stack(targets)

        return tensors, targets


    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_dataset, test_dataset = dataset.train_test_split(0.2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print('TRAIN DATASET:')
    show_class_count(train_dataset)

    print('TEST DATASET:')
    show_class_count(test_dataset)
    class M5(nn.Module):
        def __init__(self, n_input=1, n_output=35, stride=1, n_channel=32):
            super().__init__()
            self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=3, stride=stride)
            self.bn1 = nn.BatchNorm1d(n_channel)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
            self.bn2 = nn.BatchNorm1d(n_channel)
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
            self.bn3 = nn.BatchNorm1d(n_channel)
            self.pool3 = nn.MaxPool1d(3)
            self.conv4 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
            self.bn4 = nn.BatchNorm1d(2 * n_channel)
            self.pool4 = nn.MaxPool1d(3)
            self.conv5 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
            self.bn5 = nn.BatchNorm1d(2 * n_channel)
            self.pool5 = nn.MaxPool1d(3)
            self.conv6 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
            self.bn6 = nn.BatchNorm1d(2 * n_channel)
            self.pool6 = nn.MaxPool1d(3)
            self.fc1 = nn.Linear(2 * n_channel, n_channel)
            self.fc2 = nn.Linear(n_channel, n_output)

        def forward(self, x):
            # print(f'CONV1 INPUT SHAPE: {x.shape}')
            x = self.conv1(x)
            # print(f'CONV1 OUTPUT SHAPE: {x.shape}')
            x = F.relu(self.bn1(x))
            # print(f'POOL1 INPUT SHAPE: {x.shape}')
            x = self.pool1(x)
            # print(f'POOL1 OUTPUT SHAPE: {x.shape}')
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            # print(f'POOL2 INPUT SHAPE: {x.shape}')
            x = self.pool2(x)
            # print(f'POOL2 OUTPUT SHAPE: {x.shape}')
            x = self.conv3(x)
            x = F.relu(self.bn3(x))
            # print(f'POOL3 INPUT SHAPE: {x.shape}')
            x = self.pool3(x)
            # print(f'POOL3 OUTPUT SHAPE: {x.shape}')
            x = self.conv4(x)
            # print(f'BATCHNORM4 INPUT SHAPE: {x.shape}')
            x = F.relu(self.bn4(x))
            # print(f'POOL4 INPUT SHAPE: {x.shape}')
            x = self.pool4(x)
            # print(f'POOL4 OUTPUT SHAPE: {x.shape}')
            x = self.conv5(x)
            # print(f'BATCHNORM5 INPUT SHAPE: {x.shape}')
            x = F.relu(self.bn5(x))
            # print(f'POOL5 INPUT SHAPE: {x.shape}')
            x = self.pool5(x)
            # print(f'POOL5 OUTPUT SHAPE: {x.shape}')
            x = self.conv6(x)
            # print(f'BATCHNORM6 INPUT SHAPE: {x.shape}')
            x = F.relu(self.bn6(x))
            # print(f'POOL6 INPUT SHAPE: {x.shape}')
            x = self.pool6(x)
            # print(f'POOL6 OUTPUT SHAPE: {x.shape}')
            x = F.avg_pool1d(x, x.shape[-1])
            x = x.permute(0, 2, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=2)


    model = M5(n_input = num_channels, n_output = len(set(dataset.labels)))
    model.double().to(device)
    print(model)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    n = count_parameters(model)
    print("Number of parameters: %s" % n)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # reduce the learning after 20 epochs by a factor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 7, verbose = True)  # reduce learning after 7 epochs with no improvement
    def train(model, epoch, log_interval, writer: SummaryWriter):
        train_losses = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            # print(f'DATA SHAPE: {data.shape}')
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            output = model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            squeezed_output = output.squeeze()
            loss = F.nll_loss(squeezed_output, target.argmax(dim = 1))

            writer.add_scalar('Train loss', loss.item(), epoch * len(train_loader.dataset) + batch_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training stats
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # update progress bar
            pbar.update(pbar_update)
            # record loss
            train_losses.append(loss.item())
        return train_losses
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score
    )


    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()


    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)


    def test(model, epoch, writer: SummaryWriter):
        model.eval()
        correct = 0
        y_true = []
        y_pred = []
        loss_sum = 0
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)

            output = model(data)

            squeezed_output = output.squeeze()
            loss_sum += F.nll_loss(squeezed_output, target.argmax(dim = 1)).item()

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target.argmax(dim = 1))

            y_true.extend(pred.squeeze().data.cpu().numpy())
            y_pred.extend(target.data.cpu().numpy().argmax(axis = 1))

            # update progress bar
            pbar.update(pbar_update)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        writer.add_scalar('Test accuracy', accuracy, epoch)
        writer.add_scalar('Test precision', precision, epoch)
        writer.add_scalar('Test recall', recall, epoch)
        writer.add_scalar('Test f1', f1, epoch)
        writer.add_scalar('Test average loss', loss_sum / len(test_loader.dataset), epoch)

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in CLASSES],
                             columns = [i for i in CLASSES])
        plt.figure(figsize = (12,7))
        cf_matrix_figure = sn.heatmap(df_cm, annot=True).get_figure()
        writer.add_figure('Test confusion matrix', cf_matrix_figure, epoch)

        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4%})\n")
        return accuracy, precision, recall, f1, loss_sum
    writer = SummaryWriter(log_dir = 'notebooks/runs')

    log_interval = 20

    writer.add_hparams({f'data_shape_{i}': shape for i, shape in enumerate(dataset.data.shape)} | {'data_moving_average_range': MOVING_AVERAGE_RANGE, 'data_window_size': WINDOW_SIZE, 'batch_size': batch_size, 'n_epoch': n_epoch}, {'hparam/fake_accuracy_just_to_have_any_metric': 10}, run_name = RUN_NAME)

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []
    accuracies = []

    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT_PATH)

            train_losses = train(model, epoch, log_interval, writer)
            losses.extend(train_losses)

            accuracy, precision, recall, f1, loss_sum = test(model, epoch, writer)
            accuracies.append(accuracy)
            scheduler.step(loss_sum)

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            # Early stopping
            if len(accuracies) >= ACCURACY_MOVING_AVERAGE_SIZE + 1:
                is_performance_degraded = np.mean(accuracies[-ACCURACY_MOVING_AVERAGE_SIZE - 1:-1]) > np.mean(accuracies[-ACCURACY_MOVING_AVERAGE_SIZE:])
                if is_performance_degraded:
                    # Reload the last non-degraded checkpoint
                    checkpoint = torch.load(CHECKPOINT_PATH)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    break


if __name__ == '__main__':
    experiments = [
        # MIT-BIH
        {
            'moving_average_range': 17,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': False,
            'include_derivative': False,
        },
        {
            'moving_average_range': 17,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': True,
            'include_derivative': False,
        },
        # Annotated part
        {
            'moving_average_range': None,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': True,
            'include_manual_labels': False,
            'include_raw_signal': True,
            'include_derivative': False,
        },
        {
            'moving_average_range': 17,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': True,
            'include_manual_labels': False,
            'include_raw_signal': False,
            'include_derivative': False,
        },
        {
            'moving_average_range': 17,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': True,
            'include_manual_labels': False,
            'include_raw_signal': True,
            'include_derivative': False,
        },
        # 11 points
        {
            'moving_average_range': None,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': True,
            'include_manual_labels': True,
            'include_raw_signal': False,
            'include_derivative': False,
        },
        {
            'moving_average_range': None,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': True,
            'include_manual_labels': True,
            'include_raw_signal': True,
            'include_derivative': False,
        },
        # MIT-BIH with derivative
        {
            'moving_average_range': None,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': False,
            'include_derivative': True,
        },
        {
            'moving_average_range': None,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': True,
            'include_derivative': True,
        },
        {
            'moving_average_range': 17,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': False,
            'include_derivative': True,
        },
        {
            'moving_average_range': 30,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': False,
            'include_derivative': True,
        },
        {
            'moving_average_range': 17,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': True,
            'include_derivative': True,
        },
        {
            'moving_average_range': 30,
            'use_classes_from_manual_labels': True,
            'subset_from_manual_labels': False,
            'include_manual_labels': False,
            'include_raw_signal': True,
            'include_derivative': True,
        },
    ]
    for experiment in experiments:
        run(**experiment)
