import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

import uproot
import awkward as ak

from sklearn.covariance import log_likelihood
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

class UnfoldingNetwork(nn.Module):
    def __init__(self, input_dim:int = 1, hidden_dim:int = 32, num_layers:int = 5, num_classes:int = 1):
        super(UnfoldingNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()

        layers = []

        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())

        self.hidden_fc = nn.Sequential(*layers)

        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        log_r = self.log_ratio(x)
        logit = self.sigmoid(log_r)
        return logit

    def log_ratio(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        log_r = self.fc2(out)
        return log_r

    @torch.no_grad()
    def event_weights(self, x):
        log_r = self.log_ratio(x)
        return torch.exp(log_r)

class UnfoldingDataset(Dataset):
    def __init__(self, X, W, Y):
        super(UnfoldingDataset, self).__init__()

        self.X = X
        self.W = W
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.W[idx], self.Y[idx]


class UnfoldingLoss(nn.Module):
    def __init__(self):
        super(UnfoldingLoss, self).__init__()

        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, logits, targets, weights):
        loss = self.bce_loss(logits, targets)
        weighted_loss = loss* weights
        return torch.mean(weighted_loss)


class UnfoldingTrainner:
    def __init__(self, train_dataloader, val_dataloader, model, criterion, optimizer, max_epoch=1000, patience=10, device=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.patience = patience
        self.device = device

        print("===================== Unfolding Network =====================")
        print(self.model)
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"total trainable params: {total_trainable_params}")

        self.best_state = self.model.state_dict()
        self.best_epoch = None
        self.best_val_loss = None
        self.best_auc = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)

    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self):
        self.model.train()
        for batch, (inputs, weights, labels) in enumerate(self.train_dataloader):

            inputs, weights, labels = inputs.double().to(self.device, non_blocking=True), weights.double().to(self.device, non_blocking=True), labels.double().to(self.device, non_blocking=True)

            logit = self.model(inputs)

            loss = self.criterion(logit, labels, weights)

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch + 1) * len(inputs)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>5f}]", end="")

    def eval_step(self, data_loader):
        num_iterations = len(data_loader)
        self.model.eval()
        loss, auc = 0, 0
        with torch.no_grad():
            for batch, (inputs, weights, labels) in enumerate(data_loader):

                inputs, weights, labels = inputs.double().to(self.device, non_blocking=True), weights.double().to(self.device, non_blocking=True), labels.double().to(self.device, non_blocking=True)

                logit = self.model(inputs)
                loss += self.criterion(logit, labels, weights)

                auc += roc_auc_score(labels.cpu().numpy().reshape(-1), logit.cpu().numpy().reshape(-1), sample_weight=weights.cpu().numpy().reshape(-1))

        return loss/num_iterations, auc/num_iterations

    def fit(self, n_epoch=None):
        max_epoch = (self.epoch + n_epoch + 1) if n_epoch else self.max_epoch

        for epoch in range(self.epoch + 1, max_epoch):
            self.epoch = epoch

            # train
            self.train_step()

            # evaluate loss for training set
            train_loss, train_auc = self.eval_step(self.train_dataloader)

            # evaluate loss for validation set
            val_loss, val_auc = self.eval_step(self.val_dataloader)

            print("\r" + " " * (50), end="")
            print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>5f} Train_auc: {train_auc:>5f}] [Val_loss: {val_loss:>5f} Val_auc: {val_auc:>5f}]")

            if self.best_val_loss == None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_auc = val_auc
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                print(f"[Best_val_loss: {self.best_val_loss:>5f}, Best_ROC_AUC: {self.best_auc:>5f}]")
                self.model.load_state_dict(self.best_state)
                break



def nll_fit(weight, pol, phi_pol, phi):

    sin_phi = np.sin(phi_pol - phi)
    pol_sin_phi = pol* sin_phi
    pol_sin_phi2 = pol_sin_phi* pol_sin_phi
    numerator = weight* pol_sin_phi
    denominator = weight* pol_sin_phi2

    AN = np.sum(numerator)/np.sum(denominator)

    denominator2 = 1. + AN* pol_sin_phi
    denominator22 = denominator2* denominator2

    numerator3 = weight* pol_sin_phi/ denominator2
    numerator32 = numerator3* numerator3

    denominator3 = weight* pol_sin_phi2/denominator22
    sum_denominator3 = np.sum(denominator3)
    sum_denominator32 = sum_denominator3* sum_denominator3

    sum_numerator32 = np.sum(numerator32)

    error = np.sqrt(sum_numerator32/sum_denominator32)

    print(f"[===> AN = {AN:.3f} +/- {error:.3f} (0.1)]")

    return AN, error
