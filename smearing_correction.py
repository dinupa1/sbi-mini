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

from methods import UnfoldingNetwork, UnfoldingDataset, UnfoldingLoss, UnfoldingTrainner, nll_fit


plt.style.use([hep.style.ROOT, hep.style.firamath])


# dvc = "cuda" if torch.cuda.is_available() else "cpu"
dvc = "cpu"
print(f"Using {dvc} device")

pi = np.pi
batch_size = 1000
learning_rate = 0.001
num_iterations = 4

tree_sim = uproot.open("outfile.root:tree_sim")
T_sim = tree_sim["T_sim"].array().to_numpy()
R_sim = tree_sim["R_sim"].array().to_numpy()
pol_sim = tree_sim["pol_sim"].array().to_numpy()
phi_pol_sim = tree_sim["phi_pol_sim"].array().to_numpy()
AN_sim = tree_sim["AN_sim"].array().to_numpy()
effi_sim = tree_sim["effi_sim"].array().to_numpy()
weight_sim = tree_sim["weight_sim"].array().to_numpy()
zeros = tree_sim["zeros"].array().to_numpy()

tree_data = uproot.open("outfile.root:tree_data")
T_data = tree_data["T_data"].array().to_numpy()
R_data = tree_data["R_data"].array().to_numpy()
pol_data = tree_data["pol_data"].array().to_numpy()
phi_pol_data = tree_data["phi_pol_data"].array().to_numpy()
AN_data = tree_data["AN_data"].array().to_numpy()
effi_data = tree_data["effi_data"].array().to_numpy()
weight_data = tree_data["weight_data"].array().to_numpy()
ones = tree_data["ones"].array().to_numpy()

print("[===> particle level]")
AN, error = nll_fit(weight_data, pol_data, phi_pol_data, T_data)

print("[===> detector level]")
AN, error = nll_fit(weight_data, pol_data, phi_pol_data, R_data)

bining = np.linspace(-pi, pi, 21)

plt.figure(figsize=(8, 8))

hist, _ = np.histogram(T_sim[phi_pol_sim > 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = \pi/2$")

hist, _ = np.histogram(T_data[phi_pol_data > 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = \pi/2$")

plt.xlabel(r"$\phi$ [rad]")
plt.ylabel("normalized to unity")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("T_data_spin_up.png")
plt.close("all")

plt.figure(figsize=(8, 8))

hist, _ = np.histogram(R_sim[phi_pol_sim > 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = \pi/2$")

hist, _ = np.histogram(R_data[phi_pol_data > 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = \pi/2$")

plt.xlabel(r"$\phi$ [rad]")
plt.ylabel("normalized to unity")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("R_data_spin_up.png")
plt.close("all")

plt.figure(figsize=(8, 8))

hist, _ = np.histogram(T_sim[phi_pol_sim < 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = -\pi/2$")

hist, _ = np.histogram(T_data[phi_pol_data < 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = -\pi/2$")

plt.xlabel(r"$\phi$ [rad]")
plt.ylabel("normalized to unity")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("T_data_spin_down.png")
plt.close("all")

plt.figure(figsize=(8, 8))

hist, _ = np.histogram(R_sim[phi_pol_sim < 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = -\pi/2$")

hist, _ = np.histogram(R_data[phi_pol_data < 0.], bins=bining)
hist = hist/np.sum(hist)
hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = -\pi/2$")

plt.xlabel(r"$\phi$ [rad]")
plt.ylabel("normalized to unity")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("R_data_spin_down.png")
plt.close("all")


plt.figure(figsize=(8, 8))

hist, _, _ = np.histogram2d(T_sim[phi_pol_sim > 0.], R_sim[phi_pol_sim > 0.], bins=[bining, bining])
hist = hist/np.sum(hist)
hep.hist2dplot(hist, xbins=bining, ybins=bining, cmap="Blues")

plt.xlabel(r"$\phi^{particle}$ [rad]")
plt.ylabel(r"$\phi^{dectecor}$ [rad]")
plt.tight_layout()
plt.savefig("phi_smear_sim_spin_up.png")
plt.close("all")

plt.figure(figsize=(8, 8))

hist, _, _ = np.histogram2d(T_sim[phi_pol_sim < 0.], R_sim[phi_pol_sim < 0.], bins=[bining, bining])
hist = hist/np.sum(hist)
hep.hist2dplot(hist, xbins=bining, ybins=bining, cmap="Blues")

plt.xlabel(r"$\phi^{particle}$ [rad]")
plt.ylabel(r"$\phi^{dectecor}$ [rad]")
plt.tight_layout()
plt.savefig("phi_smear_sim_spin_down.png")
plt.close("all")

plt.figure(figsize=(8, 8))

hist, _, _ = np.histogram2d(T_data[phi_pol_data > 0.], R_data[phi_pol_data > 0.], bins=[bining, bining])
hist = hist/np.sum(hist)
hep.hist2dplot(hist, xbins=bining, ybins=bining, cmap="Blues")

plt.xlabel(r"$\phi^{particle}$ [rad]")
plt.ylabel(r"$\phi^{dectecor}$ [rad]")
plt.tight_layout()
plt.savefig("phi_smear_data_spin_up.png")
plt.close("all")

plt.figure(figsize=(8, 8))

hist, _, _ = np.histogram2d(T_data[phi_pol_data < 0.], R_data[phi_pol_data < 0.], bins=[bining, bining])
hist = hist/np.sum(hist)
hep.hist2dplot(hist, xbins=bining, ybins=bining, cmap="Blues")

plt.xlabel(r"$\phi^{particle}$ [rad]")
plt.ylabel(r"$\phi^{dectecor}$ [rad]")
plt.tight_layout()
plt.savefig("phi_smear_data_spin_down.png")
plt.close("all")

R_weight = np.ones(len(R_data))
T_weight = np.ones(len(R_data))

ANs = np.zeros(num_iterations)
errors = np.zeros(num_iterations)

for i in range(num_iterations):

    print(f"[===> Iteration {i+1}]")

    print("[===> detector level]")

    print("[===> spin up]")

    X = np.concatenate([R_sim[phi_pol_sim > 0.], R_data[phi_pol_data > 0.]]).reshape(-1, 1)
    W = np.concatenate([T_weight[phi_pol_sim > 0.]* ones[phi_pol_sim > 0.], ones[phi_pol_data > 0.]]).reshape(-1, 1)
    Y = np.concatenate([zeros[phi_pol_sim > 0.], ones[phi_pol_data > 0.]]).reshape(-1, 1)

    X_train, X_val, W_train, W_val, Y_train, Y_val = train_test_split(X, W, Y, test_size=0.5, shuffle=True)

    ds_train = UnfoldingDataset(X_train, W_train, Y_train)
    ds_val = UnfoldingDataset(X_val, W_val, Y_val)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model_00 = UnfoldingNetwork().double().to(dvc)

    optimizer = optim.Adam(model_00.parameters(), lr=learning_rate)
    criterion = UnfoldingLoss()

    tr = UnfoldingTrainner(train_loader, val_loader, model_00, criterion, optimizer, device=dvc)
    tr.fit()

    model_00.eval()
    ratio = model_00.event_weights(torch.from_numpy(R_sim[phi_pol_sim > 0.]).double().reshape(-1, 1).to(dvc))

    R_weight[phi_pol_sim > 0.] = T_weight[phi_pol_sim > 0.]* ratio.cpu().numpy().ravel()

    print("[===> spin down]")

    X = np.concatenate([R_sim[phi_pol_sim < 0.], R_data[phi_pol_data < 0.]]).reshape(-1, 1)
    W = np.concatenate([T_weight[phi_pol_sim < 0.]* ones[phi_pol_sim < 0.], ones[phi_pol_data < 0.]]).reshape(-1, 1)
    Y = np.concatenate([zeros[phi_pol_sim < 0.], ones[phi_pol_data < 0.]]).reshape(-1, 1)

    X_train, X_val, W_train, W_val, Y_train, Y_val = train_test_split(X, W, Y, test_size=0.5, shuffle=True)

    ds_train = UnfoldingDataset(X_train, W_train, Y_train)
    ds_val = UnfoldingDataset(X_val, W_val, Y_val)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model_01 = UnfoldingNetwork().double().to(dvc)

    optimizer = optim.Adam(model_01.parameters(), lr=learning_rate)
    criterion = UnfoldingLoss()

    tr = UnfoldingTrainner(train_loader, val_loader, model_01, criterion, optimizer, device=dvc)
    tr.fit()

    model_01.eval()
    ratio = model_01.event_weights(torch.from_numpy(R_sim[phi_pol_sim < 0.]).double().reshape(-1, 1).to(dvc))

    R_weight[phi_pol_sim < 0.] = T_weight[phi_pol_sim < 0.]* ratio.cpu().numpy().ravel()

    plt.figure(figsize=(8, 8))

    hist, _ = np.histogram(R_sim[phi_pol_sim > 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(R_data[phi_pol_data > 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(R_sim[phi_pol_sim > 0.], bins=bining, weights=R_weight[phi_pol_sim > 0])
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"weighted $\phi^{sim}, \phi_{pol} = \pi/2$")

    plt.xlabel(r"$\phi$ [rad]")
    plt.ylabel("normalized to unity")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"R_sim_spin_up_{i+1}.png")
    plt.close("all")

    plt.figure(figsize=(8, 8))

    hist, _ = np.histogram(R_sim[phi_pol_sim < 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(R_data[phi_pol_data < 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(R_sim[phi_pol_sim < 0.], bins=bining, weights=R_weight[phi_pol_sim < 0])
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"weighted $\phi^{sim}, \phi_{pol} = \pi/2$")

    plt.xlabel(r"$\phi$ [rad]")
    plt.ylabel("normalized to unity")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"R_sim_spin_down_{i+1}.png")
    plt.close("all")

    print("[===> particle level]")

    print("[===> spin up]")

    X = np.concatenate([T_sim[phi_pol_sim > 0.], T_sim[phi_pol_sim > 0.]]).reshape(-1, 1)
    W = np.concatenate([ones[phi_pol_sim > 0.], R_weight[phi_pol_sim > 0.]* ones[phi_pol_sim > 0.]]).reshape(-1, 1)
    Y = np.concatenate([zeros[phi_pol_sim > 0.], ones[phi_pol_sim > 0.]]).reshape(-1, 1)

    X_train, X_val, W_train, W_val, Y_train, Y_val = train_test_split(X, W, Y, test_size=0.5, shuffle=True)

    ds_train = UnfoldingDataset(X_train, W_train, Y_train)
    ds_val = UnfoldingDataset(X_val, W_val, Y_val)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model_10 = UnfoldingNetwork().double().to(dvc)

    optimizer = optim.Adam(model_10.parameters(), lr=learning_rate)
    criterion = UnfoldingLoss()

    tr = UnfoldingTrainner(train_loader, val_loader, model_10, criterion, optimizer, device=dvc)
    tr.fit()

    model_10.eval()
    ratio = model_10.event_weights(torch.from_numpy(T_sim[phi_pol_sim > 0.]).double().reshape(-1, 1).to(dvc))

    T_weight[phi_pol_sim > 0.] = ratio.cpu().numpy().ravel()

    print("[===> spin down]")

    X = np.concatenate([T_sim[phi_pol_sim < 0.], T_sim[phi_pol_sim < 0.]]).reshape(-1, 1)
    W = np.concatenate([ones[phi_pol_sim < 0.], R_weight[phi_pol_sim < 0.]* ones[phi_pol_sim < 0.]]).reshape(-1, 1)
    Y = np.concatenate([zeros[phi_pol_sim < 0.], ones[phi_pol_sim < 0.]]).reshape(-1, 1)

    X_train, X_val, W_train, W_val, Y_train, Y_val = train_test_split(X, W, Y, test_size=0.5, shuffle=True)

    ds_train = UnfoldingDataset(X_train, W_train, Y_train)
    ds_val = UnfoldingDataset(X_val, W_val, Y_val)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model_11 = UnfoldingNetwork().double().to(dvc)

    optimizer = optim.Adam(model_11.parameters(), lr=learning_rate)
    criterion = UnfoldingLoss()

    tr = UnfoldingTrainner(train_loader, val_loader, model_11, criterion, optimizer, device=dvc)
    tr.fit()

    model_11.eval()
    ratio = model_11.event_weights(torch.from_numpy(T_sim[phi_pol_sim < 0.]).double().reshape(-1, 1).to(dvc))

    T_weight[phi_pol_sim < 0.] = ratio.cpu().numpy().ravel()

    plt.figure(figsize=(8, 8))

    hist, _ = np.histogram(T_sim[phi_pol_sim > 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(T_data[phi_pol_data > 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(T_sim[phi_pol_sim > 0.], bins=bining, weights=T_weight[phi_pol_sim > 0])
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"weighted $\phi^{sim}, \phi_{pol} = \pi/2$")

    plt.xlabel(r"$\phi$ [rad]")
    plt.ylabel("normalized to unity")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"T_sim_spin_up_{i+1}.png")
    plt.close("all")

    plt.figure(figsize=(8, 8))

    hist, _ = np.histogram(T_sim[phi_pol_sim < 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{sim}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(T_data[phi_pol_data < 0.], bins=bining)
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"$\phi^{data}, \phi_{pol} = \pi/2$")

    hist, _ = np.histogram(T_sim[phi_pol_sim < 0.], bins=bining, weights=T_weight[phi_pol_sim < 0])
    hist = hist/np.sum(hist)
    hep.histplot(hist, bins=bining, histtype="step", label=r"weighted $\phi^{sim}, \phi_{pol} = \pi/2$")

    plt.xlabel(r"$\phi$ [rad]")
    plt.ylabel("normalized to unity")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"T_sim_spin_down_{i+1}.png")
    plt.close("all")

    AN, error = nll_fit(T_weight* weight_sim, pol_sim, phi_pol_sim, T_sim)

    ANs[i] = AN
    errors[i] = error


iters = np.arange(1, num_iterations+1)
plt.figure(figsize=(8, 8))
plt.errorbar(iters, ANs, yerr=errors, fmt="ro", capthick=2, label=r"$A_{N}^{fit}$")
plt.plot(iters, 0.1* np.ones(num_iterations), "b--", label="$A_{N}^{true}$")
plt.xlabel("iteration")
plt.ylabel(r"$A_{N}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("AN_with_iterations.png")
plt.close("all")
