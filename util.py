import numpy as np
import matplotlib.pyplot as plt

import os
import dill


def compute_snr(x, n):
    sig_power = sum(x**2) / x.shape[0]
    noise_power = sum((x-n)**2) / x.shape[0]
    snr_10 = 10 * np.log10(sig_power / noise_power)
    return snr_10


def tanh(x):
    if x < 10:
        return 1 - 2 / (np.exp(2 * x) + 1)
    else:
        return 1


def find_attractor(model):
    atts = []
    for k in range(model.K):
        atts.append(np.linalg.pinv((np.eye(model.D) - model.dynamics.As[k])).dot(model.dynamics.bs[k]))
    return atts


def compute_original_x(model, x_infer, C_refer, d_refer):
    C_est_inv = np.linalg.pinv(model.emissions.Cs[0]).squeeze()
    M = C_est_inv.dot(C_refer).squeeze()
    M_inv = np.linalg.pinv(M).squeeze()
    n = C_est_inv.dot((d_refer - model.emissions.ds[0])).squeeze()
    x_orig = M_inv.dot((x_infer - n).transpose()).transpose()
    return x_orig


def avg_nested_lists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if lst.shape[0] > maximum:
            maximum = lst.shape[0]
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < lst.shape[0]: # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp, axis=0))
    return output

def latent_space_transform(model, C_refer, d_refer):
    # Project to the original latent space
    C_est_inv = np.linalg.pinv(model.emissions.Cs[0]).squeeze()
    M = C_est_inv.dot(C_refer).squeeze()
    M_inv = np.linalg.pinv(M).squeeze()
    n = C_est_inv.dot((d_refer - model.emissions.ds[0])).squeeze()
    
    A_prime = []
    b_prime = []
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        A_prime.append(M_inv.dot(A).dot(M))
        b_prime.append(M_inv.dot(A).dot(n) + M_inv.dot(b - n))
    
    for k in range(model.K):
        model.dynamics.As[k] = A_prime[k]
        model.dynamics.bs[k] = b_prime[k]
    
    R_prime = model.transitions.Rs.dot(M)
    r_prime = model.transitions.Rs.dot(n) + model.transitions.r
    
    model.transitions.Rs = R_prime
    model.transitions.r = r_prime

    model.emissions.Cs[0] = C_refer
    model.emissions.ds[0] = d_refer


def compare_metrics(score, metric, metric_name, isplot):
    neg_score = (-1 * np.array(score)).tolist()
    corr = np.corrcoef(neg_score, metric)[0, 1]
    print("CC(-score, " + metric_name + ") = %f" % corr)
    if isplot:
        plt.scatter(neg_score, metric)
        plt.xlabel('-score')
        plt.ylabel(metric_name)
        plt.title('correlation coefficient (CC) = %f' % corr)
        plt.show()

    return corr


def clean_unused_models(path):
    if os.path.exists(path + "elbo_queue.npy"):
        model_list = list(np.load(path + "elbo_queue.npy", allow_pickle=True)[:, 1])
        for filename in os.listdir(path):
            if filename[-5:] == ".dill" and filename[:-5] not in model_list:
                os.remove(path + filename)


def save_model(path, model, elbos, q):
    with open(path, 'wb') as f:
        dill.dump(model, f)
        dill.dump(elbos, f)
        dill.dump(q, f)
    

def load_model(path):
    model = None
    q = None
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = dill.load(f)
            elbos = dill.load(f)
            # q = dill.load(f)
        return model, elbos, q
    else:
        raise FileNotFoundError
    