# %%
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import util
import plotting

num_trial = 250
trial_length = 250
latent_dim = 2

# Fixed mode parameters
att_top = np.array([1, 6])
att_bottom = np.array([6, 1])
As = np.stack([np.eye(latent_dim), 0.5*np.eye(latent_dim), 0.5*np.eye(latent_dim)])
bs = np.stack([np.array([0.05, 0.05]), 0.5*att_top, 0.5*att_bottom])

# Noise
obs_dims = [200, 100, 50, 25, 10]
Qs = [0.01]
Ss = [0, 0.01, 0.02, 0.05, 0.1, 0.2]

# Adjust the average firing rate of poisson data
target_rates = [0.25]

data_path = "data/"

def find_rate_bias(target_rate, Xs, C, d):
    biases = np.arange(-8, 2, 0.05)
    errs = np.zeros(biases.shape[0])
    
    for i, bias in enumerate(biases):
        Zs = []
        d_bias = d + bias
        for x in Xs:
            z = x.dot(C.transpose()) + d_bias.squeeze()
            Zs.append(z)
        Zs = np.stack(Zs)
        ave_rate = np.average(np.log(1 + np.exp(Zs)))
        errs[i] = np.abs(target_rate - ave_rate)
    return biases[np.argmin(errs)]


def generate_dynamics(Q, save):
    dynamic_path = data_path + "saved_dynamics/"
    if not os.path.exists(dynamic_path):
        os.mkdir(dynamic_path)

    file_path = dynamic_path + "Q" + str(Q).replace('.', '_') + ".mat"
    if os.path.exists(file_path):
        mat_file = sio.loadmat(file_path)
        Ms = mat_file["Ms"]
        Xs = mat_file["Xs"]
        print("Loaded existing dynamic ...")
    else:
        print("No existing dynamic found, creating new dynamic ...")
        start_point = np.array([0, 0])
        Ms = []
        Xs = []
        for i in range(num_trial):
            m = np.zeros([trial_length])
            m[0] = 0
            x = np.zeros([trial_length, latent_dim])
            x[0] = start_point
            for j in range(trial_length - 1):
                mode = 0
                # diagonal
                if np.abs(x[j, 1] - x[j, 0]) <= 1:
                    mode = 0
                # top point attractor
                elif x[j, 1] - x[j, 0] > 1: 
                    mode = 1
                # bottom point attractor
                else:
                    mode = 2

                m[j] = mode
                x[j + 1] = As[mode].dot(x[j]) + bs[mode] + np.random.normal(loc=0, scale=np.sqrt(Q), size=latent_dim)
            Ms.append(m)
            Xs.append(x)
        Ms = np.stack(Ms)
        Xs = np.stack(Xs)

        if save:
            dynamic_data = {"Ms": Ms, "Xs": Xs, "Q": Q}
            sio.savemat(file_path, dynamic_data)
            print("Saved new dynamic: " + file_path)

    return Ms, Xs

def generate_observations(Xs, obs_type, obs_dim, Q, S, rate, save):
    obs_path = data_path + obs_type + "/"
    if not os.path.exists(obs_path):
        os.mkdir(obs_path)

    # check if C and d already exist
    Cd_path = data_path + "raw_Cd_latentDim" + str(latent_dim) + ".mat"
    if os.path.exists(Cd_path):
        Cd_file = sio.loadmat(Cd_path)
        C_raw = Cd_file["C_raw"]
        d_raw = Cd_file["d_raw"]
        y_dim, x_dim = C_raw.shape
        if y_dim >= obs_dim:
            C_raw = C_raw[:obs_dim, :]
            d_raw = d_raw[:obs_dim]
        else:
            add_dim = obs_dim - y_dim
            C_raw_add = -1 + 2 * np.random.rand(add_dim, latent_dim)
            d_raw_add = -1 + 2 * np.random.rand(add_dim, 1)
            C_raw = np.concatenate([C_raw, C_raw_add])
            d_raw = np.concatenate([d_raw, d_raw_add])
            Cd_data = {"C_raw": C_raw, "d_raw": d_raw}
            sio.savemat(Cd_path, Cd_data)
            print("Dim of existed C and d doesn't match, created new C, d based on existed one")
        print("Loaded existing C and d")
    else:
        C_raw = -1 + 2 * np.random.rand(obs_dim, latent_dim)
        d_raw = -1 + 2 * np.random.rand(obs_dim, 1)
        Cd_data = {"C_raw": C_raw, "d_raw": d_raw}
        sio.savemat(Cd_path, Cd_data)
        print("No existing C and d found, created new C, d")
    
    if obs_type == "gaussian":
        file_path = obs_path + "ObsDim" + str(obs_dim) + "_Q" + str(Q).replace('.', '_') + "_S" + str(S).replace('.', '_') + ".mat"
        C = C_raw
        d = d_raw
    elif obs_type == "poisson":
        file_path = obs_path + "ObsDim" + str(obs_dim) + "_Q" + str(Q).replace('.', '_') + "_Rate" + str(rate).replace('.', '_') + ".mat" 
        C = C_raw
        rate_bias = find_rate_bias(rate, Xs[:20], C_raw, d_raw)
        d = d_raw + rate_bias

    if os.path.exists(file_path):
        print(file_path + " already exists, skip it.")
        return
    Zs = []
    Ys = []
    average_snr = 0
    for i in range(num_trial):
        z = Xs[i].dot(C.transpose()) + d.squeeze()
        if obs_type == "gaussian":
            y = z + np.random.normal(loc=0, scale=np.sqrt(S), size=(trial_length, obs_dim))
        elif obs_type == "poisson":
            # Use softplus as link function
            y = np.random.poisson(lam=np.log(1 + np.exp(z)), size=(trial_length, obs_dim))
        Zs.append(z)
        Ys.append(y)

        # Caculate Signal-to-Noise Ratio (SNR)
        snr = util.compute_snr(z, y)
        average_snr += snr
    
    average_snr /= num_trial
    print("##################################")
    print("y dim = ", obs_dim)
    print("average snr = ", np.average(average_snr))
    print("target rate = ", rate)
    # print("rate bias = ", rate_bias)
    print("average rate = ", np.average(np.log(1 + np.exp(Zs))))
    Zs = np.stack(Zs)
    Ys = np.stack(Ys)

    if save:
        data = {"Xs": Xs, "Zs": Zs, "Ys": Ys, "As": As, "bs": bs, "C": C, "d": d, "Q": Q, "S": S, "SNR": average_snr}
        sio.savemat(file_path, data)
        print("Saved observations: " + file_path)
    
    # import plotting
    # plotting.plot_data(Xs, "Xs", "x1", "x2")
    # plotting.plot_data(np.log(1 + np.exp(Zs)), "Rs", "r1", "r2")
    # plotting.plot_data(Ys, "Ys", "y1", "y2")

    return Zs, Ys

if __name__ == "__main__":
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # ax = plotting.plot_true_dynamics(As, bs)
    # ax.set_title("True latent dynamic")
    # plt.show()
    obs_types = ["poisson"]
    for obs_type in obs_types:
        for Q in Qs:
            # First, generate the latent dynamics x
            Zs, Xs = generate_dynamics(Q, save=True)
            # Second, generate the observations y
            if obs_type == "gaussian":
                for obs_dim in obs_dims:
                    for S in Ss:
                        generate_observations(Xs, obs_type, floor(obs_dim / 10), Q, S, 0, save=True)
            elif obs_type == "poisson":
                for obs_dim in obs_dims:
                    for rate in target_rates:
                        generate_observations(Xs, obs_type, obs_dim, Q, 0, rate, save=True)

        print("Finished creating " + obs_type + " dataset.")

# %%
