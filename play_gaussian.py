# %%
import scipy.io as sio
import numpy as np
import dill
import sys
import matplotlib.pyplot as plt
import util, train, evaluate, plotting
from scipy.stats import wilcoxon

## training parameters
predict_len = 10
latent_dim = 2
model_num = 40
pool_size = 10

DIM = '5'
S = '0_01'

# _, DIM, S = sys.argv

def split_data(data):
    n = data.shape[0]
    train = data[:int(n * 0.7)]
    val = data[int(n * 0.7):int(n * 0.8)]
    test = data[int(n * 0.8):]
    return train, val, test


 # %%
if __name__ == "__main__":
    path_prefix = "/home/jamie/Code/rSLDS/synth/"   # use your own project path
    data = sio.loadmat(path_prefix + "data/gaussian/ObsDim" + DIM + "_Q0_01_S" + S + ".mat")
    Xs = data["Xs"]
    Zs = data["Zs"]
    Ys = data["Ys"]
    C_true = data["C"]
    d_true = data["d"].squeeze()

    x_train, x_val, x_test = split_data(Xs)
    z_train, z_val, z_test = split_data(Zs)
    y_train, y_val, y_test = split_data(Ys)


    # plotting.plot_data(x_train, 'x', 'x1', 'x2')
    # plotting.plot_data(z_train, 'z', 'z1', 'z2')
    # plotting.plot_data(y_train, 'y', 'y1', 'y2')

    y_train = list(y_train)
    y_val = list(y_val)
    y_test = list(y_test)
    
    
    # %%
    lds_path = path_prefix + "models/gaussian/ObsDim" + DIM + "_Q0_01_S" + S + "_K1/"
    rslds_path = path_prefix + "models/gaussian/ObsDim" + DIM + "_Q0_01_S" + S + "_K3/"
    
#     train.train_models(lds_path, y_train, latent_dim, 1, model_num, 50, pool_size)
#     train.train_models(rslds_path, y_train, latent_dim, 3, model_num, 150, pool_size)

    lds_id = evaluate.select_best_model(lds_path, y_val, model_num, pool_size)
    rslds_id = evaluate.select_best_model(rslds_path, y_val, model_num, pool_size)
    
    # %%
    _, each_eR2s_K1, each_R2s_K1, each_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, False, x_test, C_true, d_true)
    _, each_eR2s_K3, each_R2s_K3, each_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, False, x_test, C_true, d_true)
    
    # %%
    _, train_elbos_K1, test_elbos_K1, eR2s_K1, R2s_K1, maes_K1 = evaluate.get_across_trial_evaluation(lds_path, lds_id, y_test, True, False, x_test, C_true, d_true)
    print("Gaussian LDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f | MAE = %.3f" %
          (train_elbos_K1, test_elbos_K1, R2s_K1[0], eR2s_K1[0], np.mean(each_R2s_K1[:, 0]), maes_K1[0]))
    
    _, train_elbos_K3, test_elbos_K3, eR2s_K3, R2s_K3, maes_K3 = evaluate.get_across_trial_evaluation(rslds_path, rslds_id, y_test, True, False, x_test, C_true, d_true)
    print("Gaussian 3-rSLDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f | MAE = %.3f" %
          (train_elbos_K3, test_elbos_K3, R2s_K3[0], eR2s_K3[0], np.mean(each_R2s_K3[:, 0]), maes_K3[0]))


    # %%
    with open(lds_path + str(lds_id) + ".dill", 'rb') as f:
        lds = dill.load(f)

    with open(rslds_path + str(rslds_id) + ".dill", 'rb') as f:
        rslds = dill.load(f)
    print("Best LDS score = %.3f" % evaluate.evaluate_inferred_dynamic(lds, C_true, d_true))
    print("Best rSLDS score = %.3f" % evaluate.evaluate_inferred_dynamic(rslds, C_true, d_true))
    
    # %% compute the p value between outcomes from lds and rslds model
    test_elbos_K1, best_eR2s_K1, best_R2s_K1, best_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, False, x_test, C_true, d_true)
    # print(np.mean(np.stack(best_R2s_K1, axis=0), axis=0)[0])
    test_elbos_K3, best_eR2s_K3, best_R2s_K3, best_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, False, x_test, C_true, d_true)
    # print(np.mean(np.stack(best_R2s_K3, axis=0), axis=0)[0])
    # best_R2s_K1 = [x[0] for x in best_R2s_K1]
    # best_R2s_K3 = [x[0] for x in best_R2s_K3]
    # best_maes_K1 = [x[0] for x in best_maes_K1]
    # best_maes_K3 = [x[0] for x in best_maes_K3]
    # stat, p_elbo = wilcoxon(test_elbos_K1, test_elbos_K3)
    # stat, p_R2 = wilcoxon(best_R2s_K1, best_R2s_K3)
    # stat, p_mae = wilcoxon(best_maes_K1, best_maes_K3)
    
    # print (p_elbo, p_R2, p_mae)
             

    # %%
    _, _, _, _, R2s_lv, maes_lv = evaluate.get_across_trial_evaluation(None, None, y_test)
    
    p_R2 = []
    p_mae = []
    for i in range(predict_len):
        best_K1 = [x[i] for x in best_R2s_K1]
        best_K3 = [x[i] for x in best_R2s_K3]
        stat, p = wilcoxon(best_K1, best_K3)
        p_R2.append(p)
        
        best_K1 = [x[i] for x in best_maes_K1]
        best_K3 = [x[i] for x in best_maes_K3]
        stat, p = wilcoxon(best_K1, best_K3)
        p_mae.append(p)
        
    print("p value of R^2 results: ", p_R2)
    print("p value of MAE results: ", p_mae)
    plt.plot(range(1, 11), R2s_K1, label='LDS')
    plt.plot(range(1, 11), R2s_K3, label='rSLDS')
    plt.plot(range(1, 11), R2s_lv, label='Last value')
    plt.ylabel('R^2')
    plt.xlabel('prediction steps ahead')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.plot(range(1, 11), maes_K1, label='LDS')
    plt.plot(range(1, 11), maes_K3, label='rSLDS')
    plt.plot(range(1, 11), maes_lv, label='Last value')
    plt.ylabel('MAE')
    plt.xlabel('prediction steps ahead')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # %%