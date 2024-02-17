# %%
import scipy.io as sio
import numpy as np
import dill
import sys
import matplotlib.pyplot as plt
import util, train, evaluate, plotting

## training parameters
predict_len = 10
latent_dim = 2
model_num = 20
pool_size = 5

DIM = '100'
RATE = '1'

_, DIM, RATE = sys.argv

def split_data(data):
    n = data.shape[0]
    train = data[:int(n * 0.7)]
    val = data[int(n * 0.7):int(n * 0.8)]
    test = data[int(n * 0.8):]
    return train, val, test


 # %%
if __name__ == "__main__":
    path_prefix = "/home/jamie/Code/rSLDS/synth/"   # use your own project path
    data = sio.loadmat(path_prefix + "data/poisson/ObsDim" + DIM + "_Q0_01_Rate" + RATE + ".mat")
    Xs = data["Xs"]
    Zs = data["Zs"]
    Ys = data["Ys"].astype(int)
    C_true = data["C"]
    d_true = data["d"].squeeze()

    x_train, x_val, x_test = split_data(Xs)
    z_train, z_val, z_test = split_data(Zs)
    y_train, y_val, y_test = split_data(Ys)

    r_train = np.log(1 + np.exp(z_train))
    r_val = np.log(1 + np.exp(z_val))
    r_test = np.log(1 + np.exp(z_test))
    
    
    # plotting.plot_data(x_train, 'x', 'x1', 'x2')
    # plotting.plot_data(z_train, 'z', 'z1', 'z2')
    # plotting.plot_data(r_train, 'r', 'r1', 'r2')
    # plotting.plot_data(y_train, 'y', 'y1', 'y2')
    
    # print("expected MAE = %.3f" % np.mean(np.linalg.norm(y_test - r_test, ord=2, axis=-1)))
    
    
    # %%
    y_train = list(y_train)
    y_val = list(y_val)
    y_test = list(y_test)
    
    lds_path = path_prefix + "models/poisson/ObsDim" + DIM + "_Q0_01_Rate" + RATE + "_K1/"
    rslds_path = path_prefix + "models/poisson/ObsDim" + DIM + "_Q0_01_Rate" + RATE + "_K3/"
    
    # train.train_models(lds_path, y_train, latent_dim, 1, model_num, 50, pool_size)
    # train.train_models(rslds_path, y_train, latent_dim, 3, model_num, 150, pool_size)

    lds_id = evaluate.select_best_model(lds_path, y_val, model_num, pool_size)
    rslds_id = evaluate.select_best_model(rslds_path, y_val, model_num, pool_size)
    
    # %%
    _, each_R2s_K1, each_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, True, x_test, C_true, d_true)
    _, each_R2s_K3, each_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, True, x_test, C_true, d_true)
    

    # %%
    _, train_elbos_K1, test_elbos_K1, eR2s_K1, R2s_K1, maes_K1 = evaluate.get_across_trial_evaluation(lds_path, lds_id, y_test, True, True, x_test, C_true, d_true)
    print("Poisson LDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f | MAE = %.3f" %
          (train_elbos_K1, test_elbos_K1, R2s_K1[0], eR2s_K1[0], np.mean(each_R2s_K1[:, 0]), maes_K1[0]))
    
    _, train_elbos_K3, test_elbos_K3, eR2s_K3, R2s_K3, maes_K3 = evaluate.get_across_trial_evaluation(rslds_path, rslds_id, y_test, True, True, x_test, C_true, d_true)
    print("Poisson 3-rSLDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f | MAE = %.3f" %
          (train_elbos_K3, test_elbos_K3, R2s_K3[0], eR2s_K3[0], np.mean(each_R2s_K3[:, 0]), maes_K3[0]))

    
    # %%
    # plt.plot(range(1, 11), R2s_K1, label='LDS')
    # plt.plot(range(1, 11), R2s_K3, label='rSLDS')
    # plt.ylabel('R^2')
    # plt.xlabel('prediction steps ahead')
    # plt.xticks(range(1, 11))
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # plt.plot(range(1, 11), maes_K1, label='LDS')
    # plt.plot(range(1, 11), maes_K3, label='rSLDS')
    # plt.ylabel('MAE')
    # plt.xlabel('prediction steps ahead')
    # plt.xticks(range(1, 11))
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # %% compute the p value between outcomes from lds and rslds model
    # test_elbos_K1, best_R2s_K1, best_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test)
    # # print(np.mean(np.stack(best_R2s_K1, axis=0), axis=0)[0])
    # test_elbos_K3, best_R2s_K3, best_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test)
    # # print(np.mean(np.stack(best_R2s_K3, axis=0), axis=0)[0])
    
    # # %%
    # plt.hist([x[-1] for x in best_R2s_K1], label='LDS', alpha=0.5)
    # plt.hist([x[-1] for x in best_R2s_K3], label='rSLDS', alpha=0.5)
    # plt.legend()
    # plt.xlabel('MAE')
    # plt.ylabel('trials')
    # plt.show()
    
    with open(lds_path + str(lds_id) + ".dill", 'rb') as f:
        lds = dill.load(f)

    with open(rslds_path + str(rslds_id) + ".dill", 'rb') as f:
        rslds = dill.load(f)
    print("Best LDS score = %.3f" % evaluate.evaluate_inferred_dynamic(lds, C_true, d_true))
    print("Best rSLDS score = %.3f" % evaluate.evaluate_inferred_dynamic(rslds, C_true, d_true))
    

    # %% compute the p value between outcomes from lds and rslds model
    test_elbos_K1, best_R2s_K1, best_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, True, x_test, C_true, d_true)
    # print(np.mean(np.stack(best_R2s_K1, axis=0), axis=0)[0])
    test_elbos_K3, best_R2s_K3, best_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, True, x_test, C_true, d_true)
    # print(np.mean(np.stack(best_R2s_K3, axis=0), axis=0)[0])
    from scipy.stats import wilcoxon
    best_R2s_K1 = [x[0] for x in best_R2s_K1]
    best_R2s_K3 = [x[0] for x in best_R2s_K3]
    best_maes_K1 = [x[0] for x in best_maes_K1]
    best_maes_K3 = [x[0] for x in best_maes_K3]
    stat, p_elbo = wilcoxon(test_elbos_K1, test_elbos_K3)
    stat, p_R2 = wilcoxon(best_R2s_K1, best_R2s_K3)
    stat, p_mae = wilcoxon(best_maes_K1, best_maes_K3)
    
    print (p_elbo, p_R2, p_mae)

    # %% visualize the comparison of estimated latent dynamic between lds and rslds
    # with open(lds_path + str(lds_id) + ".dill", 'rb') as f:
    #     lds = dill.load(f)

    # with open(rslds_path + str(rslds_id) + ".dill", 'rb') as f:
    #     rslds = dill.load(f)
        
    # print(evaluate.evaluate_inferred_dynamic(lds, C_true, d_true))
    # print(evaluate.evaluate_inferred_dynamic(rslds, C_true, d_true))
    
    # # %% plot the estimated latent vector field
    # util.latent_space_transform(lds, C_true, d_true)
    # util.latent_space_transform(rslds, C_true, d_true)
    # plotting.plot_most_likely_dynamics(lds, xlim=(-5, 10), ylim=(-5, 10))
    # plotting.plot_most_likely_dynamics(rslds, xlim=(-5, 10), ylim=(-5, 10))
    
    # %%
    # lds_err = 0
    # rslds_err = 0
    # # frist 1 test trials
    # for i in range(1):
    #     elbo_test, q_test = lds.approximate_posterior(y_test[i],                      
    #                                                 num_iters=10,
    #                                                 continuous_tolerance=1e-12,
    #                                                 continuous_maxiter=400,
    #                                                 verbose=0)
    #     lds_infer = q_test.mean_continuous_states[0]
    #     lds_infer = util.compute_original_x(lds, lds_infer, C_true, d_true)
    #     lds_err += np.sum(np.abs(lds_infer - x_test[i]))
        
    #     elbo_test, q_test = rslds.approximate_posterior(y_test[i],                      
    #                                                     num_iters=200,
    #                                                     continuous_tolerance=1e-12,
    #                                                     continuous_maxiter=400,
    #                                                     verbose=0)
    #     rslds_infer = q_test.mean_continuous_states[0]
    #     rslds_infer = util.compute_original_x(rslds, rslds_infer, C_true, d_true)
    #     rslds_err += np.sum(np.abs(rslds_infer - x_test[i]))
    #     plt.figure()
    #     plt.title("trial %d" % i)
    #     plt.plot(x_test[i, :, 0], x_test[i, :, 1], label='true', color='black')
    #     plt.plot(lds_infer[:, 0], lds_infer[:, 1], label='LDS infer')
    #     plt.plot(rslds_infer[:, 0], rslds_infer[:, 1], label='rSLDS infer')
    #     plt.xlim(-2, 10)
    #     plt.ylim(-2, 10)
    #     plt.xlabel('x1')
    #     plt.ylabel('x2')
    #     plt.xlim(-5, 10)
    #     plt.ylim(-5, 10)
    #     plt.legend()
    #     plt.show()
    
    # print("LDS inference error (MAE) = %.3f" % lds_err)
    # print("rSLDS inference error (MAE) = %.3f" % rslds_err)

    # # %% visualize the comparison of predictions between lds and rslds
    # tr = 0 # trial id
    # index = list(range(10, y_test[0].shape[tr]-predict_len, 30))
    # lds_y = np.zeros((len(index), predict_len, y_test[0].shape[1]))
    # rslds_y = np.zeros((len(index), predict_len, y_test[0].shape[1]))
    # for j, t in enumerate(index):
    #     # LDS
    #     elbo_test, q_test = lds.approximate_posterior(y_test[tr][:t, :],                                                       
    #                                         num_iters=10,
    #                                         continuous_tolerance=1e-12,
    #                                         continuous_maxiter=400,
    #                                         verbose=0)
    #     x_infer = q_test.mean_continuous_states[0]
    #     z_infer = lds.most_likely_states(x_infer, y_test[tr][:t, :])
    #     prefix = [z_infer[:t], x_infer[:t], y_test[tr][:t, :]]
    #     z_pred, x_pred, _ = lds.sample(predict_len, prefix=prefix, with_noise=False)
    #     r_pred = lds.emissions.mean(np.matmul(lds.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
    #         + lds.emissions.ds).squeeze()
    #     lds_y[j] = r_pred
            
    #     # rSLDS
    #     elbo_test, q_test = rslds.approximate_posterior(y_test[tr][:t, :],                                                       
    #                                         num_iters=200,
    #                                         continuous_tolerance=1e-12,
    #                                         continuous_maxiter=400,
    #                                         verbose=0)
    #     x_infer = q_test.mean_continuous_states[0]
    #     z_infer = rslds.most_likely_states(x_infer, y_test[tr][:t, :])
    #     prefix = [z_infer[:t], x_infer[:t], y_test[tr][:t, :]]
    #     z_pred, x_pred, _ = rslds.sample(predict_len, prefix=prefix, with_noise=False)
    #     r_pred = rslds.emissions.mean(np.matmul(rslds.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
    #         + rslds.emissions.ds).squeeze()
    #     rslds_y[j] = r_pred
    
    # # %%
    # for n in range(y_test[tr].shape[1]):
    #     plt.figure()
    #     plt.title("Neuron %d" % n)
    #     plt.xlabel("Time steps")
    #     # plt.ylabel("True firing rate")
    #     plt.ylabel("True spike counts")
    #     plt.plot(y_test[tr][:, n], 'grey')
    #     for j, t in enumerate(index):
    #         plt.plot(range(t, t + predict_len), lds_y[j, :, n], 'r')
    #         plt.plot(range(t, t + predict_len), rslds_y[j, :, n], 'g')
    

    # # %% visualize latent dynamics of all trained models
    # for i in range(12):
    #     with open(rslds_path + str(i) + ".dill", 'rb') as f:
    #         rslds = dill.load(f)
    #     util.latent_space_transform(rslds, C_true, d_true)
    #     plotting.plot_most_likely_dynamics(rslds, xlim=(-5, 10), ylim=(-5, 10))
        
    # %%