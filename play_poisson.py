# %%
import scipy.io as sio
import numpy as np
import dill
import matplotlib.pyplot as plt
import util, train, evaluate, plotting

## training parameters
predict_len = 10
latent_dim = 2
model_num = 20
pool_size = 10

def split_data(data):
    n = data.shape[0]
    train = data[:int(n * 0.7)]
    dev = data[int(n * 0.7):int(n * 0.8)]
    test = data[int(n * 0.8):]
    return train, dev, test    


 # %%
if __name__ == "__main__":
    path_prefix = "/home/jamie/Code/rSLDS/synth/"
    data = sio.loadmat(path_prefix + "data/synth_poisson/ObsDim10_Q0_01_Rate2.mat")
    Xs = data["Xs"]
    Zs = data["Zs"]
    Ys = data["Ys"].astype(int)
    C_true = data["C"]
    d_true = data["d"].squeeze()

    x_train, x_dev, x_test = split_data(Xs)
    z_train, z_dev, z_test = split_data(Zs)
    y_train, y_dev, y_test = split_data(Ys)

    r_train = np.log(1 + np.exp(z_train))
    r_dev = np.log(1 + np.exp(z_dev))
    r_test = np.log(1 + np.exp(z_test))
    
    
    # plotting.plot_data(x_train, 'x', 'x1', 'x2')
    # plotting.plot_data(z_train, 'z', 'z1', 'z2')
    # plotting.plot_data(r_train, 'r', 'r1', 'r2')
    # plotting.plot_data(y_train, 'y', 'y1', 'y2')
    
    # print("expected MAE = %.3f" % np.mean(np.linalg.norm(y_test - r_test, ord=2, axis=-1)))
    
    
    # %%
    y_train = [y.astype(int) for y in y_train]
    y_dev = [y.astype(int) for y in y_dev]
    y_test = [y.astype(int) for y in y_test]
    
    lds_path = path_prefix + "models/ObsDim10_Q0_01_Rate2_K1/"
    rslds_path = path_prefix + "models/ObsDim10_Q0_01_Rate2_K3/"
    
    elbo_ids_K1 = train.train_models(lds_path, y_train, latent_dim, 1, model_num, 50, pool_size)
    elbo_ids_K3 = train.train_models(rslds_path, y_train, latent_dim, 3, model_num, 150, pool_size)
    
    # %%
    ids_K1, train_elbos_K1, test_elbos_K1, R2s_K1, eR2s_K1, maes_K1 = evaluate.evaluate_best_model(lds_path, elbo_ids_K1[0][1], y_test, r_test, predict_len)
    print("1-mode LDS model on sythetic Poisson dataset:")
    print("Training ELBO = %.3f, Testing ELBO = %.3f, R^2 = %.4f, expected R^2 = %.4f, MAE = %.3f" %
          (train_elbos_K1, test_elbos_K1, R2s_K1[0], eR2s_K1[0], maes_K1[0]))
    
    ids_K3, train_elbos_K3, test_elbos_K3, R2s_K3, eR2s_K3, maes_K3 = evaluate.evaluate_best_model(rslds_path, elbo_ids_K3[0][1], y_test, r_test, predict_len)
    print("3-mode rSLDS model on sythetic Poisson dataset:")
    print("Training ELBO = %.3f, Testing ELBO = %.3f, R^2 = %.4f, expected R^2 = %.4f, MAE = %.3f" %
          (train_elbos_K3, test_elbos_K3, R2s_K3[0], eR2s_K3[0], maes_K3[0]))
    
    # %%
    with open(lds_path + str(elbo_ids_K1[0][1]) + ".dill", 'rb') as f:
        lds = dill.load(f)

    with open(rslds_path + str(elbo_ids_K3[0][1]) + ".dill", 'rb') as f:
        rslds = dill.load(f)
    
    lds_err = 0
    rslds_err = 0
    for i in range(1):
        elbo_test, q_test = lds.approximate_posterior(y_test[i],                      
                                                    num_iters=10,
                                                    continuous_tolerance=1e-12,
                                                    continuous_maxiter=400,
                                                    verbose=0)
        lds_infer = q_test.mean_continuous_states[0]
        lds_infer = util.compute_original_x(lds, lds_infer, C_true, d_true)
        lds_err += np.sum(np.abs(lds_infer - x_test[i]))
        
        elbo_test, q_test = rslds.approximate_posterior(y_test[i],                      
                                                        num_iters=200,
                                                        continuous_tolerance=1e-12,
                                                        continuous_maxiter=400,
                                                        verbose=0)
        rslds_infer = q_test.mean_continuous_states[0]
        rslds_infer = util.compute_original_x(rslds, rslds_infer, C_true, d_true)
        rslds_err += np.sum(np.abs(rslds_infer - x_test[i]))
        plt.figure()
        plt.title("trial %d" % i)
        plt.plot(x_test[i, :, 0], x_test[i, :, 1], label='true', color='black')
        plt.plot(lds_infer[:, 0], lds_infer[:, 1], label='LDS infer')
        plt.plot(rslds_infer[:, 0], rslds_infer[:, 1], label='rSLDS infer')
        plt.xlim(-2, 10)
        plt.ylim(-2, 10)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(-5, 10)
        plt.ylim(-5, 10)
        plt.legend()
        plt.show()
    
    print("LDS inference error (MAE) = %.3f" % lds_err)
    print("rSLDS inference error (MAE) = %.3f" % rslds_err)

    # %%
    tr = 0 # trial id
    index = list(range(10, y_test[0].shape[tr]-predict_len, 30))
    lds_y = np.zeros((len(index), predict_len, y_test[0].shape[1]))
    rslds_y = np.zeros((len(index), predict_len, y_test[0].shape[1]))
    for j, t in enumerate(index):
            elbo_test, q_test = lds.approximate_posterior(y_test[tr][:t, :],                                                       
                                                num_iters=10,
                                                continuous_tolerance=1e-12,
                                                continuous_maxiter=400,
                                                verbose=0)
            x_infer = q_test.mean_continuous_states[0]
            z_infer = lds.most_likely_states(x_infer, y_test[tr][:t, :])
            prefix = [z_infer[:t], x_infer[:t], y_test[tr][:t, :]]
            z_pred, x_pred, _ = lds.sample(predict_len, prefix=prefix, with_noise=False)
            r_pred = lds.emissions.mean(np.matmul(lds.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
                + lds.emissions.ds).squeeze()

            lds_y[j] = r_pred
            
            elbo_test, q_test = rslds.approximate_posterior(y_test[tr][:t, :],                                                       
                                                num_iters=200,
                                                continuous_tolerance=1e-12,
                                                continuous_maxiter=400,
                                                verbose=0)
            x_infer = q_test.mean_continuous_states[0]
            z_infer = rslds.most_likely_states(x_infer, y_test[tr][:t, :])
            prefix = [z_infer[:t], x_infer[:t], y_test[tr][:t, :]]
            z_pred, x_pred, _ = rslds.sample(predict_len, prefix=prefix, with_noise=False)
            r_pred = rslds.emissions.mean(np.matmul(rslds.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
                + rslds.emissions.ds).squeeze()

            rslds_y[j] = r_pred
    
    # %%
    for n in range(y_test[tr].shape[1]):
        plt.figure()
        plt.title("Neuron %d" % n)
        plt.xlabel("Time steps")
        # plt.ylabel("True firing rate")
        plt.ylabel("True spike counts")
        plt.plot(y_test[tr][:, n], 'grey')
        for j, t in enumerate(index):
            plt.plot(range(t, t + predict_len), lds_y[j, :, n], 'r')
            plt.plot(range(t, t + predict_len), rslds_y[j, :, n], 'g')
    
    
    # %%
    util.latent_space_transform(lds, C_true, d_true)
    util.latent_space_transform(rslds, C_true, d_true)
    plotting.plot_most_likely_dynamics(lds, xlim=(-5, 10), ylim=(-5, 10))
    plotting.plot_most_likely_dynamics(rslds, xlim=(-5, 10), ylim=(-5, 10))