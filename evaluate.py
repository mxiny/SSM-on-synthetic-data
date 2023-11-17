import numpy as np
from multiprocessing import get_context
import os

import util

def compute_approximate_sum(mu):
    if mu < 2:
        appr_term = (2.3465 ** (mu - 0.492293)) * (mu ** 2)
    else: 
        appr_term = (np.exp(mu) + 0.532784) * np.log(mu + 0.525349) * mu
    return np.exp(-mu) * appr_term - mu * np.log(mu + 1e-8)


def make_poisson_rate_prediction(model, y_test, r_test, predict_len):
    if model.K == 1:
        num_iters = 10
    else:
        num_iters = 200

    '''
        I = # Trials
        T = # Timepoints
        O = # prediction steps ahead
        N = # Neurons
        
        >> y_trues (Dim = N * T * O * N): 
            The ground truth firing rates or spike counts.
        
        >> y_preds (Dim = N * T * O * N):
            The predicted firing rates or spike counts.
            
        >> y_means: (Dim = N):
            The average firing rates of N neurons accoss T timepoints.
    '''
    
    elbo = 0
    index = list(range(10, y_test[0].shape[0]-predict_len, 30))
    y_trues = np.zeros((len(y_test), len(index), predict_len, y_test[0].shape[1]))
    y_preds = np.zeros((len(y_test), len(index), predict_len, y_test[0].shape[1]))
    for i in range(len(y_test)):         
        for j, t in enumerate(index):
            elbo_test, q_test = model.approximate_posterior(y_test[i][:t],                                                       
                                                        num_iters=num_iters,
                                                        continuous_tolerance=1e-16,
                                                        continuous_maxiter=1000,
                                                        verbose=0)
            elbo += elbo_test[-1]
        
            x_infer = q_test.mean_continuous_states[0]
            z_infer = model.most_likely_states(x_infer, y_test[i][:t])
            prefix = [z_infer, x_infer, y_test[i][:t]]
            z_pred, x_pred, _ = model.sample(predict_len, prefix=prefix, with_noise=False)
            r_pred = model.emissions.mean(np.matmul(model.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
                + model.emissions.ds).squeeze()

            y_preds[i, j] = r_pred
            y_trues[i, j] = y_test[i][t:t+predict_len, :]
            
    elbo /= len(y_test)
    mae = np.mean(np.mean(np.linalg.norm(y_preds - y_trues, ord=2, axis=-1), axis=0), axis=0)

    y_means = np.mean(np.mean(np.stack(y_test, axis=0), axis=0), axis=0)
    _numerator = 0
    for i in range(y_trues.shape[0]):
        for t in range(y_trues.shape[1]):
            for n in range(y_trues.shape[-1]):
                _numerator += compute_approximate_sum((y_preds[i, t, 0, n]))
    
    numerator = np.sum(np.sum(np.sum(y_trues * np.log(y_trues / (y_preds + 1e-8) + 1e-8) - (y_trues - y_preds), axis=-1), axis=0), axis=0)
    denominator = np.sum(np.sum(np.sum(y_trues * np.log(y_trues / (y_means + 1e-8)  + 1e-8), axis=-1), axis=0), axis=0)
    eR2 = 1 - _numerator / (denominator + 1e-8) 
    R2 = 1 - numerator / (denominator + 1e-8) 
    return elbo, R2, eR2, mae


def evaluator(path, y_test, r_test, predict_len, id):
    model = None
    if os.path.exists(path + str(id) + ".dill"):
        model, train_elbos, q = util.load_model(path + str(id) + ".dill")
    elif os.path.exists(path + str(id) + ".npz"):
        file = np.load(path + str(id) + ".npz", allow_pickle=True)
        model = file["model"].tolist()
        train_elbos = file["elbos"]

    test_elbo, R2, eR2, mae = make_poisson_rate_prediction(model, y_test, r_test, predict_len)
    return np.array([id, train_elbos[-1], test_elbo, R2, eR2, mae], dtype=object)


def evaluate_models(path, y_test, r_test, model_num, predict_len, pool_size):
    result_path = path + "outcomes.npy"
    if not os.path.exists(result_path):
        return_packs = []
        with get_context("spawn").Pool(processes=pool_size) as pool:
            for i in range(model_num):
                return_packs.append(pool.apply_async(evaluator, args=(path, y_test, r_test, predict_len, str(i))))
            pool.close()
            pool.join()

        # For testing
        # return_packs = evaluator(path, y_test, predict_len, 0)

        results = []
        for res in return_packs:
            results.append(res.get())

        np.save(result_path, np.array(results))
    else:
        results = list(np.load(result_path, allow_pickle=True))

    sorted_results = sorted(results, key=lambda x:x[1], reverse=True)
    sorted_results_array = np.stack(sorted_results)
    sorted_ids = np.array(sorted_results_array[:, 0])
    train_elbos = np.stack(sorted_results_array[:, 1])
    test_elbos = np.stack(sorted_results_array[:, 2])
    R2s = np.stack(sorted_results_array[:, 3])
    eR2s = np.stack(sorted_results_array[:, 4])
    maes = np.stack(sorted_results_array[:, 5])
    return sorted_ids, train_elbos, test_elbos, R2s, eR2s, maes


def evaluate_best_model(path, id, y_test, r_test, predict_len):
    result_path = path + "best_outcome_30_sampled.npy"
    if not os.path.exists(result_path):

        return_pack = evaluator(path, y_test, r_test, predict_len, id)
        np.save(result_path, np.array(return_pack))
    else:
        return_pack = list(np.load(result_path, allow_pickle=True))

    _, train_elbo, test_elbo, R2, eR2, mae = return_pack
    return id, train_elbo, test_elbo, R2, eR2, mae