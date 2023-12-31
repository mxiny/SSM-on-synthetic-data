import numpy as np
from multiprocessing import get_context
import os
import dill

import util

def compute_approximate_sum(mu):
    if mu < 2:
        appr_term = (2.3465 ** (mu - 0.492293)) * (mu ** 2)
    else: 
        appr_term = (np.exp(mu) + 0.532784) * np.log(mu + 0.525349) * mu
    return np.exp(-mu) * appr_term - mu * np.log(mu + 1e-8)


def make_poisson_rate_prediction(path, model_id, y_test, predict_len=10, interval=30, cache=1):
    '''
        I = # Trials
        T = # Timepoints
        O = # prediction steps ahead
        N = # Neurons
        
        >> y_trues (Dim = N * T, O, N): 
            The ground truth firing rates or spike counts.
        
        >> y_preds (Dim = N * T, O, N):
            The predicted firing rates or spike counts.
            
        >> y_means: (Dim = N):
            The average firing rates of N neurons accoss T timepoints.
    '''

    if cache == 1 and os.path.exists(path + 'prediction_cache.dill'):
        with open(path + 'prediction_cache.dill', 'rb') as f:
            save_y_pred = dill.load(f)
            save_y_true = dill.load(f)
            train_elbos = dill.load(f)
            test_elbos = dill.load(f)
    else:
        if os.path.exists(path + str(model_id) + ".dill"):
            model, train_elbos, q = util.load_model(path + str(model_id) + ".dill")
        else:
            return FileNotFoundError

        if model.K == 1:
            num_iters = 10
        else:
            num_iters = 200
        test_elbos = []
        save_y_pred = []
        save_y_true = []
        for i in range(len(y_test)): 
            elbo_test, _ = model.approximate_posterior(y_test[i],                                                       
                                                            num_iters=num_iters,
                                                            continuous_tolerance=1e-12,
                                                            continuous_maxiter=400,
                                                            verbose=0)
            test_elbos.append(elbo_test[-1])

            y_preds = []
            y_trues = []
            index = list(range(10, y_test[i].shape[0]-predict_len, interval))     
            for t in index:
                elbo_test, q_test = model.approximate_posterior(y_test[i][:t],                                                       
                                                            num_iters=num_iters,
                                                            continuous_tolerance=1e-12,
                                                            continuous_maxiter=400,
                                                            verbose=0)

            
                x_infer = q_test.mean_continuous_states[0]
                z_infer = model.most_likely_states(x_infer, y_test[i][:t])
                prefix = [z_infer, x_infer, y_test[i][:t]]
                z_pred, x_pred, _ = model.sample(predict_len, prefix=prefix, with_noise=False)
                r_pred = model.emissions.mean(np.matmul(model.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
                    + model.emissions.ds).squeeze()

                y_preds.append(r_pred)
                y_trues.append(y_test[i][t:t+predict_len, :])
                
            save_y_pred.append(y_preds)
            save_y_true.append(y_trues)

    if cache == 1:
        with open(path + 'prediction_cache.dill', 'wb') as f:
            dill.dump(save_y_pred, f)
            dill.dump(save_y_true, f)
            dill.dump(train_elbos, f)
            dill.dump(test_elbos, f)            
    
    return save_y_pred, save_y_true, train_elbos, test_elbos


def get_across_trial_evaluation(path, model_id, y_test, cache=True):
    save_y_pred, save_y_true, train_elbos, test_elbos = make_poisson_rate_prediction(path, model_id, y_test, cache=cache)

    y_trues = []
    y_preds = []
    for i in range(len(save_y_pred)):
        for t in range(len(save_y_pred[i])):
            y_preds.append(save_y_pred[i][t])
            y_trues.append(save_y_true[i][t])


    y_trues = np.stack(y_trues, axis=0)
    y_preds = np.stack(y_preds, axis=0)
    mae = np.mean(np.linalg.norm(y_preds - y_trues, ord=2, axis=-1), axis=0)

    y_means = np.mean(y_trues[:, 0, :], axis=0)
    _numerator = 0
    for i in range(y_trues.shape[0]):
        for n in range(y_trues.shape[-1]):
            _numerator += compute_approximate_sum((y_preds[i, 0, n]))
    
    numerator = np.sum(np.sum(y_trues * np.log(y_trues / (y_preds + 1e-8) + 1e-8) - (y_trues - y_preds), axis=-1), axis=0)
    denominator = np.sum(np.sum(y_trues * np.log(y_trues / (y_means + 1e-8)  + 1e-8), axis=-1), axis=0)
    eR2 = 1 - _numerator / (denominator + 1e-8) 
    R2 = 1 - numerator / (denominator + 1e-8) 
    test_elbo = np.mean(test_elbos)
    return model_id, train_elbos[-1], test_elbo, eR2, R2, mae


def get_individual_trial_evaluation(path, model_id, y_test):
    save_y_pred, save_y_true, train_elbos, test_elbos = make_poisson_rate_prediction(path, model_id, y_test)

    R2s = []
    maes = []
    for i in range(len(save_y_pred)):
        y_preds = np.stack(save_y_pred[i], axis=0)
        y_trues = np.stack(save_y_true[i], axis=0)
        y_means = np.mean(y_trues[:, 0, :], axis=0)
        numerator = np.sum(np.sum(y_trues * np.log(y_trues / (y_preds + 1e-8) + 1e-8) - (y_trues - y_preds), axis=-1), axis=0)
        denominator = np.sum(np.sum(y_trues * np.log(y_trues / (y_means + 1e-8)  + 1e-8), axis=-1), axis=0)
        R2 = 1 - numerator / (denominator + 1e-8) 
        
        mae = np.mean(np.linalg.norm(y_preds - y_trues, ord=2, axis=-1), axis=0)
            
        R2s.append(R2)
        maes.append(mae)

    return test_elbos, R2s, maes
    

def select_best_model(path, y_val, model_num, pool_size):
    result_path = path + "all_model_evaluation.npy"
    if not os.path.exists(result_path):
        return_packs = []
        with get_context("spawn").Pool(processes=pool_size) as pool:
            for i in range(model_num):
                return_packs.append(pool.apply_async(get_across_trial_evaluation, args=(path, i, y_val, False)))
            pool.close()
            pool.join()

        # # For testing
        # return_packs = get_across_trial_evaluation(path, y_val, 0, 0)

        results = []
        for res in return_packs:
            results.append(res.get())

        np.save(result_path, np.array(results))
    else:
        results = list(np.load(result_path, allow_pickle=True))

    sorted_results = sorted(results, key=lambda x:x[4][0], reverse=True)
    best_model_id = sorted_results[0][0]
    
    sorted_results_array = np.stack(sorted_results)
    sorted_ids = np.array(sorted_results_array[:, 0])
    train_elbos = np.stack(sorted_results_array[:, 1])
    test_elbos = np.stack(sorted_results_array[:, 2])
    eR2s = np.stack(sorted_results_array[:, 3])
    R2s = np.stack(sorted_results_array[:, 4])
    maes = np.stack(sorted_results_array[:, 5])
    
    return best_model_id
