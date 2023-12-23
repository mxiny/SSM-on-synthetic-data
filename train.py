import numpy as np
from multiprocessing import get_context

import os
import ssm
import util

def trainer(y_train, latent_dim, K, path, id, max_iters):
    np.random.seed()
    model_path = path + str(id) + ".dill"
    obs_dim = np.max([tr.shape[-1] for tr in y_train])
    if not os.path.exists(model_path):
        # Define and initialize the model
        model = ssm.SLDS(N=obs_dim, K=K, D=latent_dim, transitions="recurrent_only",
                        dynamics="diagonal_gaussian",
                        emissions="poisson",
                        emission_kwargs=dict(link="softplus"))
        model.initialize(y_train, num_init_iters=20)
        elbos, q = model.fit(y_train, method="laplace_em", num_iters=max_iters, initialize=False)
    else:
        model, elbos, q = util.load_model(model_path)

    util.save_model(model_path, model, elbos, q)


def train_models(path, y_train, latent_dim, K, model_num, max_iters, pool_size):
    if not os.path.exists(path):
        os.mkdir(path)
    
    print("Training begins...")
    return_packs = []
    with get_context("spawn").Pool(processes=pool_size) as pool:
        for i in range(model_num):
            return_packs.append(pool.apply_async(trainer, args=(y_train, latent_dim, K, path, str(i), max_iters)))
        pool.close()
        pool.join()
        
    print("Training completed!")

