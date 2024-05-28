import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import util

## parameter for ploting
color_names = ["windows blue", "red", "amber", "faded green", "purple"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

def plot_data(x_train, title, xlabel, ylabel):
    plt.figure(figsize=(5, 5))
    trial_num = x_train.shape[0]
    trial_len = x_train.shape[1]
    plt.title(title)
    for i in range(trial_num):
        plt.plot(x_train[i, :trial_len, 0], x_train[i, :trial_len, 1])
    plt.grid(True, lw=1, ls='--', c='c')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # for i in range(trial_num):
    # ax[1].plot(x_train[i, :trial_len, 0], 'grey', alpha=0.5)
    # ax[1].grid(True, lw=1, ls='--', c='c')
    # ax[1].set_xlabel('time bins')
    # ax[1].set_title('1st dimension')
    
    # for i in range(trial_num):
    # ax[2].plot(x_train[i, :trial_len, 1], 'grey', alpha=0.5)
    # ax[2].grid(True, lw=1, ls='--', c='c')
    # ax[2].set_xlabel('time bins')
    # ax[2].set_title('2nd dimension')
    plt.show()


def plot_synthetic_dynamics(As, bs, xlim=(-5, 10), ylim=(-5, 10), nxpts=20, nypts=20,
alpha=0.8, ax=None, figsize=(6, 6)):
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    z = np.zeros(xy.shape[0])

    for i, p in enumerate(xy):
        # diagonal
        if np.abs(p[1] - p[0]) <= 1:
            z[i] = 0
        # top point attractor
        elif p[1] - p[0] > 1: 
            z[i] = 1
        # bottom point attractor
        else:
            z[i] = 2
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    for k, (A, b) in enumerate(zip(As, bs)):
        dxydt_m = xy.dot(A.T) + b - xy
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                    dxydt_m[zk, 0], dxydt_m[zk, 1],
                    color=colors[k % len(colors)], alpha=alpha)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    return ax


def plot_most_likely_dynamics(model, xlim=(-50, 50), ylim=(-50, 50), nxpts=20, nypts=20,
alpha=0.8, ax=None, figsize=(6, 6)):
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                    dxydt_m[zk, 0], dxydt_m[zk, 1],
                    color=colors[k % len(colors)], alpha=alpha)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    atts = util.find_attractor(model)
    
    R = model.transitions.Rs
    r = model.transitions.r
    x = np.linspace(start=-50, stop=50)

    for i in range(model.K):
        for j in range(i + 1, model.K):
            slope = (R[j, 0] - R[i, 0]) / (R[i, 1] - R[j, 1])
            bias = -(r[i] - r[j]) / (R[i, 1] - R[j, 1])
            y = slope * x + bias 
            # ax.plot(x, y, label='%d & %d' % (i, j))
            ax.plot(x, y)

    for k in range(model.K):
        ax.scatter(atts[k][0], atts[k][1], color=colors[k % len(colors)], marker='*')

    plt.tight_layout()
    return ax


def plot_latent_dynamic_specification(model):
    K = model.K
    print("\n###########################################################")
    print("Rs = ", model.transitions.Rs)
    print("r = ", model.transitions.r)
    print("C = ", model.emissions.Cs[0])
    print("d = ", model.emissions.ds[0])
    print("Q = ", model.emissions.inv_etas)
    for k in range(K):
        print("A #%d = " % k, model.dynamics.As[k])
        print("b #%d = " % k, model.dynamics.bs[k])
        print("S #%d = " % k, model.dynamics.Sigmas[k])
        eigval, eigvec = np.linalg.eig(model.dynamics.As[k])
        print("eigen value of A = ", eigval)
        print("modulus of eigen value = ", abs(eigval))
        print("eigen vector of A= ", eigvec)

        # Plot the eigen value of A in the complex plane
        theta = np.linspace(0, 2 * np.pi, 100)
        cicle = np.exp(theta * 1j)
        plt.figure(figsize=(6, 6))
        plt.xticks(np.arange(-2, 2, 0.4))
        plt.yticks(np.arange(-2, 2, 0.4))
        plt.plot(cicle.real, cicle.imag, '--')
        plt.plot(eigval.real, eigval.imag, '*')
        plt.grid()
        plt.xlabel("real part")
        plt.ylabel("imaginary part")
        plt.title("The eigen value of A in the complex plane")

        # Find attractors
        att = util.find_attractor(model)[k]
        print("attractor = ", att)
        
        
def update_point(num, x, z, ax):
        ax.plot(x[num:num+2, 0], x[num:num+2, 1], color=colors[z[num+1] % len(colors)])


def plot_latent_trajectory_animation(model, C_refer, d_refer, y_test, save_path_prefix):
    # for all trials
    for i in range(len(y_test)):
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(xlim=(-15, 15), ylim=(-15, 15))
        ax = plot_most_likely_dynamics(model, C_refer, d_refer, ax=ax, original=True, xlim=(-15, 15), ylim=(-15, 15))
        # reconstruct the firing rate 
        elbo_test, q_test = model.approximate_posterior(y_test[i], num_iters=25, verbose=0)
        x_infer = q_test.mean_continuous_states[0]
        original_x = util.compute_original_x(model, x_infer, C_refer, d_refer)
        z_infer = model.most_likely_states(x_infer, y_test[i])

        # ax.plot(original_x[:, 0], original_x[:, 1], color="black"))
        anim = animation.FuncAnimation(fig, update_point, fargs=(original_x, z_infer, ax), frames=original_x.shape[0]-2, interval=100)
        writergif = animation.PillowWriter(fps=2) 
        anim.save(save_path_prefix + "%d.gif" %i, writer=writergif)