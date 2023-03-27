import numpy as np
import scipy.stats as stats
import tensorly as tl
from itertools import product


def gen_basis(n, idx):
    if hasattr(idx, '__iter__'):
        pass
    else:
        idx = (idx,)
    basis = []
    for k in idx:
        e = np.zeros(n)
        e[k] = 1.
        basis.append(e)
    return np.column_stack(basis)

def kron_basis(modes, idx):
    if not hasattr(modes, '__iter__'):
        return gen_basis(modes, idx)
    else:
        e = 1
        for k, n in zip(modes, idx):
            e = np.kron(e, gen_basis(k, n))
        return e
    
def marginal_gen_matrix(modes, margin_mode, r_min, r_max):
    # it generates a marginal generating matrix. see Ohlson et al. (2013) Section 3.2. for more detail.
    # check validity of r_min, r_max
    assert r_min <= r_max
    assert r_min <= modes[margin_mode] and r_max <= modes[margin_mode]
    idx_list = []
    new_modes = list(modes)
    new_modes[margin_mode] = r_max - r_min + 1
    Mr = np.zeros((np.product(new_modes), np.product(modes)))
    for i, k in enumerate(modes):
        if i == margin_mode:
            idx_list.append(np.arange(r_min, r_max+1))
        else:
            idx_list.append(np.arange(k))
    Ir = list(product(*idx_list))
    for i in Ir:
        e1 = np.kron(kron_basis(modes[:margin_mode], i[:margin_mode]), kron_basis(r_max - r_min + 1, i[margin_mode]-r_min))
        e2 = np.kron(e1, kron_basis(modes[margin_mode+1:], i[margin_mode+1:]))
        Mr += np.outer(e2, kron_basis(modes, i))
    return Mr

# MLE-3D from Ohlson et al. (2013)
def mle_3d(modes, samples, eps, init_theta=None, iter_max=300):
    if init_theta == None:
        theta = [np.eye(d_n) for d_n in modes]
    else:
        # check theta size
        for d, U in enumerate(init_theta):
            assert U.shape[0] == modes[d]
        theta = init_theta
    assert len(eps) == len(modes)

    # sample size sufficiency check
    min_size = max([i / np.product([n for n in modes if n!= i]) for i in modes]) + 1
    if len(samples) > min_size:
        pass
    else:
        print('Need %i samples to be sufficient!' %(min_size - len(samples)))

    theta_diff = eps.copy()
    mu = samples.mean(0)

    i = 0
    for i in range(iter_max):
        for d in range(len(modes)):
            theta_fix = theta.copy()
            modes_fix = modes.copy()
            del theta_fix[d]
            del modes_fix[d]
            theta_fix_inv = [np.linalg.inv(U) for U in theta_fix]
            theta_new = [(tl.unfold(samples[i] - mu, d)) @ tl.tenalg.kronecker(theta_fix_inv) @ (tl.unfold(samples[i] - mu, d)).T for i in range(len(samples))]
            theta_new = np.sum(theta_new, 0) / (len(samples) * np.product(modes_fix))
            theta_diff[d] = np.linalg.norm(theta[d] - theta_new, ord=1)
            theta[d] = theta_new
        i += 1
        if theta_diff < eps:
            break
        else:
            pass
    if i == iter_max:
        print('terminated by max_iter')
    else:
        print('terminated by convergence in %i iterations' %i)
    return theta

# use the following function to generate a initial theta to pass to mle_3d
def gen_random_cov(modes):
    cov_matrices = []
    for n_d in modes:
        eigvals = np.sort(np.random.rand(n_d))
        eigvecs = stats.ortho_group.rvs(n_d)
        cov_matrices.append(eigvecs @ np.diag(eigvals) @ eigvecs.T)
    return cov_matrices


if __name__ == '__main__':
    modes = [5, 4, 3]
    K = tl.tenalg.kronecker(gen_random_cov(modes))
    X = stats.multivariate_normal(np.zeros(np.product(modes)), K)

    samples = X.rvs(100).reshape(-1, *modes)
    theta_mle = mle_3d(modes, samples, [1e-8 for _ in range(len(modes))], init_theta=gen_random_cov(modes))
    theta_mm = np.cov(tl.unfold(samples, 0).T, ddof=1)

    print('Moment Matching:', np.linalg.norm(theta_mm - K, ord='fro'))
    print('MLE-3D         :', np.linalg.norm(tl.tenalg.kronecker(theta_mle) - K, ord='fro'))
