import numpy as np
import scipy.stats as stats
import tensorly as tl


def gen_cp_components(input_modes, output_modes, cp_rank):
    u_components = [stats.ortho_group.rvs(i)[:, :cp_rank] for i in input_modes]
    v_components = [stats.ortho_group.rvs(i)[:, :cp_rank] for i in output_modes]
    return u_components + v_components


class TOTReg:
    def __init__(self, rank=2):
        self.rank = rank
    
    def fit(self, X, Y, iter_max=1000, eps=1e-8):
        N = len(X)
        assert N == len(Y), 'input and output lengths must be equal!'
        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        L = len(input_dims)
        M = len(output_dims)

        self.theta = gen_cp_components(input_dims, output_dims, cp_rank=self.rank)
        B_old = tl.cp_to_tensor((np.ones(self.rank), self.theta))

        i = 0
        for i in range(iter_max):
            for l in range(L):
                theta_ = self.theta.copy()
                del theta_[l]
                C = []
                contract_idx = [ll+1 for ll in range(L)]
                del contract_idx[l]
                for r in range(self.rank):
                    theta_r = [u[:, [r]] for u in theta_]
                    Brl = tl.cp_to_tensor((np.ones(1), theta_r))
                    Cr = tl.tenalg.tensordot(X, Brl, modes=[contract_idx, range(L-1)])
                    C.append(tl.unfold(Cr, 1).T)

                C = np.hstack(C)
                theta_new = np.linalg.solve(C.T @ C, C.T @ tl.tensor_to_vec(Y))
                self.theta[l] = tl.vec_to_tensor(theta_new, self.theta[l].shape)    
            
            # V update
            for m in range(M):
                Ym = tl.unfold(Y, m+1)
                theta_ = self.theta.copy()
                del theta_[L+m]
                D = []
                for r in range(self.rank):
                    theta_r = [u[:, [r]] for u in theta_]
                    Brl = tl.cp_to_tensor((np.ones(1), theta_r))
                    dr = tl.tensor_to_vec(tl.tenalg.tensordot(X, Brl, modes=[range(1, L+1), range(L)]))
                    D.append(dr)
                D = np.column_stack(D)
                self.theta[L+m] = np.linalg.solve(D.T @ D, D.T @ Ym.T).T
            
            B_new = tl.cp_to_tensor((np.ones(self.rank), self.theta))
            if (abs(B_old - B_new)).max() < eps:
                break
            else:
                B_old = B_new
                i += 1

    def predict(self, X_new):
        L = X_new.shape[1:]
        B = tl.cp_to_tensor((np.ones(self.rank), self.theta))
        Y_new = tl.tenalg.tensordot(X_new, B, modes=[range(1, L+1), range(L)])
        return Y_new


if __name__ == '__main__':
    N = 15
    input_dims = (3, 3, 2, 2)
    output_dims = (2, 2, 3)
    L = len(input_dims)
    M = len(output_dims)
    X_train = np.random.rand(N, *input_dims)
    B = np.random.rand(*input_dims, *output_dims)
    Y_train = tl.tenalg.tensordot(X_train, B, modes=[range(1, L+1), range(L)])

    model = TOTReg()
    model.fit(X_train, Y_train)
    B_ls = tl.cp_to_tensor((np.ones(model.rank), model.theta))
    print('Estimated Coefficient Error:', np.linalg.norm(B - B_ls))
