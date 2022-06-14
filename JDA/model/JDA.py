from util import * 
import scipy.linalg
import sklearn.metrics

__all__ = ['JDA']


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(
        self, 
        kernel_type: Literal['primal', 'linear', 'rbf'] = 'primal', 
        out_dim: int = 30,  # dimension after transfer 
        lambda_: float = 1.,  # lambda value in equation
        gamma: float = 1.,  # kernel bandwidth for rbf kernel 
        num_epochs: int = 10,
    ):
        self.kernel_type = kernel_type
        self.dim = out_dim
        self.lamb = lambda_
        self.gamma = gamma
        self.num_epochs = num_epochs

    def fit(
        self, 
        feat_S: FloatArray, 
        label_S: IntArray, 
        feat_T: FloatArray, 
        use_tqdm: bool = True, 
    ) -> tuple[FloatArray, FloatArray]:
        """
        [input]
            feat_S: float[N_S x in_dim]
            label_S: int[N_S]
            feat_T: float[N_T x in_dim]
        [output]
            out_S: float[N_S x out_dim]
            out_T: float[N_T x out_dim]
        """
        label_S = label_S.reshape(-1, 1)
        
        list_acc = []
        X = np.hstack((feat_S.T, feat_T.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(feat_S), len(feat_T)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(label_S))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        
        assert self.num_epochs > 0 
        
        for epoch in tqdm(range(self.num_epochs), desc='JDA', disable=not use_tqdm):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = label_S == c
                    e[np.where(tt == True)] = 1 / len(label_S[np.where(label_S == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
                    
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            out_S, out_T = Z[:, :ns].T, Z[:, ns:].T

        return out_S, out_T
