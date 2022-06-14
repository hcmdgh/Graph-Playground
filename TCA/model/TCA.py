from util import * 
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(
        self, 
        kernel_type: Literal['primal', 'linear', 'rbf'] = 'primal', 
        out_dim: int = 30,  # dimension after transfer
        lambda_: float = 1.,  # lambda value in equation
        gamma: float = 1.,  # kernel bandwidth for rbf kernel
    ):
        self.kernel_type = kernel_type
        self.out_dim = out_dim
        self.lambda_ = lambda_
        self.gamma = gamma

    def fit(
        self, 
        feat_S: FloatArray, 
        feat_T: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        [input]
            feat_S: float[N_S x in_dim]
            feat_T: float[N_T x in_dim]
        [output]
            out_S: float[N_S x out_dim]
            out_T: float[N_T x out_dim]
        """
        
        X = np.hstack((feat_S.T, feat_T.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(feat_S), len(feat_T)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lambda_ * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.out_dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        out_S, out_T = Z[:, :ns].T, Z[:, ns:].T
        
        return out_S, out_T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred
    