from util import * 
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

N = len(X)
num_train = int(N * 0.8)
shuffled_indices = np.random.permutation(N)

train_mask = np.zeros(N, dtype=bool)
eval_mask = np.zeros(N, dtype=bool)
train_mask[shuffled_indices[:num_train]] = True
eval_mask[shuffled_indices[num_train:]] = True

f1_micro, f1_macro = xgb_multiclass_classification(
    feat = X,
    label = y,
    train_mask = train_mask,
    eval_mask = eval_mask, 
)

print(f1_micro, f1_macro)
