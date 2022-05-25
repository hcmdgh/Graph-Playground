# Graph Neural Network Playground

# 1 Methodology

## 1.1 HAN

## 1.2 GAT

# 2 Experiment

实验结果：
（训练500epoch）

正常情况：
best_val_f1_micro: 0.9426 in epoch 250, best_val_f1_macro: 0.9398 in epoch 250
best_test_f1_micro: 0.9141 in epoch 300, best_test_f1_macro: 0.9147 in epoch 300

去除softmax：
best_val_f1_micro: 0.9327 in epoch 330, best_val_f1_macro: 0.9292 in epoch 150
best_test_f1_micro: 0.9229 in epoch 500, best_test_f1_macro: 0.9236 in epoch 500

去除semantic attention：
best_val_f1_micro: 0.9327 in epoch 220, best_val_f1_macro: 0.9300 in epoch 220
best_test_f1_micro: 0.9244 in epoch 400, best_test_f1_macro: 0.9259 in epoch 400