import numpy as np

N, D_in, H, D_out = 60, 1000, 100, 10

x = np.random.randn(N, D_in) # (60, 1000)
y = np.random.randn(N, D_out) # (60, 10)

w1 = np.random.randn(D_in, H) # (1000, 100)
w2 = np.random.randn(H, D_out) # (100, 10)

learning_rate = 1e-6 # lr

for iter in range(5000):
    h = x.dot(w1) # (60, 100)
    h_relu = np.maximum(h, 0) # (60, 100)
    y_pred = h_relu.dot(w2) # (60, 10)

    loss = np.square(y_pred - y).sum()
    if iter % 25 == 0:
        print('第{}轮 loss为:{}'.format(iter+1, loss))
    
    y_pred_grad = 2 * (y_pred - y)
    w2_grad = h_relu.T.dot(y_pred_grad)
    h_relu_grad = y_pred_grad.dot(w2.T)
    _h = h_relu_grad.copy()
    _h[h < 0] = 0
    w1_grad = x.T.dot(_h)

    # update paras
    w1 -= learning_rate * w1_grad
    w2 -= learning_rate * w2_grad
