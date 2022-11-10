import numpy as np
import itertools
import time
import matplotlib.pyplot as plt



class RBM:
    def __init__(self, v_dim, h_dim, lr=10e-2, gibbs_num=15):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.lr = lr
        self.v_bias = np.random.normal(0, 0.01, size = (1,self.v_dim))
        self.h_bias = np.random.normal(0, 0.01, size =(1,self.h_dim))
        self.W = np.random.normal(size = (self.v_dim, self.h_dim))
        self.gibbs_num = gibbs_num
        self.v_w, self.v_v, self.v_h = 0, 0, 0
        self.momentum = 0.9

    
    # when a input vector V is recieved, p(h=1|v) can be computed
    # using the activition probability, the states of hidden layer could be sampled
    def sample_h(self, v_input):
        p_h_v = 1/(1 + np.exp(-(np.dot(v_input, self.W) + self.h_bias)))
        state_h = self.state_sample(p_h_v)

        return state_h, p_h_v
    
    # in constract, once the states of hidden layer are obtained, p(v=1|h) can be computed
    # with probability, the states of visible layer could be sampled 
    def sample_v(self, h):
        p_v_h = 1/(1 + np.exp(-(np.dot(h, self.W.T) + self.v_bias)))
        state_v = self.state_sample(p_v_h)
        
        return state_v, p_v_h

    def state_sample(self, p):
        state = []
        uni = np.random.uniform(0,1, size=p[0].shape[0])
        for i in range(len(p)):
            condition = np.less(p[i], uni)
            state_node = np.where(condition, 0, 1)
            state.append(state_node)
        return np.array(state).reshape(p.shape[0], p.shape[1])

    # taking use of the visible layer state V(i) and probability p(v|h), k'th V state could be computed 
    # and its corresponding p(h(k)|V).
    def gibbs_sampling(self, v):
        i = 0 
        k = self.gibbs_num
        v_0 = v

        _, p_h0_v = self.sample_h(v_0)

        while bool(i < k):
            state_h, _ = self.sample_h(v)
            state_v, _ = self.sample_v(state_h)
            i += 1
            #state_v[v0 < 0] = v_0[v0 < 0]
        else: 
            v_k = state_v
            #v_k[v0 < 0] = v_0[v0 < 0]
            _, p_hk_v = self.sample_h(v_k)
        
        return v_0, v_k, p_h0_v, p_hk_v
        

    # with the refered formulas, the gradient of each parameter(W, b-->h, a-->v) could be computed
    def gradient_compute(self, v_0, v_k, p_h0_v, p_hk_v):
        dw = np.dot(v_0.T, p_h0_v) - np.dot(v_k.T, p_hk_v)
        dh_bias = np.sum(p_h0_v - p_hk_v)
        dv_bias = np.sum(v_0 - v_k)
        
        self.v_w = self.momentum * self.v_w + (1 - self.momentum) * dw
        self.v_h = self.momentum * self.v_h + (1 - self.momentum) * dh_bias
        self.v_v = self.momentum * self.v_v + (1 - self.momentum) * dv_bias 

        self.W += self.lr * self.v_w
        self.v_bias += self.lr * self.v_v
        self.h_bias += self.lr * self.v_h 
    # set the number of iteration and go over all the training data (binary vectors) for one epoch.


def compute_px_with_Z(train_data, W, v_bias, h_bias): 
    probability = []
    for l in range(len(train_data)):
        train_data_one_piece = train_data[l]
        product_value = 1
        exp_av = np.exp(np.dot(v_bias, train_data_one_piece))
        for i in range(h_bias.shape[1]):
            product_value = product_value * (np.exp(np.dot(W.T[i], train_data_one_piece) + h_bias.T[i]) + 1)
        px_with_Z = exp_av * product_value
        probability.append(px_with_Z[0])
    return probability


def compute_Z(v_dim, W, v_bias, h_bias):
    def dic_build(dic_candidate, dim):
        i= 0
        for item in itertools.product('01',repeat=dim):
            if i not in dic_candidate:
                dic_candidate[i] = item
            i += 1
        return dic_candidate
        
    dic_v = {}  
    dic_v = dic_build(dic_v, v_dim)

    Z = 0
    for l in range(len(dic_v)):
        train_data_one = np.array(dic_v[l]).astype(int)
        exp_av = np.exp(np.dot(v_bias, train_data_one))
        product = 1
        for j in range(h_bias.shape[1]):
            product = product * (np.exp(np.dot(train_data_one.T, W.T[j]) + h_bias.T[j]) + 1)
        total = exp_av * product

        Z += total
    return Z



if __name__ == '__main__':
    train_data = np.loadtxt('./Bars-and-Stripes-4x4.txt')

    visible_node_num = train_data.shape[1]
    hidden_node_num = 57
    lr = 5*10e-3
    gibbs_num = 20

    epochs = 2000
    batch_size = 1

    rbm = RBM(visible_node_num, hidden_node_num, lr, gibbs_num)

    KL_list = []
    log_LKH_list = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for index in range(0, train_data.shape[0]-batch_size, batch_size):
            v0 = train_data[index: index + batch_size]
            vk = train_data[index: index + batch_size]
            _, p_h0_v = rbm.sample_h(v0)
            v0, vk, p_h0_v, p_hk_v = rbm.gibbs_sampling(v0)
            rbm.gradient_compute(v0, vk, p_h0_v, p_hk_v)
        

        Z = compute_Z(train_data.shape[1], rbm.W, rbm.v_bias, rbm.h_bias) 
        probability_list = compute_px_with_Z(train_data, rbm.W, rbm.v_bias, rbm.h_bias)

        logLKH = 0
        KL = 0
        
        
        for i in range(len(probability_list)):
            px_with_Z = probability_list[i]
            N = len(probability_list)
            log_lkh = np.log(px_with_Z) - np.log(Z) #####
            logLKH += log_lkh

            kl = -np.log(N)/N - np.log(px_with_Z)/N + np.log(Z)/N
            KL += kl
        KL /= N
        logLKH /= N
        KL_list.append(KL)
        log_LKH_list.append(logLKH)

        probability_list = [probability_list[i]/Z for i in range(len(probability_list))]
        x = np.sum(probability_list)
        
        epoch_end_time = time.time()
        
        results = 'epoch:{} ==>  KL = {}, logLKH = {}, prob_sum = {:.4f}, time = {:.2f}s'.format(epoch+1, KL, logLKH, x, epoch_end_time-epoch_start_time)
        f=open("a.txt","a")
        f.write(results + '\n')
        f.close()


    kl_time = [i for i in range(len(KL_list))]
    plt.figure(figsize=(10,5))
    plt.plot(kl_time, KL_list)
    plt.savefig('KL_divergence')

    plt.figure(figsize=(10,5))
    plt.plot(kl_time, log_LKH_list)
    plt.savefig('log_Likelihood')

    np.save('W.npy', rbm.W)
    np.save('h_bias.npy', rbm.h_bias)
    np.save('v_bias.npy', rbm.v_bias)

    

