import numpy as np
import itertools

class RBM:
    def __init__(self, v_dim, h_dim, lr=10e-2, gibbs_num=10):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.lr = lr
        self.v_bias = np.random.normal(size = (1,self.v_dim))
        self.h_bias = np.random.normal(size =(1,self.h_dim))
        self.W = np.random.normal(size = (self.v_dim, self.h_dim))
        self.gibbs_num = gibbs_num

    
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
            state_v[v0 < 0] = v_0[v0 < 0]
        else: 
            v_k = state_v
            v_k[v0 < 0] = v_0[v0 < 0]
            _, p_hk_v = self.sample_h(v_k)
        
        return v_0, v_k, p_h0_v, p_hk_v
        

    # with the refered formulas, the gradient of each parameter(W, b-->h, a-->v) could be computed
    def gradient_compute(self, v_0, v_k, p_h0_v, p_hk_v):
        dw = np.dot(v_0.T, p_h0_v) - np.dot(v_k.T, p_hk_v)
        dh_bias = np.sum(p_h0_v - p_hk_v)
        dv_bias = np.sum(v_0 - v_k)
        
        self.W += self.lr * dw
        self.v_bias += self.lr * dv_bias
        self.h_bias += self.lr * dh_bias  
    # set the number of iteration and go over all the training data (binary vectors) for one epoch.


train_data = np.loadtxt(r'C:\Users\86185\Desktop\master thesis\data\Bars-and-Stripes-3x3.txt')
rbm = RBM(train_data.shape[1], 16, lr=10e-2, gibbs_num= 10)

losses = [] 

epochs = 1
batch_size = 1
for epoch in range(epochs):
    train_loss = 0
    epoch_loss = []
    s = 0.

    # learn parameters
    for index in range(0, train_data.shape[0]-batch_size, batch_size):
        v0 = train_data[index: index + batch_size]
        vk = train_data[index: index + batch_size]
        _, p_h0_v = rbm.sample_h(v0)
        v0, vk, p_h0_v, p_hk_v = rbm.gibbs_sampling(v0)
        rbm.gradient_compute(v0, vk, p_h0_v, p_hk_v)
        
        train_loss += np.mean(np.abs(v0[v0>=0] - vk[v0>=0])) # acc = 1 - train_loss
        epoch_loss.append(train_loss/s)
        s += 1.

    # compute KL divergence and model likelihood
    for index in range(0, train_data.shape[0]-batch_size, batch_size):
        v0 = train_data[index: index + batch_size]
        vk = train_data[index: index + batch_size]
        _, p_h0_v = rbm.sample_h(v0)
        v0, vk, p_h0_v, p_hk_v = rbm.gibbs_sampling(v0)

        data_entropy = []   
        data_dic = {}
        for key in v0[0]:
            data_dic[key] = data_dic.get(key, 0) + 1

        print(v0, data_dic.values())
        
    print('====')
    


    # epoch_loss --> KL (RMB & training) ⬇      likelihood ⬆


    losses.append(epoch_loss[-1])
    #if(epoch % 10 == 0):
    #    print('Epoch:{0:4d} Train Loss:{1:1.4f}'.format(epoch, train_loss/s))

