import numpy as np
from tqdm import tqdm

class RBM:
    def __init__(self,
                 v_dim = 14, h_dim = 27,
                 lr=5e-4, exp_lrd = 0,
                 weight_decay = 0,
                 gibbs_num = 1,
                 epochs = 50000,
                 batch_size = 14,
                 chain_num = 2
                 ):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.lr = lr
        self.exp_lrd = exp_lrd
        self.weight_decay = weight_decay
        self.v_bias = np.random.normal(-0.1, 0.1, size = (1,self.v_dim))
        self.h_bias = np.random.normal(-0.1, 0.1, size =(1,self.h_dim))
        self.W = np.random.normal(size = (self.v_dim, self.h_dim))
        self.gibbs_num = gibbs_num
        self.v_w, self.v_v, self.v_h = 0, 0, 0
        self.momentum = 0.9
        self.epochs = epochs
        self.batch_size = batch_size
        self.allcases = self.get_all_cases(self.v_dim)
        self.chain_num = chain_num
        self.chains = np.float32(np.random.binomial(1, 0.5, (self.chain_num, self.v_dim)))
        self.beta = np.linspace(0.0, 1.0, self.chain_num).reshape(self.chain_num, 1)

    def sample_h(self, v_input):
        p_h_v = 1 / (1 + np.exp(-(np.dot(v_input, self.W) + self.h_bias)))
        state_h = self.state_sample(p_h_v)
        return state_h, p_h_v

    def sample_v(self, h, beta=1):
        p_v_h = 1 / (1 + np.exp(-(np.dot(h, self.W.T) + self.v_bias)))
        state_v = self.state_sample(p_v_h)
        return state_v, p_v_h

    def state_sample(self, p):
        uni = np.random.uniform(0,1, size = (p.shape[0], p.shape[1]))
        condition = np.less(p, uni)
        state_node = np.where(condition, 0, 1)
        return state_node

    def gibbs_sampling(self, v):
        i = 0
        k = self.gibbs_num
        v_0, v_init = v.copy(), v.copy()
        _, p_h0_v = self.sample_h(v_0)

        while i < k:
            state_h, _ = self.sample_h(v_init)
            state_v, _ = self.sample_v(state_h)
            v_init = state_v
            i += 1
        else:
            v_k = state_v
            _, p_hk_v = self.sample_h(v_k)
        return v_0, v_k, p_h0_v, p_hk_v

    def parallel_tempering(self, v):
        chain_num, beta = self.chain_num, self.beta
        samples = np.empty((v.shape[0], self.v_dim))
        for i in range(v.shape[0]):
            x = (np.dot(self.chains, self.W) + self.h_bias) * beta
            p_h_v = np.power(np.exp(-x) + 1, -1)
            hid = self.state_sample(p_h_v)
            for _ in range(self.gibbs_num):
                x = (np.dot(hid, self.W.T) + self.v_bias) * beta
                p_v_h = np.power(np.exp(-x) + 1, -1)
                vis = self.state_sample(p_v_h)

                x = (np.dot(vis, self.W) + self.h_bias) * beta
                p_h_v = np.power(np.exp(-x) + 1, -1)
                hid = self.state_sample(p_h_v)

            self.chains = np.power(np.exp(-((np.dot(hid, self.W.T) + self.v_bias) * beta)) + 1, -1)
            # p_v_h
            self.swap_state(self.chains, hid)
            samples[i, :] = np.copy(self.chains[self.beta.shape[0] - 1, :])

            self.chains = self.state_sample(self.chains)

        v_k = self.state_sample(samples)
        p_hk_v = np.power(np.exp(-(np.dot(v_k, self.W) + self.h_bias)) + 1, -1)
        return samples, v_k, p_hk_v

    def swap_state(self, chain, h):
        # r = exp{(beta_i - beta_i+1) * (E_i - E_i+1)}
        beta = self.beta
        particle = np.arange(self.chain_num)

        energy = - np.einsum("cv,vh,ch->c",chain,self.W,h) - np.einsum("lv,cv->c", self.v_bias,chain) - np.einsum("lh,ch->c", self.h_bias, h)
        energy = energy.reshape(self.chain_num, 1) * beta

        for t in range(0, beta.shape[0] - 1, 2): # beta (4,1)
            r = np.exp((energy[t + 1, 0] - energy[t, 0]) * (beta[t + 1, 0] - beta[t, 0]))
            if r > 1.0: r = 1.0
            if r > np.random.rand():
                chain[[t, t + 1], :] = chain[[t + 1, t], :]
                energy[[t, t + 1], :] = energy[[t + 1, t], :]
                h[[t, t + 1], :] = h[[t + 1, t], :]
                particle[t], particle[t+1] = particle[t+1], particle[t]

        for t in range(1, beta.shape[0] - 1, 2): # beta (4,1)
            r = np.exp((energy[t + 1, 0] - energy[t, 0]) * (beta[t + 1, 0] - beta[t, 0]))
            if r > 1.0: r = 1.0
            if r > np.random.rand():
                chain[[t, t + 1], :] = chain[[t + 1, t], :]
                energy[[t, t + 1], :] = energy[[t + 1, t], :]
                h[[t, t + 1], :] = h[[t + 1, t], :]
                particle[t], particle[t+1] = particle[t+1], particle[t]
        # print(particle)
        # print("*"*20)

    def gradient_compute(self, v_0, v_k, p_h0_v, p_hk_v):
        dw = (np.dot(v_0.T, p_h0_v) - np.dot(v_k.T, p_hk_v)) / self.batch_size
        dh_bias = (np.sum(p_h0_v - p_hk_v, axis = 0)) / self.batch_size
        dv_bias = (np.sum(v_0 - v_k, axis = 0)) / self.batch_size

        self.v_w = self.momentum * self.v_w + (1 - self.momentum) * dw
        self.v_h = self.momentum * self.v_h + (1 - self.momentum) * dh_bias
        self.v_v = self.momentum * self.v_v + (1 - self.momentum) * dv_bias

        self.W += self.lr * self.v_w - self.lr * self.weight_decay * self.W
        self.v_bias += self.lr * self.v_v
        self.h_bias += self.lr * self.v_h

    def get_all_cases(self, v_dim):
        def all_cases(nums, v_dim):
            res = []
            backtracking(nums, v_dim, [], 0, res)
            return res

        def backtracking(nums, v_dim, path, pos, res):
            if len(path) > v_dim:
                return
            if len(path) == v_dim:
                res.append(list(path))
            for i in range(len(nums)):
                path.append(nums[i])
                backtracking(nums, v_dim, path, i + 1, res)
                path.pop()
        return np.array(all_cases([0, 1], self.v_dim))

    def compute_px_with_Z(self, train_data, W, v_bias, h_bias):
        train_data = np.float32(train_data)
        first_part = np.dot(train_data, v_bias.T).reshape(train_data.shape[0], 1)
        second_part = np.sum(np.log(1 + np.exp(np.dot(train_data, W) + h_bias)), axis = 1)
        second_part = second_part.reshape(train_data.shape[0], 1)
        pxz = np.exp(first_part + second_part)
        return pxz.reshape(-1)
        #(14,)

    def compute_Z(self, W, v_bias, h_bias):
        first_part = np.dot(self.allcases, v_bias.T).reshape(len(self.allcases), 1)
        second_part = np.sum(np.log(1 + np.exp(np.dot(self.allcases, W) + h_bias)), axis = 1)
        second_part = second_part.reshape(len(self.allcases), 1)
        Z = np.sum(np.exp(first_part + second_part).reshape(-1))
        return Z

    def exp_decay(self, epoch): #9 * 1e-11
       initial_lrate = self.lr
       k = self.exp_lrd
       lrate = initial_lrate * np.exp(-k * epoch)
       return lrate

    def train(self, train_data):
        idx = [i for i in range(train_data.shape[0])]
        start = [i for i in idx if i%self.batch_size == 0]
        end = []
        for start_idx in start:
            end_idx = start_idx + self.batch_size
            if end_idx < len(idx):
                end.append(end_idx)
            else:
                end.append(len(idx))
        data_num = len(start)

        lowest_KL_epoch = 0
        lowest_KL = float("inf")
        highest_NLL = float("-inf")
        highest_probsum = float("-inf")

        for epoch in tqdm(range(self.epochs)):
        #for epoch in range(self.epochs):
            self.lr = self.exp_decay(epoch)
            for index in range(data_num):
                # positive sampling
                v0 = train_data[start[index]: end[index]]
                _, p_h0_v = self.sample_h(v0)

                # negative sampling
                vk = v0.copy()
                #_, vk, _, p_hk_v = self.gibbs_sampling(v0)
                _, vk, p_hk_v= self.parallel_tempering(v0)
                self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

            if epoch + 1 == self.epochs or (epoch + 1) % 2000 == 0 or epoch == 0:
                logLKH, KL = 0, 0
                Z = self.compute_Z(self.W, self.v_bias, self.h_bias)
                probability_list = self.compute_px_with_Z(train_data, self.W, self.v_bias, self.h_bias)

                for i in range(len(probability_list)):
                    px_with_Z = probability_list[i]
                    N = len(probability_list)
                    log_lkh = np.log(px_with_Z) - np.log(Z)
                    logLKH += log_lkh

                    kl = -np.log(N)/N - np.log(px_with_Z)/N + np.log(Z)/N
                    KL += kl
                KL /= N
                logLKH /= N
                probability_list = [probability_list[i]/Z for i in range(len(probability_list))]
                x = np.sum(probability_list)
                results = 'epoch {}: KL = {:.4f}, logLKH = {:.4f}, prob_sum = {:.4f}, lr = {:.7f}'.format(epoch + 1, KL, logLKH, x, self.lr)
                tqdm.write(results)
                #f.write(results + '\n')

                if KL < lowest_KL:
                    lowest_KL = KL
                    lowest_KL_epoch = epoch
                    highest_NLL = logLKH
                    highest_probsum = x
        record = "KL {} NLL {} prob_sum {}".format(np.round(lowest_KL, 4), np.round(highest_NLL, 4), np.round(highest_probsum, 4))
        #f.write(record + '\n')
        #f.write('\n')c
        #tqdm.write(record)
        tqdm.write(record)
        #f.close()




if __name__ == "__main__":
    train_data = np.loadtxt(r'./data/Bars-and-Stripes-3x3.txt')
                            # './data/Bars-and-Stripes-3x3.txt' (14, 9) 27

    rbm = RBM(v_dim = train_data.shape[1],
                gibbs_num = 1,
                h_dim = 27,
                lr = 3e-3,
                epochs= 350000, # 300000, lr 3e-3 lr_decay 1e-10
                batch_size = 14,
                chain_num = 2,
                weight_decay = 2.5e-5,
                exp_lrd = 1e-10,
                )
    rbm.train(train_data)