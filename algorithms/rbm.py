import numpy as np
from tqdm import tqdm

class RBM:
    def __init__(self,
                 v_dim, h_dim,
                 opt_type, sampling_type,
                 lr=1e-3, exp_lrd = 0,
                 weight_decay = 0,
                 gibbs_num = 1,
                 epochs = 100000,
                 batch_size = 14,
                 chain_num = 2,
                 output_epoch = 10000
                 ):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.v_bias = np.random.normal(-0.1, 0.1, size = (1,self.v_dim))
        self.h_bias = np.random.normal(-0.1, 0.1, size =(1,self.h_dim))
        self.W = np.random.normal(size = (self.v_dim, self.h_dim))

        self.lr = lr
        self.exp_lrd = exp_lrd
        self.weight_decay = weight_decay
        
        self.sampling_type = sampling_type
        self.gibbs_num = gibbs_num
        if self.sampling_type == "parallel_tempering":
            self.chain_num = chain_num
            self.chains = np.float32(np.random.binomial(1, 0.5, (self.chain_num, self.v_dim)))
            self.beta = np.linspace(0.0, 1.0, self.chain_num).reshape(self.chain_num, 1)

        self.opt_type = opt_type
        self.v_w, self.v_v, self.v_h = 0, 0, 0
        self.momentum = 0.9
        self.epochs = epochs
        self.batch_size = batch_size

        self.output_epoch = output_epoch
        self.allcases = self.get_all_cases(self.v_dim)
        

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
        v_0, v_init = v.copy(), v.copy()
        _, p_h0_v = self.sample_h(v_0)

        while i < self.gibbs_num:
            state_h, _ = self.sample_h(v_init)
            state_v, _ = self.sample_v(state_h)
            v_init = state_v
            i += 1
        else:
            v_k = state_v
            _, p_hk_v = self.sample_h(v_k)
        return v_0, v_k, p_h0_v, p_hk_v

    def parallel_tempering(self, v):
        beta =  self.beta
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

    def weighted_gradient_compute(self, v_0, v_k, p_h0_v, p_hk_v):
        v_k = np.float16(v_k)
        p_hk_v_copy = p_hk_v.copy()

        weights = self.compute_weight(v_k, self.W, self.v_bias, self.h_bias)

        if len(v_k) == len(weights) == len(p_hk_v) == self.batch_size:
            for i in range(self.batch_size):
                v_k[i] = v_k[i] * weights[i]
                p_hk_v_copy[i] = p_hk_v[i] * weights[i]

        else:
            print("*" * 20)

        negative_sampling_w = np.dot(v_k.T, p_hk_v)
        negative_sampling_h_bias = np.sum(p_hk_v_copy, axis = 0)
        negative_sampling_v_bias = np.sum(v_k, axis = 0)

        dw = np.dot(v_0.T, p_h0_v)  / self.batch_size - negative_sampling_w
        dh_bias = (np.sum(p_h0_v, axis = 0)) / self.batch_size - negative_sampling_h_bias
        dv_bias = (np.sum(v_0, axis = 0)) / self.batch_size - negative_sampling_v_bias

        self.v_w = self.momentum * self.v_w + (1 - self.momentum) * dw
        self.v_h = self.momentum * self.v_h + (1 - self.momentum) * dh_bias
        self.v_v = self.momentum * self.v_v + (1 - self.momentum) * dv_bias

        self.W += self.lr * self.v_w - self.lr * self.weight_decay * self.W
        self.v_bias += self.lr * self.v_v
        self.h_bias += self.lr * self.v_h

    def compute_weight(self, train_data, W, v_bias, h_bias):
        weights = self.compute_px_with_Z(train_data, self.W, self.v_bias, self.h_bias)
        return weights / np.sum(weights)

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

    def compute_Z(self, W, v_bias, h_bias):
        first_part = np.dot(self.allcases, v_bias.T).reshape(len(self.allcases), 1)
        second_part = np.sum(np.log(1 + np.exp(np.dot(self.allcases, W) + h_bias)), axis = 1)
        second_part = second_part.reshape(len(self.allcases), 1)
        Z = np.sum(np.exp(first_part + second_part).reshape(-1))
        return Z

    def exp_decay(self, epoch):
       initial_lrate = self.lr
       k = self.exp_lrd
       lrate = initial_lrate * np.exp(-k * epoch)
       return lrate

    def batch_index(self, train_data):
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
        return start, end, data_num

    def compute_metrics(self, epoch):
        logLKH, KL, x = 0, 0, 0
        Z = self.compute_Z(self.W, self.v_bias, self.h_bias)
        probability_list = self.compute_px_with_Z(train_data, self.W, self.v_bias, self.h_bias) / Z
        scaled_probability_list = probability_list / np.sum(probability_list)

        for i in range(len(probability_list)):
            px= probability_list[i]
            N = len(probability_list)
            logLKH += np.log(px)

            x += px

            kl = -np.log(N)/N - np.log(px)/N
            KL += kl
        KL /= N
        logLKH /= N
        Entropy = -np.sum(scaled_probability_list * np.log(scaled_probability_list))
        results = 'epoch {}: KL = {:.5f}, logLKH = {:.4f}, prob_sum = {:.4f}, entropy = {:.4f}, lr = {:.7f}'.format(epoch + 1, KL, logLKH, x, Entropy, self.lr)
        return results, KL, logLKH, x, Entropy

    def train(self, train_data):
        start, end, data_num = self.batch_index(train_data)
        lowest_KL, highest_NLL, highest_probsum = float("inf"), float("inf"), float("inf")

        #f = open("records/wcd.log","w")

        if self.opt_type == "pcd" or "wpcd":
            persistent_chain = None
        
        for epoch in tqdm(range(self.epochs)):
        #for epoch in range(self.epochs):
            self.lr = self.exp_decay(epoch)
            for index in range(data_num):
                # positive sampling
                v0 = train_data[start[index]: end[index]]
                _, p_h0_v = self.sample_h(v0)

                if (self.opt_type == "pcd" or "wpcd") and persistent_chain is None:
                        persistent_chain = np.random.binomial(1, 0.5, size=(self.batch_size, self.v_dim))
                
                # negative sampling
                vk = v0.copy()
                if self.sampling_type == "gibbs_sampling" and self.opt_type == "cdk":
                    _, vk, _, p_hk_v = self.gibbs_sampling(v0)
                    self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "gibbs_sampling" and self.opt_type == "pcd":
                    _, vk, _, p_hk_v = self.gibbs_sampling(persistent_chain)
                    self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "gibbs_sampling" and self.opt_type == "wcd":
                    _, vk, _, p_hk_v = self.gibbs_sampling(v0)
                    self.weighted_gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "gibbs_sampling" and self.opt_type == "wpcd":
                    _, vk, _, p_hk_v = self.gibbs_sampling(persistent_chain)
                    self.weighted_gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "parallel_tempering" and self.opt_type == "cdk":
                    _, vk, p_hk_v =  self.parallel_tempering(v0)
                    self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "parallel_tempering" and self.opt_type == "pcd":
                    exit("choose cdk / wcd !")

                elif self.sampling_type == "parallel_tempering" and self.opt_type == "wcd":
                    _, vk, p_hk_v =  self.parallel_tempering(v0)
                    self.weighted_gradient_compute(v0, vk, p_h0_v, p_hk_v)
                    
                elif self.sampling_type == "parallel_tempering" and self.opt_type == "wpcd":
                    exit("choose cdk / wcd !")

                else:
                    exit("sampling_type = gibbs sampling / parallel tempering !" + '\n' 
                       + "opt_type = cdk / pcd / wcd / wpcd")
                
                if self.opt_type == "pcd" or "pwcd":
                    persistent_chain = vk          

            if epoch + 1 == self.epochs or (epoch + 1) % self.output_epoch == 0 or epoch == 0:
                results, KL, logLKH, x, Entropy = self.compute_metrics(epoch)
                tqdm.write(results)
                #f.write(results + '\n')

                if KL < lowest_KL:
                    lowest_KL = KL
                    highest_NLL = logLKH
                    highest_probsum = x

        optimal_record = "KL {} NLL {} prob_sum {}".format(np.round(lowest_KL, 4), np.round(highest_NLL, 4), np.round(highest_probsum, 4))
        #f.write(record + '\n')
        #f.write('\n')c
        #tqdm.write(record)
        tqdm.write(optimal_record)
        #f.close()




if __name__ == "__main__":
    train_data = np.loadtxt(r'./data/Bars-and-Stripes-4x4.txt')
                            # './data/Bars-and-Stripes-3x3.txt' (14, 9) 27
                            # './data/Bars-and-Stripes-4x4.txt' (30, 16) 48
                            # './data/Labeled-Shifter-4-11.txt' (48, 11) 33
                            # './data/Labeled-Shifter-5-13.txt' (96, 13) 39

    rbm = RBM(v_dim = train_data.shape[1],
                h_dim = train_data.shape[1] * 3,
                gibbs_num = 1,
                opt_type = "cdk",
                sampling_type = "gibbs_sampling", # parallel_tempering gibbs_sampling
                lr = 1e-3,
                epochs= 350000, # 300000, lr 3e-3 lr_decay 1e-10
                batch_size = 14,
                chain_num = 2,
                weight_decay = 2.5e-5, # 2.5e-5
                exp_lrd = 1e-10, # 1e-10
                output_epoch = 10000
                )

    rbm.train(train_data)