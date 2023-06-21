import numpy as np
from tqdm import tqdm

class RBM:
    def __init__(self,
                 v_dim, h_dim,
                 sampling_type,
                 opt_type,
                 lr=1e-3,
                 gibbs_num = 1,
                 epochs = 100000,
                 batch_size = 14,
                 chain_num = 2,
                 output_epoch = 10000
                 ):
        self.v_dim = v_dim
        self.h_dim = h_dim
        # self.v_bias = np.random.normal(-0.1, 0.1, size = (1,self.v_dim))
        # self.h_bias = np.random.normal(-0.1, 0.1, size =(1,self.h_dim))
        # self.W = np.random.normal(size = (self.v_dim, self.h_dim))

        self.v_bias = np.zeros((1,self.v_dim))
        self.h_bias = np.zeros((1,self.h_dim))
        self.W = np.random.normal(0, 0.01, size = (self.v_dim, self.h_dim))

        self.lr = lr
        self.init_lr = lr

        self.opt_type = opt_type
        self.v_w, self.v_v, self.v_h = 0, 0, 0
        self.momentum = 0.9
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.sampling_type = sampling_type
        self.gibbs_num = gibbs_num
        if self.sampling_type == "parallel_tempering":
            self.chain_num = chain_num
            # self.chains = 2 * np.float32(np.random.binomial(1, 0.5, (self.chain_num, self.v_dim))) - 1
            self.chains_opt = 2 * np.float16(np.random.binomial(1, 0.5, (self.batch_size, self.chain_num, self.v_dim))) - 1
            self.beta = np.linspace(0.0, 1.0, self.chain_num).reshape(self.chain_num, 1)
            self.swap_time = max(int(np.sqrt(self.chain_num)), 2)

        self.output_epoch = output_epoch
        self.allcases = self.get_all_cases(self.v_dim)

        if (self.opt_type == "cdk" or self.opt_type == "wcd") and self.sampling_type == "gibbs_sampling":
            self.if_lr_decay = False
            self.weight_decay = 0
        elif (self.opt_type == "cdk" or self.opt_type == "wcd") and self.sampling_type == "parallel_tempering":
            self.if_lr_decay = True
            self.weight_decay = 2.5e-5
        else:
            self.if_lr_decay = True
            self.weight_decay = 2.5e-5

        

    def sample_h(self, v_input):
        v_input = np.float16(v_input)
        var = -(np.dot(v_input, self.W) + self.h_bias)
        p_h_v =  1 / (1 + np.exp(2*var))
        state_h = self.state_sample(p_h_v)
        return state_h, p_h_v
    
    def sample_v(self, h):
        h = np.float16(h)
        var = -(np.dot(h, self.W.T) + self.v_bias)
        p_v_h = 1 / (1 + np.exp(2*var))
        state_v = self.state_sample(p_v_h)
        return state_v, p_v_h

    def state_sample(self, p):
        uni = np.random.uniform(0,1, size = (p.shape))
        condition = np.less(p, uni)
        state_node = np.where(condition, -1, 1)
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

    def parallel_tempering_opt(self, v):
        x = (np.einsum("bcv,vh->bch", self.chains_opt, self.W) + self.h_bias) * self.beta
        p_h_v = np.power(np.exp(-2 * x) + 1, -1)
        hid = self.state_sample(p_h_v)

        for _ in range(self.gibbs_num):
            x = (np.einsum("bch,vh->bcv", hid, self.W) + self.v_bias) * self.beta
            p_v_h = np.power(np.exp(-2 * x) + 1, -1)
            vis = self.state_sample(p_v_h)

            x = (np.einsum("bcv,vh->bch", vis, self.W) + self.h_bias) * self.beta
            p_h_v = np.power(np.exp(-2 * x) + 1, -1)
            hid = self.state_sample(p_h_v)

        x = (np.einsum("bch,vh->bcv", hid, self.W) + self.v_bias) * self.beta
        self.chains_opt = np.power(np.exp(-2 * x) + 1, -1)
        self.swap_state_opt(self.chains_opt, hid)

        samples = self.chains_opt[:,-1,:]
        v_k = self.state_sample(samples)
        p_hk_v = np.power(np.exp((-2) * (np.dot(v_k, self.W) + self.h_bias)) + 1, -1)

        return samples, v_k, p_hk_v

    def swap_state_opt(self, chain, hid):
        energy = - np.einsum("bcv,vh,bch->bc",chain, self.W, hid) - np.einsum("lv,bcv->bc", self.v_bias,chain) - np.einsum("lh,bch->bc", self.h_bias, hid)
        energy = energy.reshape(self.batch_size, self.chain_num, 1) * self.beta
        #particle = np.resize(np.arange(self.chain_num), (self.batch_size, self.chain_num))
        for _ in range(self.swap_time):
            odd_ind = [i for i in range(0, len(self.beta) - 1, 2)]
            odd_ind_next = [i+1 for i in range(0, len(self.beta) - 1, 2)]
            odd_e = energy.reshape(self.batch_size, self.chain_num)[:,odd_ind]
            odd1_e = energy.reshape(self.batch_size, self.chain_num)[:,odd_ind_next]
            odd_beta = self.beta.reshape(self.chain_num)[odd_ind]
            odd1_beta = self.beta.reshape(self.chain_num)[odd_ind_next]
            r_odd = np.exp((odd1_e - odd_e) * (odd1_beta - odd_beta))
            r_odd = np.where(r_odd > np.random.rand(self.batch_size, len(odd_ind)), 1, 0)
            for i in range(r_odd.shape[0]):
                for j in range(r_odd.shape[1]):
                    if r_odd[i][j] == 1:
                        #particle[i][odd_ind[j]], particle[i][odd_ind[j] + 1] = particle[i][odd_ind[j] + 1], particle[i][odd_ind[j]]
                        temp1 = chain[i][odd_ind[j],:].copy()
                        temp2 = chain[i][odd_ind[j]+1,:].copy()
                        chain[i][odd_ind[j]+1,:] = temp1
                        chain[i][odd_ind[j],:] = temp2

                        temp1 = energy[i][odd_ind[j],:].copy()
                        temp2 = energy[i][odd_ind[j]+1,:].copy()
                        energy[i][odd_ind[j]+1,:] = temp1
                        energy[i][odd_ind[j],:] = temp2

                        temp1 = hid[i][odd_ind[j],:].copy()
                        temp2 = hid[i][odd_ind[j]+1,:].copy()
                        hid[i][odd_ind[j]+1,:] = temp1
                        hid[i][odd_ind[j],:] = temp2

            even_ind = [i for i in range(1, len(self.beta) - 1, 2)]
            even_ind_next = [i+1 for i in range(1, len(self.beta) - 1, 2)]
            even_e = energy.reshape(self.batch_size, self.chain_num)[:,even_ind]
            even1_e = energy.reshape(self.batch_size, self.chain_num)[:, even_ind_next]
            even_beta = self.beta.reshape(self.chain_num)[even_ind]
            even1_beta = self.beta.reshape(self.chain_num)[even_ind_next]
            r_even = np.exp((even1_e - even_e) * (even1_beta - even_beta))
            r_even = np.where(r_even > np.random.rand(self.batch_size, len(even_ind)), 1, 0)
            for i in range(r_even.shape[0]):
                for j in range(r_even.shape[1]):
                    if r_even[i][j] == 1:
                        #particle[i][even_ind[j]], particle[i][even_ind[j] + 1] = particle[i][even_ind[j] + 1], particle[i][even_ind[j]]
                        temp1 = chain[i][even_ind[j],:].copy()
                        temp2 = chain[i][even_ind[j]+1,:].copy()
                        chain[i][even_ind[j]+1,:] = temp1
                        chain[i][even_ind[j],:] = temp2

                        temp1 = energy[i][even_ind[j],:].copy()
                        temp2 = energy[i][even_ind[j]+1,:].copy()
                        energy[i][even_ind[j]+1,:] = temp1
                        energy[i][even_ind[j],:] = temp2

                        temp1 = hid[i][even_ind[j],:].copy()
                        temp2 = hid[i][even_ind[j]+1,:].copy()
                        hid[i][even_ind[j]+1,:] = temp1
                        hid[i][even_ind[j],:] = temp2   

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

        return dw, dv_bias, dh_bias

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

        return dw, dv_bias, dh_bias

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
        return np.array(all_cases([-1, 1], self.v_dim))

    def compute_px_with_Z(self, train_data, W, v_bias, h_bias):
        train_data = np.float16(train_data)
        first_part = np.dot(train_data, v_bias.T).reshape(train_data.shape[0], 1)
        second_part = np.sum(np.log(np.exp(-np.dot(train_data, W) - h_bias) + np.exp(np.dot(train_data, W) + h_bias)), axis = 1)
        second_part = second_part.reshape(train_data.shape[0], 1)
        pxz = np.exp(first_part + second_part)
        return pxz.reshape(-1)

    def compute_Z(self, W, v_bias, h_bias):
        first_part = np.dot(self.allcases, v_bias.T).reshape(len(self.allcases), 1)
        second_part = np.sum(np.log(np.exp(-np.dot(self.allcases, W) - h_bias) + np.exp(np.dot(self.allcases, W) + h_bias)), axis = 1)
        second_part = second_part.reshape(len(self.allcases), 1)
        Z = np.sum(np.exp(first_part + second_part).reshape(-1))
        return Z

    def lr_decay(self, epoch):
    #    initial_lrate = self.lr
    #    k = self.exp_lrd 
    #    lrate = initial_lrate * np.exp(-k * epoch)
       lrate = (1e-6 - self.init_lr)/ self.epochs * epoch + self.init_lr
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

    def compute_metrics(self, epoch, batch, train_data):
        logLKH, KL, x = 0, 0, 0
        Z = self.compute_Z(self.W, self.v_bias, self.h_bias)
        probability_list = self.compute_px_with_Z(train_data, self.W, self.v_bias, self.h_bias) / Z
        N = len(probability_list)

        KL = -np.log(N) - np.sum((np.log(probability_list) / N))
        logLKH = np.sum(np.log(probability_list))
        x = np.sum(probability_list)
        Entropy = (-np.sum(probability_list * np.log(probability_list))) / np.log(batch)
        results = 'epoch {}: KL = {:.5f}, logLKH = {:.4f}, prob_sum = {:.4f}, entropy_per = {:.4f}, lr = {:.7f}'.format(epoch + 1, KL, logLKH, x, Entropy, self.lr)

        return results, KL, logLKH, x, Entropy

    def train(self, train_data, step=0):
        start, end, data_num = self.batch_index(train_data)
        batch = train_data.shape[0]

        if self.opt_type == "pcd" or "wpcd":
            persistent_chain = None
        
        pbar = tqdm(total=self.epochs, leave=False, desc='eval', dynamic_ncols=True, bar_format='{l_bar}')
        for epoch in range(self.epochs):
            if self.if_lr_decay == True:
                self.lr = self.lr_decay(epoch)
            for index in range(data_num):
                # positive sampling
                v0 = train_data[start[index]: end[index]]
                _, p_h0_v = self.sample_h(v0)

                if (self.opt_type == "pcd" or "wpcd") and persistent_chain is None:
                        persistent_chain = 2 * np.random.binomial(1, 0.5, size=(self.batch_size, self.v_dim)) - 1
                
                # negative sampling
                vk = v0.copy()
                if self.sampling_type == "gibbs_sampling" and self.opt_type == "cdk":
                    _, vk, _, p_hk_v = self.gibbs_sampling(v0)
                    dw, dvb, dhb = self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "gibbs_sampling" and self.opt_type == "pcd":
                    _, vk, _, p_hk_v = self.gibbs_sampling(persistent_chain)
                    dw, dvb, dhb = self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "gibbs_sampling" and self.opt_type == "wcd":
                    _, vk, _, p_hk_v = self.gibbs_sampling(v0)
                    dw, dvb, dhb = self.weighted_gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "gibbs_sampling" and self.opt_type == "wpcd":
                    _, vk, _, p_hk_v = self.gibbs_sampling(persistent_chain)
                    dw, dvb, dhb = self.weighted_gradient_compute(v0, vk, p_h0_v, p_hk_v)

                elif self.sampling_type == "parallel_tempering" and self.opt_type == "cdk":
                    _, vk, p_hk_v =  self.parallel_tempering_opt(v0)
                    dw, dvb, dhb = self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

                else:
                    exit("sampling_type = gibbs sampling / parallel tempering !" + '\n' 
                       + "opt_type = cdk / pcd / wcd / wpcd")
                
                if self.opt_type == "pcd" or "wpcd":
                    persistent_chain = vk          

            if epoch + 1 == self.epochs or (epoch + 1) % self.output_epoch == 0 or epoch == 0:
                results, KL, logLKH, x, Entropy = self.compute_metrics(epoch, batch, train_data)
                tqdm.write(results)

            pbar.set_description("step: {}  epoch: {}/{}".format(step+1, epoch+1, self.epochs))
            pbar.update() 


if __name__ == "__main__":
    train_data = np.loadtxt(r'./algorithms/data/BS4.txt')
    train_data = 2 * train_data - 1

    rbm = RBM(v_dim = train_data.shape[1],
                            h_dim = train_data.shape[1] * 3,
                            batch_size = train_data.shape[0],
                            lr = 0.005,
                            opt_type = "cdk",
                            sampling_type = "gibbs_sampling",
                            epochs= 100000,
                            gibbs_num = 1,
                            output_epoch = 1000,
                            chain_num = 2,)

    KL_record = rbm.train(train_data)
