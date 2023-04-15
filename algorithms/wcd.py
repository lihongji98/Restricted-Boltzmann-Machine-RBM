import numpy as np
import time
from tqdm import tqdm

class RBM:
    def __init__(self,
                 v_dim, h_dim,
                 lr=5e-4,
                 weight_decay = 1e-5,
                 gibbs_num = 1,
                 epochs = 50000,
                 batch_size = 1,
                 compute_detail = True,
                 binary_kind = "withoutzero"):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.v_bias = np.random.normal(-0.1, 0.1, size = (1,self.v_dim))
        self.h_bias = np.random.normal(-0.1, 0.1, size =(1,self.h_dim))
        self.W = np.random.normal(size = (self.v_dim, self.h_dim))
        self.gibbs_num = gibbs_num
        self.v_w, self.v_v, self.v_h = 0, 0, 0
        self.momentum = 0.9
        self.epochs = epochs
        self.compute_detail = compute_detail
        self.batch_size = batch_size
        self.binary_kind = binary_kind
        self.allcases = self.get_all_cases(self.binary_kind, self.v_dim)

    def sample_h(self, v_input):
        if self.binary_kind == "withoutzero":
            var = -(np.dot(v_input, self.W) + self.h_bias)
            p_h_v = 1 / (1 + np.exp(2 * var))
            state_h = self.state_sample(p_h_v)
        elif self.binary_kind == "withzero":
            p_h_v = 1 / (1 + np.exp(-(np.dot(v_input, self.W) + self.h_bias)))
            state_h = self.state_sample(p_h_v)
        else:
            print("enter 'withzero' or 'withoutzero'!")

        return state_h, p_h_v

    def sample_v(self, h):
        if self.binary_kind == "withoutzero":
            var = -(np.dot(h, self.W.T) + self.v_bias)
            p_v_h = 1 / (1 + np.exp(2 * var))
            state_v = self.state_sample(p_v_h)
        elif self.binary_kind == "withzero":
            p_v_h = 1 / (1 + np.exp(-(np.dot(h, self.W.T) + self.v_bias)))
            state_v = self.state_sample(p_v_h)
        else:
            print("enter 'withzero' or 'withoutzero'!")

        return state_v, p_v_h

    def state_sample(self, p):
        state = []
        uni = np.random.uniform(0,1, size=p[0].shape[0])
        for i in range(len(p)):
            condition = np.less(p[i], uni)
            if self.binary_kind == "withoutzero":
                state_node = np.where(condition, -1, 1)
            elif self.binary_kind == "withzero":
                state_node = np.where(condition, 0, 1)
            else:
                print("enter 'withzero' or 'withoutzero'!")
            state.append(state_node)

        return np.array(state).reshape(p.shape[0], p.shape[1])

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

    def gradient_compute(self, v_0, v_k, p_h0_v, p_hk_v):
        v_k_copy = v_k.copy()
        v_k = np.float16(v_k)
        p_hk_v_copy = p_hk_v.copy()
        negative_sampling = np.dot(v_k.T, p_hk_v)/ self.batch_size

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

        self.W += self.lr * self.v_w #- self.lr * self.weight_decay * self.W
        self.v_bias += self.lr * self.v_v
        self.h_bias += self.lr * self.v_h

    def get_all_cases(self, binary_kind, v_dim):
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
        if binary_kind == "withzero":
            return np.array(all_cases([0, 1], self.v_dim))

        elif binary_kind == "withoutzero":
            return np.array(all_cases([-1, 1], self.v_dim))
        else:
            print("enter 'withzero' or 'withoutzero'!")

    def compute_px_with_Z(self, train_data, W, v_bias, h_bias):
        probability = []
        if self.binary_kind == "withoutzero":
            for l in range(len(train_data)):
                train_data_one_piece = train_data[l]
                product_value = 1
                exp_av = np.exp(np.dot(v_bias, train_data_one_piece))
                for i in range(h_bias.shape[1]):
                    product_value = product_value * (np.exp(np.dot(W.T[i], train_data_one_piece)+ h_bias.T[i]) +
                                                     np.exp(-np.dot(W.T[i], train_data_one_piece)- h_bias.T[i]))
                px_with_Z = exp_av * product_value
                probability.append(px_with_Z[0])

        elif self.binary_kind == "withzero":
            for l in range(len(train_data)):
                train_data_one_piece = train_data[l]
                product_value = 1
                exp_av = np.exp(np.dot(v_bias, train_data_one_piece))
                for i in range(h_bias.shape[1]):
                    product_value = product_value * (np.exp(np.dot(W.T[i], train_data_one_piece) + h_bias.T[i]) + 1)
                px_with_Z = exp_av * product_value
                probability.append(px_with_Z[0])

        else:
            print("enter 'withzero' or 'withoutzero'!")

        return probability

    def compute_Z(self, W, v_bias, h_bias):
        Z = 0
        if self.binary_kind == "withoutzero":
            for l in range(len(self.allcases)):
                train_data_one = self.allcases[l]
                exp_av = np.exp(np.dot(v_bias, train_data_one))
                product = 1
                for j in range(h_bias.shape[1]):
                    product = product * (np.exp(np.dot(train_data_one.T, W.T[j]) + h_bias.T[j]) +
                                         np.exp(-np.dot(train_data_one.T, W.T[j]) - h_bias.T[j]))
                total = exp_av * product

                Z += total

        elif self.binary_kind == "withzero":
            for l in range(len(self.allcases)):
                train_data_one = self.allcases[l]
                exp_av = np.exp(np.dot(v_bias, train_data_one))
                product = 1
                for j in range(h_bias.shape[1]):
                    product = product * (np.exp(np.dot(W.T[j], train_data_one) + h_bias.T[j]) + 1)
                total = exp_av * product
                Z += total[0]
        else:
            print("enter 'withzero' or 'withoutzero'!")

        return Z

    def compute_weight(self, train_data, W, v_bias, h_bias):
        weights = []
        for l in range(len(train_data)):
            train_data_one_piece = train_data[l]
            product_value = 1
            exp_av = np.exp(np.dot(v_bias, train_data_one_piece))
            for i in range(h_bias.shape[1]):
                product_value = product_value * (np.exp(np.dot(W.T[i], train_data_one_piece) + h_bias.T[i]) + 1)
            px_with_Z = exp_av * product_value
            weights.append(px_with_Z[0])
        return weights / np.sum(weights)

    def exp_decay(self, epoch, k = 1 * 1e-10): #9 * 1e-11
       initial_lrate = self.lr
       lrate = initial_lrate * np.exp(-k * epoch)
       return lrate

    def train(self, train_data):
        idx = [i for i in range(train_data.shape[0])]
        start = [i for i in idx if i % self.batch_size == 0]
        end = []
        for start_idx in start:
            end_idx = start_idx + self.batch_size
            if end_idx < len(idx):
                end.append(end_idx)
            else:
                end.append(len(idx))
        data_num = len(start)

        lowest_KL = float("inf")
        lowest_KL_epoch = 0

        #f = open("records/wcd.log","w")

        for epoch in tqdm(range(self.epochs)):
        #for epoch in range(self.epochs):
            np.random.shuffle(train_data)
            #self.lr = self.exp_decay(epoch)
            epoch_start_time = time.time()
            for index in range(data_num):
                # positive sampling
                v0 = train_data[start[index]: end[index]]
                _, p_h0_v = self.sample_h(v0)

                # negative sampling
                vk = v0.copy()
                _, vk, _, p_hk_v = self.gibbs_sampling(v0)
                self.gradient_compute(v0, vk, p_h0_v, p_hk_v)

            if self.compute_detail:
                KL_list, log_LKH_list = [], []
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

                KL_list.append(KL)
                log_LKH_list.append(logLKH)

                probability_list = [probability_list[i]/Z for i in range(len(probability_list))]
                x = np.sum(probability_list)

                epoch_end_time = time.time()

                # if KL[0] < lowest_KL:
                #     lowest_KL = KL[0]
                #     lowest_KL_epoch = epoch

                if(epoch % 100 == 0):
                    results = 'epoch:{} ==>  KL = {:.4f}, logLKH = {:.4f}, prob_sum = {:.4f}, time = {:.2f}s' \
                              .format(epoch, KL, logLKH, x, epoch_end_time-epoch_start_time)

                    #f=open("log -1&1.txt","a")
                    #f=open("50000epochs_new.txt","a")
                    #f.write(results + '\n')
                    #f.close()
                    tqdm.write(results)

            else:
                if epoch + 1 == self.epochs or (epoch + 1) % 100 == 0 or epoch == 0:
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
                    #tqdm.write(results)
                    #f.write(results + '\n')

                    if KL < lowest_KL:
                        lowest_KL = KL
                        lowest_KL_epoch = epoch

        record = "The lowest KL is {} in epoch {}".format(lowest_KL, lowest_KL_epoch)
        #f.write(record + '\n')
        #f.write('\n')
        print(record)
        #f.close()

if __name__ == "__main__":
    train_data = np.loadtxt(r'../3x3.txt')

    visible_node_num = train_data.shape[1]
    hidden_node_num = 20
    lr = 2.5 * 1e-2
    # epoch:100000 lr: 5e-4  -------->  k = 1

    for i in range(10):
        rbm = RBM(visible_node_num, hidden_node_num, lr,
            binary_kind="withzero",
            epochs= 400000 , batch_size = 14, gibbs_num = 1, weight_decay = 1e-5,
            compute_detail=False)
        rbm.train(train_data)
