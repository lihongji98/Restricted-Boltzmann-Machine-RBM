import numpy as np
from tqdm import tqdm

class RBM:
    def __init__(self,
                 v_dim, h_dim,
                 lr=5e-4, exp_lrd = 0,
                 epochs = 50000,
                 batch_size = 14,
                 output_epoch = 10000
                 ):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.lr = lr
        self.exp_lrd = exp_lrd
        self.v_bias = np.random.normal(-0.1, 0.1, size = (1,self.v_dim))
        self.h_bias = np.random.normal(-0.1, 0.1, size =(1,self.h_dim))
        self.W = np.random.normal(size = (self.v_dim, self.h_dim))
        self.v_w, self.v_v, self.v_h = 0, 0, 0
        self.momentum = 0.9
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_epoch = output_epoch
        self.allcases = self.get_all_cases(self.v_dim)

    def sample_h(self, v_input):
        p_h_v = 1 / (1 + np.exp( -(np.dot(v_input, self.W) + self.h_bias)))
        state_h = self.state_sample(p_h_v)
        return state_h, p_h_v

    def sample_v(self, h):
        p_v_h = 1 / (1 + np.exp(-(np.dot(h, self.W.T) + self.v_bias)))
        state_v = self.state_sample(p_v_h)
        return state_v, p_v_h

    def state_sample(self, p):
        uni = np.random.uniform(0,1, size = (p.shape[0], p.shape[1]))
        condition = np.less(p, uni)
        state_node = np.where(condition, 0, 1)
        return state_node

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

    def compute_exp_model(self,Z):
        v = np.array(self.allcases.copy()).reshape(2**self.v_dim, self.v_dim)
        _, phx = self.sample_h(v)
        phx = phx.reshape(2**self.v_dim, 1, self.h_dim) # 512, 1, 20
        X = np.array(self.allcases).reshape(2**self.v_dim, 1, self.v_dim) # 512, 1, 9
        px = self.compute_px_with_Z(v, self.W, self.v_bias, self.h_bias).reshape(2**self.v_dim, 1) / Z # 512, 1

        X = np.float32(X)

        phx_X = np.einsum("abc,abd->acd", X, phx) # 512, 9, 20
        dw_exp_model_matrix = np.einsum("ab,acd->acd", px, phx_X)
        dw_exp_model = np.sum(dw_exp_model_matrix, axis = 0)

        px_X = np.einsum("ab,abc->abc", px, X)
        dvb_exp_model = np.sum(px_X, axis = 0)

        px_phx = np.einsum("ab,abc->abc",px, phx)
        dhb_exp_model = np.sum(px_phx, axis = 0)

        return dw_exp_model, dvb_exp_model, dhb_exp_model

    def compute_exp_data(self, data, phx):
        data = np.float32(data)

        dw_exp_data = np.dot(data.T, phx) / self.batch_size

        dvb_exp_data = np.sum(data, axis = 0) / self.batch_size

        dhb_exp_data = np.sum(phx, axis = 0) / self.batch_size

        return dw_exp_data, dvb_exp_data, dhb_exp_data

    def gradient_compute(self, v, phx, Z):
        dw_exp_model, dvb_exp_model, dhb_exp_model = self.compute_exp_model(Z)
        dw_exp_data, dvb_exp_data, dhb_exp_data = self.compute_exp_data(v, phx)

        dw = dw_exp_data - dw_exp_model
        dv_bias = dvb_exp_data - dvb_exp_model
        dh_bias = dhb_exp_data - dhb_exp_model

        self.v_w = self.momentum * self.v_w + (1 - self.momentum) * dw
        self.v_h = self.momentum * self.v_h + (1 - self.momentum) * dh_bias
        self.v_v = self.momentum * self.v_v + (1 - self.momentum) * dv_bias

        self.W += self.lr * self.v_w
        self.v_bias += self.lr * self.v_v
        self.h_bias += self.lr * self.v_h

    def exp_decay(self, epoch):
       initial_lrate = self.lr
       k = self.exp_lrd
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
        highest_NLL = float("-inf")
        highest_probsum = float("-inf")

        for epoch in tqdm(range(self.epochs)):
            Z = self.compute_Z(self.W, self.v_bias, self.h_bias)
            probability_list = self.compute_px_with_Z(train_data, self.W, self.v_bias, self.h_bias)

            self.lr = self.exp_decay(epoch)
            np.random.shuffle(train_data)
            for index in range(data_num):
                v0 = train_data[start[index]: end[index]]
                _, p_h0_v = self.sample_h(v0)
                self.gradient_compute(v0, p_h0_v, Z)

            if epoch + 1 == self.epochs or (epoch + 1) % self.output_epoch == 0 or epoch == 0:
                logLKH, KL = 0, 0
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

                if KL < lowest_KL:
                    lowest_KL = KL
                    highest_NLL = logLKH
                    highest_probsum = x

        record = "KL {} NLL {} prob_sum {}".format(np.round(lowest_KL, 4), np.round(highest_NLL, 4), np.round(highest_probsum, 4))
        #f.write(record + '\n')
        #f.write('\n')
        print(record)
        #f.close()


    def gradient_output(self, v, phx, Z):
        dw_exp_model, dvb_exp_model, dhb_exp_model = self.compute_exp_model(Z)
        dw_exp_data, dvb_exp_data, dhb_exp_data = self.compute_exp_data(v, phx)

        dw = dw_exp_data - dw_exp_model
        dv_bias = dvb_exp_data - dvb_exp_model
        dh_bias = dhb_exp_data - dhb_exp_model

        self.v_w = self.momentum * self.v_w + (1 - self.momentum) * dw
        self.v_h = self.momentum * self.v_h + (1 - self.momentum) * dh_bias
        self.v_v = self.momentum * self.v_v + (1 - self.momentum) * dv_bias

        self.W += self.lr * self.v_w #- self.lr * self.weight_decay * self.W
        self.v_bias += self.lr * self.v_v
        self.h_bias += self.lr * self.v_h

        return dw, dv_bias, dh_bias

    def gradient_compare(self, train_data):
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

        dw_list, dv_bias_list, dh_bias_list = [], [], []
        for epoch in tqdm(range(self.epochs)):
            Z = self.compute_Z(self.W, self.v_bias, self.h_bias)
            probability_list = self.compute_px_with_Z(train_data, self.W, self.v_bias, self.h_bias)

            np.random.shuffle(train_data)
            for index in range(data_num):
                v0 = train_data[start[index]: end[index]]
                _, p_h0_v = self.sample_h(v0)
                dw, dv_bias, dh_bias = self.gradient_output(v0, p_h0_v, Z)

                if epoch % 1 == 0:
                    dw_list.append(dw)
                    dv_bias_list.append(dv_bias)
                    dh_bias_list.append(dh_bias)
        dw_list = np.array(dw_list).reshape(len(dw_list), self.v_dim, self.h_dim)
        dv_bias_list = np.array(dv_bias_list).reshape(len(dv_bias_list), 1, self.v_dim)
        dh_bias_list = np.array(dh_bias_list).reshape(len(dh_bias_list), 1, self.h_dim)
        np.save("./gradient/exact_dw.npy", dw_list)
        np.save("./gradient/exact_dvb.npy", dv_bias_list)
        np.save("./gradient/exact_dhb.npy", dh_bias_list)


if __name__ == "__main__":
    train_data = np.loadtxt(r'./data/Bars-and-Stripes-3x3.txt')
                            # './data/Bars-and-Stripes-3x3.txt' (14, 9) 27
                            # './data/Bars-and-Stripes-4x4.txt' (30, 16) 48
                            # './data/Labeled-Shifter-4-11.txt' (48, 11) 33
                            # './data/Labeled-Shifter-5-13.txt' (96, 13) 39

    rbm = RBM(v_dim = train_data.shape[1],
                h_dim = train_data.shape[1] * 3,
                lr = 3e-3,
                exp_lrd = 1e-10,
                epochs= 350000, # 300000, lr 3e-3 lr_decay 1e-10
                batch_size = 14,
                output_epoch = 5000
                )
    
    rbm.W = np.load('initial/W_3.npy')
    rbm.v_bias = np.load('initial/v_bias_3.npy')
    rbm.h_bias = np.load('initial/h_bias_3.npy')
    
    rbm.train(train_data)