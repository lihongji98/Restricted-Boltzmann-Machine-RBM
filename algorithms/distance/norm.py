import numpy as np

# input -> exact_gradient -> dw(1000000, 9, 20) dvb(1000000, 1, 9) dhb(1000000, 1, 20)
#          specific_opt   -> dw(1000000, 9, 20) dvb(1000000, 1, 9) dhb(1000000, 1, 20)
# norm  -> L1, L2, cos-sim, pierson-sim
# output -> gradient difference -> dif_w(1000000, 1), dif_dvb(1000000, 1), dif_dhb(1000000, 1)

class norm_compute:
    def __init__(self, name = "RBM"):
        self.name = name

    def Frobenius_norm(self, exact_dw,exact_dvb, exact_dhb, opt_dw, opt_dvb, opt_dhb):
        dw_F   = np.sqrt(np.sum(np.power(exact_dw - opt_dw, 2), axis = (1, 2)))
        dvb_F  = np.sqrt(np.sum(np.power(exact_dvb - opt_dvb, 2), axis = (1, 2)))
        dhb_F  = np.sqrt(np.sum(np.power(exact_dhb - opt_dhb, 2), axis = (1, 2)))
        return dw_F, dvb_F, dhb_F

    def cos_sim(self, exact_dw,exact_dvb, exact_dhb, opt_dw, opt_dvb, opt_dhb):
        epsilon = 1e-12
        dw_cos_sim = np.sum((exact_dw * opt_dw), axis = (1 ,2)) / (np.sqrt(np.sum(np.power(exact_dw, 2), axis = (1,2)))*np.sqrt(np.sum(np.power(opt_dw, 2), axis = (1,2))) + epsilon)
        dvb_cos_sim = np.sum((exact_dvb * opt_dvb), axis = (1 ,2)) / (np.sqrt(np.sum(np.power(exact_dvb, 2), axis = (1,2)))*np.sqrt(np.sum(np.power(opt_dvb, 2), axis = (1,2))) + epsilon)
        dhb_cos_sim = np.sum((exact_dhb * opt_dhb), axis = (1 ,2)) / (np.sqrt(np.sum(np.power(exact_dhb, 2), axis = (1,2)))*np.sqrt(np.sum(np.power(opt_dhb, 2), axis = (1,2))) + epsilon)
        return dw_cos_sim, dvb_cos_sim, dhb_cos_sim

    def pearson_sim(self, exact_dw,exact_dvb, exact_dhb, opt_dw, opt_dvb, opt_dhb):
        pass