from wrf import getvar, ALL_TIMES, to_np, interplevel, smooth2d
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from scipy.misc import derivative
from sympy import diff
import math
from scipy import interpolate


# P = (getvar(ncfile, 'p') + getvar(ncfile, 'PB')) * 0.1  # 气压mb
# # QVAPOR = getvar(ncfile, 'QVAPOR')  # 水汽混合比
# # e = QVAPOR * P / (0.622 + QVAPOR)  # 水汽压
# # # N = 77.6/T*(P+(4810*e)/T)
# # n = 1 + 77.6 * 10 ** (-6) * P / T + 0.372 * e / (T ** 2)  # 折射指数
# # N = 10 ** 6 * (n - 1)  # 折射率
# # print(N.shape)
# # length = 2000  # 插值长度
# # z = getvar(ncfile, 'z', units="m")
# # # height = z
# # N_PBLH_INTERP = np.zeros([length, height.shape[1], height.shape[2]], dtype=np.float64)  # 插值后的N
# # n_PBLH_INTERP = np.zeros([length, height.shape[1], height.shape[2]], dtype=np.float64)  # 插值后的n
# # # EDH = getvar(ncfile,'EDh')
# # index = 0
# # for i in range(0, 20000, 10):
# #     N_PBLH = interplevel(N, height.data, i, missing=True)
# #     N_PBLH_INTERP[index, :, :] = N_PBLH.squeeze()
# #     n_PBLH = interplevel(n, height.data, i, missing=True)
# #     n_PBLH_INTERP[index, :, :] = n_PBLH.squeeze()
# #     index += 1
# # # io.savemat("n.mat", {"n": n_PBLH_INTERP})
# # # io.savemat("N.mat", {"N": N_PBLH_INTERP})
# # plt.imshow(N[10, :, :])
#
# e_s0 = 6.1078  # 0度时饱和水汽压
# ncfile = Dataset(wrfout_file)


# key = ncfile.variables.keys()
# # slp = getvar(ncfile, 'slp')
# # print(key)
# times = getvar(ncfile, 'times', timeidx=ALL_TIMES)  # 时间
# XLAT = getvar(ncfile, 'XLAT', timeidx=ALL_TIMES)  # 经度
# XLONG = getvar(ncfile, 'XLONG')  # 纬度
# height = getvar(ncfile, 'height')  # 高度
# T = getvar(ncfile, 'T') + 300  # 温度
#

class WrfUse:
    def __init__(self, wrf_file_path, length, dlen, timeidx=0):
        self.timeidx = timeidx
        self.ncfile = Dataset(wrfout_file)
        self.length = length
        self.dlen = dlen

    @property
    def get_keys(self):
        return self.ncfile.variables.keys()

    def get_var(self, key):
        var = getvar(self.ncfile, key, timeidx=self.timeidx)
        return var

    @property
    def get_T(self):
        """
        相对温度k
        :return:
        """
        T = self.get_var('T') + 300
        return T

    @property
    def get_P(self):
        """
         压强，hpa
        :return:
        """
        P = (self.get_var('p') + self.get_var('PB')) * 0.01  # 气压mb
        return P

    @property
    def get_e(self):
        """
        水汽压hpa
        :return:
        """
        QVAPOR = self.get_var('QVAPOR')
        e = QVAPOR * self.get_P / (0.628 + QVAPOR)  # 水汽压
        return e

    @property
    def get_n(self):
        """
        折射指数
        :return:
        """
        n = 1 + 77.6 * 10 ** (-6) * self.get_P / self.get_T \
            + 0.372 * self.get_e / (self.get_T ** 2)  # 折射指数
        return n

    @property
    def get_N(self):
        """
        折射率
        :return:
        """
        N = 10 ** 6 * (self.get_n - 1)  # 折射率
        return N

    def get_gradient(self, x, y):
        gradients = np.zeros([x.shape[0], x.shape[1], x.shape[2]])
        for xx in range(x.shape[1]):
            for yy in range(x.shape[2]):
                coefficient = np.polyfit(x[:, xx, yy], y[:, xx, yy], 6)
                # plt.plot(x.data[:, xx, yy], y.data[:, xx, yy])
                # import pdb;pdb.set_trace()
                gradient_coefficient = np.polyder(coefficient)
                gradient = np.polyval(gradient_coefficient, x[:, xx, yy])
                # plt.plot(x.data[:, xx, yy], gradient)
                gradients[:, xx, yy] = gradient
        return gradients

    @property
    def get_N_gradient(self):
        # N_gradient = -77.6 / self.get_T ** 2 * (self.get_P + 9620 * self.get_e / self.get_T) * \
        #              self.get_gradient(self.get_height, self.get_T) + 77.6 / self.get_T * \
        #              self.get_gradient(self.get_height,self.get_P) + 373256 / self.get_T ** 2 *\
        #              self.get_gradient(self.get_height, self.get_e)
        N_gradient = self.get_gradient(self.get_height, self.get_N)
        return N_gradient

    @property
    def get_HGT(self):
        """
        海平面高度
        :return:
        """
        HGT = self.get_var('HGT')
        return HGT

    @property
    def get_height(self):
        """
        位势高度
        :return:
        """
        # height = self.get_var('z')
        height = (w.get_var('PH') + w.get_var("PHB")) / 9.81
        return height[0:-1, :, :]

    def insert_value(self, data):
        '''

        :param length:长度
               dlen:步长
        :return:按高度插值后的量
        '''
        # height = self.get_height - (self.get_height[0, :, :] - self.get_HGT)
        height = self.get_height
        N_PBLH_INTERP = np.zeros([self.length // self.dlen, height.shape[1], height.shape[2]], dtype=np.float64)  # 插值后的N
        # import pdb;pdb.set_trace()
        index = 0
        for i in range(0, self.length, self.dlen):
            N_PBLH = interplevel(data, height.data, i, missing=np.nan)
            N_PBLH_INTERP[index, :, :] = N_PBLH
            index += 1
        return N_PBLH_INTERP

    @property
    def get_N_PBLH_INTERP(self):
        """

        :return: 插值后的折射率
        """
        res = self.insert_value(self.get_N)
        return res

    @property
    def get_n_PBLH_INTERP(self):
        """

        :return: 插值后的折射指数
        """
        res = self.insert_value(self.get_n)
        return res

    def imshow(self, data):
        data = data.squeeze()
        r, l = data.shape
        res = data
        for i in range(r):
            res[i, :] = data[r - i - 1, :]
        plt.imshow(res)
        plt.colorbar()

    def savemat(self, savepath, key, value):
        value = value.data
        io.savemat(savepath + key + '.mat', {key: value})

    def get_height_in(self, shape=None):
        """

        :param delen: 步长
        :param shape: 插值范围大小
        :return:
        """
        if not shape:
            height_in = np.zeros(shape)
        else:
            height_in = np.zeros_like(self.get_N_PBLH_INTERP)
        index = 0
        for h in range(height_in.shape[0]):
            height_in[h, :, :] = index
            index += self.dlen
        return height_in

    @property
    def get_pre_duct_h(self):
        N_gradient = self.get_gradient(self.get_height, self.get_N)
        N_gradient_in = self.insert_value(N_gradient)
        height_in = self.get_height_in(shape=N_gradient_in.shape)
        duct_h = np.zeros([N_gradient_in.shape[1], N_gradient_in.shape[2]])
        for i in range(N_gradient_in.shape[1]):
            for j in range(N_gradient_in.shape[2]):
                for k in range(N_gradient_in.shape[0]):
                    if N_gradient_in[k, i, j] < -0.157 and not np.isnan(N_gradient_in[k, i, j]):
                        duct_h[i, j] = height_in[k, i, j]
                        break
        return duct_h


savepath = "../savedata/"
wrfout_file = '/home/ionolab/download/wrf/WRF/run/wrfout_d01_2019-07-31_00:00:00'
w = WrfUse(wrfout_file, length=1000,dlen=1)
# n_PBLH_INTERP = w.get_n_PBLH_INTERP
N_PBLH_INTERP = w.get_N_PBLH_INTERP
HGT = w.get_HGT
n = w.get_n
N = w.get_N
P = w.get_P
height = w.get_height
N_gradient = w.get_gradient(height, N)
N_gradient_in = w.insert_value(N_gradient)
per_duct_h = w.get_pre_duct_h
# plt.imshow(HGT)
# # io.savemat('HGT.mat', {"HGT": HGT})
# w.savemat(savepath, 'HGT', HGT)
# w.savemat(savepath, 'N', N_PBLH_INTERP)
# w.savemat(savepath, 'n', n_PBLH_INTERP)
# w.savemat(savepath, 'P', P)
# w.savemat(savepath, 'height', height)
per_duct_h_land = per_duct_h - HGT
# per_duct_h_land = per_duct_h_land[per_duct_h_land]
per_duct_h_land[per_duct_h_land < 0] = np.nan
# w.imshow(per_duct_h_land)
