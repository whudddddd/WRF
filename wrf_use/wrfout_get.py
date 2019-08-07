from wrf import getvar, ALL_TIMES, to_np, interplevel, smooth2d, interpz3d
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import math
from scipy import interpolate


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
# P = (getvar(ncfile, 'p') + getvar(ncfile, 'PB')) * 0.1  # 气压mb
# QVAPOR = getvar(ncfile, 'QVAPOR')  # 水汽混合比
# e = QVAPOR * P / (0.622 + QVAPOR)  # 水汽压
# # N = 77.6/T*(P+(4810*e)/T)
# n = 1 + 77.6 * 10 ** (-6) * P / T + 0.372 * e / (T ** 2)  # 折射指数
# N = 10 ** 6 * (n - 1)  # 折射率
# print(N.shape)
# length = 2000  # 插值长度
# z = getvar(ncfile, 'z', units="m")
# # height = z
# N_PBLH_INTERP = np.zeros([length, height.shape[1], height.shape[2]], dtype=np.float64)  # 插值后的N
# n_PBLH_INTERP = np.zeros([length, height.shape[1], height.shape[2]], dtype=np.float64)  # 插值后的n
# # EDH = getvar(ncfile,'EDh')
# index = 0
# for i in range(0, 20000, 10):
#     N_PBLH = interplevel(N, height.data, i, missing=True)
#     N_PBLH_INTERP[index, :, :] = N_PBLH.squeeze()
#     n_PBLH = interplevel(n, height.data, i, missing=True)
#     n_PBLH_INTERP[index, :, :] = n_PBLH.squeeze()
#     index += 1
# # io.savemat("n.mat", {"n": n_PBLH_INTERP})
# # io.savemat("N.mat", {"N": N_PBLH_INTERP})
# plt.imshow(N[10, :, :])


class WrfUse:
    def __init__(self, wrf_file_path, timeidx=0):
        self.timeidx = timeidx
        self.ncfile = Dataset(wrfout_file)

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
        P = (self.get_var('p') + self.get_var('PB')) * 0.1  # 气压mb
        return P

    @property
    def get_e(self):
        """
        水汽压hpa
        :return:
        """
        QVAPOR = self.get_var('QVAPOR')
        e = QVAPOR * self.get_P / (0.622 + QVAPOR)  # 水汽压
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
        height = self.get_var('z')
        return height

    def insert_value(self, data, length=2000):
        '''

        :param length:长度

        :return:按高度插值后的量
        '''
        height = self.get_height - (self.get_height[0, :, :] - self.get_HGT)
        # height = self.get_height
        N_PBLH_INTERP = np.zeros([length, height.shape[1], height.shape[2]], dtype=np.float64)  # 插值后的N
        index = 0
        for i in range(10, 10000, 10):
            N_PBLH = interplevel(data, height.data, i, missing=0)
            N_PBLH_INTERP[index, :, :] = smooth2d(N_PBLH.squeeze(), 5)

            index += 1
        # for i in range(length):
        #     for j in range(height.shape[1]):
        #         for k in range(height.shape[2]):
        #             if N_PBLH_INTERP[i, j, k] == 0.0:
        #                 sum = 0
        #                 n = 0
        #                 # import pdb;pdb.set_trace()
        #                 if i > 0 and not N_PBLH_INTERP[i - 1, j, k]==0.0:
        #                     sum += N_PBLH_INTERP[i - 1, j, k]
        #                     n += 1
        #                 elif not N_PBLH_INTERP[i + 1, j, k]==0.0:
        #                     sum += N_PBLH_INTERP[i + 1, j, k]
        #                     n += 1
        #                 if j > 0 and not N_PBLH_INTERP[i, j - 1, k] == 0.0:
        #                     sum += N_PBLH_INTERP[i, j - 1, k]
        #                     n += 1
        #                 elif not N_PBLH_INTERP[i, j + 1, k] == 0.0:
        #                     sum += N_PBLH_INTERP[i, j + 1, k]
        #                     n += 1
        #                 if k > 0 and N_PBLH_INTERP[i, j, k - 1]==0.0:
        #                     sum += N_PBLH_INTERP[i, j, k - 1]
        #                     n += 1
        #                 elif not N_PBLH_INTERP[i, j, k + 1] == 0.0:
        #                     sum += N_PBLH_INTERP[i, j, k + 1]
        #                     n += 1
        #                 if n>0:
        #                     sum /= n
        #                 N_PBLH_INTERP[i, j, k] = sum

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

    def savemat(self, savepath, key, value):
        io.savemat(savepath + key + '.mat', {key: value})


savepath = "../savedata/"
wrfout_file = '/home/ionolab/download/wrf/WRF/run/wrfout_d01_2019-07-31_00:00:00'
w = WrfUse(wrfout_file)
# n_PBLH_INTERP = w.get_n_PBLH_INTERP
# N_PBLH_INTERP = w.get_N_PBLH_INTERP
HGT = w.get_HGT
n = w.get_n
N = w.get_N
# io.savemat('HGT.mat', {"HGT": HGT})
w.savemat(savepath, 'HGT', HGT)
w.savemat(savepath, 'N', N)
w.savemat(savepath, 'n', n)
