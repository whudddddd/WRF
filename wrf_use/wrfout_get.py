from wrf import getvar, ALL_TIMES, to_np, interplevel, smooth2d, rh, td, tvirtual, tk,latlon_coords
from wrf.g_temp import get_tv
from wrf.g_slp import get_slp
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import cartopy_imshow
from math import log, sqrt, atan, pi, pow
import math


class Wrf:
    def __init__(self, wrf_file_path, length, dlen, timeidx):
        self.timeidx = timeidx
        self.wrf_file_path = wrf_file_path
        self.ncfile = Dataset(wrf_file_path)
        self.length = length
        self.dlen = dlen

    @property
    def get_keys(self):

        """
        wrf输出文件包含的所有变量
        :return:
        """
        return self.ncfile.variables.keys()

    def get_var(self, key):
        """

        :param key: 变量名,字符串
        :return:
        """
        var = getvar(self.ncfile, key, timeidx=self.timeidx)
        return var.data

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
        P = (self.get_var('P') + self.get_var('PB')) * 0.01  # 气压mb

        # P = self.get_var('P')
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
        # N = 10 ** 6 * (self.get_n - 1)  # 折射率
        N = 77.6 * self.get_P / self.get_T - 5.6 * self.get_e / self.get_T + 3.75 * (10 ** 5) * self.get_e / (
                self.get_T ** 2)
        return N

    @property
    def get_M(self):

        M = self.get_N + 0.157 * self.get_height
        return M

    def get_gradient(self, x, y):
        gradients = np.zeros([1000, x.shape[1], x.shape[2]])
        for xx in range(x.shape[1]):
            for yy in range(x.shape[2]):
                coefficient = np.polyfit(x[:, xx, yy], y[:, xx, yy], 6)
                # plt.plot(x.data[:, xx, yy], y.data[:, xx, yy])
                # import pdb;pdb.set_trace()
                gradient_coefficient = np.polyder(coefficient)
                # gradient = np.polyval(gradient_coefficient, x[:, xx, yy])
                # # plt.plot(x.data[:, xx, yy], gradient)
                # gradients[:, xx, yy] = gradient
                gradient = np.polyval(gradient_coefficient, np.arange(1000))
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

        :param length:heghit
               dlen:步长
        :return:按高度插值后的量
        '''
        # height = self.get_height - (self.get_height[0, :, :] - self.get_HGT)
        height = self.get_height
        N_PBLH_INTERP = np.zeros([self.length // self.dlen, height.shape[1], height.shape[2]],
                                 dtype=np.float64)  # 插值后的N
        # import pdb;pdb.set_trace()
        index = 0
        for i in range(0, self.length, self.dlen):
            N_PBLH = interplevel(data, height, i, missing=np.nan)
            N_PBLH_INTERP[index, :, :] = smooth2d(N_PBLH, 3, cenweight=4)
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
        """
        :param savepath: 保存路径
        :param key: 保存变量名，字符串
        :param value: 保存变量
        :return:保存后的.mat文件
        """

        value = value.data
        io.savemat(savepath + key + '.mat', {key: value})

    def get_height_in(self, shape=None):
        """
        :param delen: 步长
        :param shape: 插值范围大小
        :return:插值后的高度
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
    def get_duct_h(self):
        """

        :return: wrf插值后的波导高度
        """
        N_gradient = self.get_gradient(self.get_height, self.get_N)
        # N_gradient_in = self.insert_value(N_gradient)
        # height_in = self.insert_value(self.get_height)
        # duct_h = np.zeros([N_gradient_in.shape[1], N_gradient_in.shape[2]]
        duct_h = np.zeros([N_gradient.shape[1], N_gradient.shape[2]])
        for i in range(N_gradient.shape[1]):
            for j in range(N_gradient.shape[2]):
                for k in range(1, N_gradient.shape[0]):
                    if N_gradient[k, i, j] < -0.157:
                        # duct_h[i, j] = height_in[k, i, j]
                        duct_h[i, j] = self.dlen * k
                        break
        return duct_h

    @property
    def get_sst(self):
        """

        :return: 海面压力
        """
        return self.get_var("SST")


    def savedata(self,dataname,data):
        with open(dataname+'.dat', 'w') as f:
            for i in range(data.shape[0]):
                for k in range(data.shape[2]):
                    for j in range(data.shape[1]):
                        f.write(str(data[i,j,k]))
                        f.write(',')
                    f.write('\n')
                f.write('\n')


class NpsModel(Wrf):
    def __init__(self, wrf_file_path, length, dlen, timeidx):
        Wrf.__init__(self, wrf_file_path, length, dlen, timeidx)

    def psi_t(self, x):
        """
        温度普适函数
        :param x:
        :return:温度普适函数，特征尺度
        """
        # x1 = np.cbrt(1 - 12.87 * x)
        # # import pdb;pdb.set_trace()
        # p_c = 1.5 * log((1 + x1 + x1 ** 2) / 3) - sqrt(3) * atan((1 + 2 * x1) / sqrt(3)) + pi / sqrt(3)
        # if x >= 0:
        #     p_t = -4.7 * x
        # else:
        #     p_t = 2 / (1 + x ** 2) * log((1 + (1 - 9 * x) ** 0.5) / 2) + x ** 2 / (1 + x ** 2) * p_c
        ah = bh = 5
        ch = 3
        Bh = sqrt(5)
        p_h = -bh / 2 * log(1 + ch * x + x ** 2) * (-ah / Bh + bh * ch / 2 * Bh) * (
                log((2 * x + ch * Bh) / (2 * x + ch + Bh)) - log((ch - Bh) / (ch + Bh)))
        return p_h

    def theta_z(self, z):
        pass

    def q_z(self, z):
        pass

    def qs(self, T, p):
        """
        计算饱和比湿
        :param T: 温度
        :param p: 压强
        :return: 饱和比湿
        """
        a = 17.2693882
        b = 35.86
        # es = 6.1078 * math.e ** (a * (T - 273.16) / (T - b))  # 饱和水汽压
        es = 6.112 * math.e ** ((T - 273.16) * 17.67 / (T - 273.16 + 243.5))
        res = 0.622 * es / (p - 0.378 * es)  # 饱和比湿
        # res=0.622*self.get_e/(self.get_P-0.738*self.get_e)
        return res

    def get_Tqz(self, L, z, z0t=0.002, z0q=0.002):
        """

        :param L: monin-bukhov参数
        :param z0t: 温度粗糙高度
        :param z0q: 湿度的粗糙高度
        :param z: 高度
        :return: 温度剖面和比湿剖面
        """
        T0 = self.get_sst  # 海表温度
        P0 = self.get_var('slp')  # 海表压强

        q0 = self.qs(T0, P0) * 0.98  # 海表比湿
        ##TODO
        zr = 6  # 参考高度
        Td = 0.00976  # 干绝热递减率
        ##todo 0.4
        ##todo 参考高度温度用第二eta层温度代替
        theta_r = 0.4 * (log(zr / z0t) - self.psi_t(zr / L)) ** (-1) * (self.get_T[1, :, :] - T0)  # 温度特征尺度
        q_r = 0.4 * (log(zr / z0q) - self.psi_t(zr / L)) ** (-1) * (
                self.qs(self.get_T[1, :, :], self.get_P[1, :, :]) - q0)  # 比湿特征尺度
        Tz = T0 + theta_r / 0.4 * (log(z / z0t) - self.psi_t(z / L)) - Td * z
        qz = q0 + q_r / 0.4 * (log(z / z0t) - self.psi_t(z / L))  # - Td * z
        return Tz, qz

    def T_p(self, pz):
        """
        温度转化为位温
        :param pz:
        :return:温度和位温转换
        """
        p0 = 1000
        Tz, _ = self.get_Tqz()
        t = Tz * (p0 / pz) ** 0.286
        return t

    def e(self, pz, qz):
        """
        :param qz:气压剖面
        :param pz: 比湿剖面
        :return: 水汽压剖面
        """
        epsilon = 0.62197
        res = qz * pz / (epsilon + (1 - epsilon) * qz)
        return res

    def Tv(self, z1, z2):
        """

        :param z1: 高度z1
        :param z2: 高度z2
        :return: 平均虚温
        """
        tv = get_tv(self.ncfile, timeidx=self.timeidx)  # 虚温
        tv1 = tv.data[z1, :, :]
        tv2 = tv.data[z2, :, :]
        tv_a = (tv1 + tv2) / 2
        return tv_a

    def get_pz(self, z1, z2=0):
        """

        :param z1: z1处的高度
        :param z2: z2处的高度，默认为海面高度
        :return:气压剖面
        """
        g = 0.981
        R = 287.04
        Tv = self.Tv(10, z2)
        pz2 = self.get_P[z2, :, :]
        pz1 = pz2 * math.e ** (g * (z2 - z1) / (R * Tv))
        return pz1

    def correct_N_profile(self, L, z, z0t=0.0002, z0q=0.0002, z2=0):
        """

        :param L: monin-obukhov长度（相关长度）
        :param z0t: 大气温度粗糙度高度
        :param z0q: 大气压强粗糙度高度
        :param z: 高度
        :param z2: 参考高度，默认值为海面
        :return: 修正折射误差剖面
        """
        TZ, qz = self.get_Tqz(L, z, z0t, z0q)
        # import pdb;pdb.set_trace()
        pz = self.get_pz(z, z2)
        ez = self.e(pz, qz)
        Nz = 77.6 * pz / TZ - 5.6 * ez / TZ + 3.75 * (10 ** 5) * ez / (TZ ** 2)
        Mz = Nz + 0.157 * z
        return Mz

    def eva_duct(self):
        """

        :return:蒸发波导高度
        """
        # p0=1000
        # Ra=287.05
        #
        # ##todo
        # Cpa=1004
        # A=77.6
        # B=4810
        # epsilon = 0.62197
        # c2=(p/p0)**(Ra/Cpa)*(-(A/T**2)-(2*A*B*q)/(T**3*(epsilon+(1+epsilon)*q)))
        # c1=(-)
        # zd=-(c2)
        dmap = np.zeros_like(self.get_sst)
        derivative = 1
        dz = 0.1
        z1 = 1
        z2 = z1 + dz
        while z2 < 50:  # 一般认为蒸发波导在50km以下
            derivative = abs((self.correct_N_profile(3, z2) - self.correct_N_profile(3, z1)) / dz)
            local = np.where(derivative < 0.001)
            for index in range(len(local[0])):
                if dmap[local[0][index], local[1][index]] < 0.1:  # 元素为浮点型，因此不能直接判断是否为0
                    dmap[local[0][index], local[1][index]] = z1
            if (dmap > 0).all():
                return dmap

            z1 = z2
            z2 += dz
        return dmap


savepath = "../savedata/"
wrfout_file = '/home/ionolab/download/wrfout_d01_2019-07-31_00:00:00'
w = Wrf(wrfout_file, length=500, dlen=1, timeidx=0)
npsmodel = NpsModel(wrfout_file, length=100, dlen=1, timeidx=0)
# n_PBLH_INTERP = w.get_n_PBLH_INTERP
# N_PBLH_INTERP = w.get_N_PBLH_INTERP
HGT = w.get_HGT
# n = w.get_n
# N = w.get_N
# P = w.get_P
# height = w.get_height
# N_gradient = w.get_gradient(height, N)
# # N_gradient_in = w.insert_value(N_gradient)
# per_duct_h = w.get_pre_duct_h
# per_duct_h_sea = per_duct_h
# # per_duct_h_sea[np.where(HGT > 0.01)] = np.nan
# # plt.imshow(HGT)
# # # io.savemat('HGT.mat', {"HGT": HGT})
# # w.savemat(savepath, 'HGT', HGT)
# # w.savemat(savepath, 'N', N_PBLH_INTERP)
# # w.savemat(savepath, 'n', n_PBLH_INTERP)
# # w.savemat(savepath, 'P', P)
# # w.savemat(savepath, 'height', height)


# dct_h = npsmodel.eva_duct()
# cartopy_imshow.car_imshow(dct_h)
ncfile = Dataset(wrfout_file)

# Get the sea level pressure
slp = getvar(ncfile, "slp")
# slp=w.get_var('slp')
T = w.get_T#温度
P=w.get_P#压强
N=w.get_N#折射指数
M=w.get_M#修正折射指数
e=w.get_e#水汽压
lons=w.get_var('XLONG')#经度
lats=w.get_var('XLAT')#纬度度
height = w.get_height#高度
#
# with open('lons.dat','w') as f:
#         for j in range(99):
#             for k in range(133):
#                 # import pdb;pdb.set_trace()
#                 f.write(str(lons[k,j]))
#                 f.write(',')
#             f.write('\n')
#         f.write('\n')

# T=w.insert_value(T)
# p=w.insert_value(P)
# M=w.insert_value(M)
# N=w.insert_value(N)
# e=w.insert_value(e)
# height=w.insert_value(height)
#
# w.savedata('T',T)
# w.savedata('P',P)
# w.savedata('M',M)
# w.savedata('N',N)
# w.savedata('e',e)
# w.savedata('height',height)
