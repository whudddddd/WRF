from wrf import getvar, ALL_TIMES
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import scipy.io as io

e_s0 = 6.1078  # 0度时饱和水汽压
wrfout_file = '/home/ionolab/download/wrf/WRF/run/wrfout_d01_2019-07-31_00:00:00'
ncfile = Dataset(wrfout_file)
key = ncfile.variables.keys()
# slp = getvar(ncfile, 'slp')
# print(key)
times = getvar(ncfile, 'times', timeidx=ALL_TIMES)#时间
XLAT = getvar(ncfile, 'XLAT',timeidx=ALL_TIMES)#经度
XLONG = getvar(ncfile, 'XLONG')#纬度
height = getvar(ncfile, 'height')#高度
T = getvar(ncfile, 'T') + 300  # 温度
P = (getvar(ncfile, 'p') + getvar(ncfile, 'PB')) * 0.1  # 气压mb
QVAPOR = getvar(ncfile, 'QVAPOR')  # 水汽混合比
e = QVAPOR * P / (0.622 + QVAPOR)  # 水汽压
# N = 77.6/T*(P+(4810*e)/T)
n = 1 + 77.6 * 10 ** (-6) * P / T + 0.372 * e / (T ** 2)  # 折射指数
N = 10 ** 6 * (n - 1)  # 折射率
print(N.shape)
# io.savemat("n.mat", {"n": n})
# io.savemat("N.mat", {"N": N})
plt.imshow(N[20, :, :])
