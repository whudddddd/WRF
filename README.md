# WRF
wrf install and use
##WRF install 


###**1.sudo python wrfinstall.py**

###**2.修改配置文件**

sudo gedit ~/.bashrc
末尾添加

    #for zlib
    export ZLIB_HOME=/usr/local/zlib
    export LD_LIBRARY_PATH=$ZLIB_HOME/lib:$LD_LIBRARY_PATH
    #for libpng
    export ZLIB_HOME=/usr/local/libpng
    export LIBPNGLIB=/usr/local/libpng/lib
    export LIBPNGINC=/usr/local/libpng/include
    #set JASPER
    export JASPER=/usr/local/JASPER
    export JASPERLIB=/usr/local/JASPER/lib
    export JASPERINC=/usr/local/JASPER/include

修改完毕保存
激活修改配置
source ~/.bashrc

###**3.sudo python wrfinstall2.py**

###**4.安装synaptic**

sudo apt-get install synaptic	
命令行输入sudo synaptic
搜索libjpeg8，选中下边4个，点击apply	
glibc，然后就会看到结果有三个红点的glibc选项，也apply
grib2，然后就会看到结果有libgrib2c-dev/libgrib2c0d选项，也apply

###**5.gedit ~/.bashrc**
    # for WRF
    export JASPERLIB=/usr/local/JASPER/lib
    export JASPERINC=/usr/local/JASPER/include
    export NETCDF=/usr/local/NETCDF
    # for WPS
    export JASPERLIB=/usr/local/JASPER/lib
    export JASPERINC=/usr/local/JASPER/include
    export LD_LIBRARY_PATH=/usr/local/jasper/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/libpng/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/zlib/lib:$LD_LIBRARY_PATH
    
激活
source ~/.bashrc

###**6.sudo python wrfinstall3.py**

###**7.安装WPF**
sudo ./clean 
sudo ./configure 
选32
sudo ./compile em_real>&checkwrf.log

###**9. 安装WPS**
./clean 
source ~/.bashrc
sudo ./configure 
选1
sudo ./compile > checkwps.log

##WRF use

###1.静态下载数据

http://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html

###GFS数据下载

sudo wget ftp://ftpprd.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20190731/00/gfs.t00z.pgrb2.0p50.f012

###2. 解压

###3. 修改namelist.wps（参见附录namelist.wps编辑说明） 
修改时间


修改geog_data_path wps_geog 文件路径

###4. 切换到WPS目录，

./geogrid.exe,生成geo_em* 文件

###5. 在WPS目录下执行ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable

###6. 在WPS目录下执行

./ungrib.exe

./metgrid.exe >& log.metgrid

运行后生成met开头的对应时间的文件，文件名称时间与实测数据一致

###7. 切换到WRF/run目录

###8. 修改namelist.input 

修改时间，对应namelist.wps

###9. 把wps下边生成的met开头的文件考到wrf/run下边

ln -sf  ../../WPS/met_em* ./

###10. 执行./real.exe  

###11. 执行mpirun -np 8 ./wrf.exe

###12. 运行后输出文件wrfout*


***备注***

export LD_LIBRARY_PATH=/usr/local/NETCDF/lib/:$LD_LIBRARY_PATH
