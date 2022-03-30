#Изменение размера директории для временных файлов (если это нужно)
mount -o remount,size=50G /tmp/

#Нарезка и ресэмплинг

#tmp (https://archive.ceda.ac.uk)
ncks -d lon,-145.,-52. -d lat,14.,71. \
/home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/cru_ts4.04.1901.2019.tmp.dat.nc \
-O /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/tmp_crop.nc

#pre (https://archive.ceda.ac.uk)
ncks -d lon,-145.,-52. -d lat,14.,71. \
/home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/cru_ts4.04.1901.2019.pre.dat.nc \
-O /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/pre_crop.nc

#awc (https://data.apps.fao.org/map/catalog/srv/rus/catalog.search#/metadata/cc0ebca9-0df9-4061-a338-6ffec9e9acb4)
gdalwarp -of "GTiff" -te -145 14 -52 71 /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/AWC_CLASS.tif \
/home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/soil_crop.tif

gdal_translate -sds -of GTiff \
-b 1 /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/pre_crop.nc \
/home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/resample.tif

python /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/soil.py

#Расчет индексов (https://climate-indices.readthedocs.io/en/latest/)

#При первом запуске: создание виртуальной оболочки
#conda create -n indices_env

#Активация виртуальной оболочки
conda activate indices_env

#При первом запуске: установка пакетов (если они еще не установлены)
#pip install climate-indices
#conda install -c conda-forge nco

#Запуск расчета. Выполняется за 30-40 мин
process_climate_indices --index palmers --periodicity monthly \
--netcdf_precip /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/pre_crop.nc \
--netcdf_temp /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/tmp_crop.nc \
--netcdf_awc /home/grlo/LG_0/wrking/dendro/cl_indexes/ceda_catalogue/soil5_rep.nc \
--output_file_base /home/grlo/LG_0/wrking/dendro/cl_indexes/res1 \
--var_name_precip pre \
--var_name_temp tmp \
--var_name_awc Band1 \
--calibration_start_year 1951 \
--calibration_end_year 2010 \
--multiprocessing all_but_one

#Перевод результатов в версию CRU TS
ncpdq -a time,lat,lon \
-v scpdsi /home/grlo/LG_0/wrking/dendro/cl_indexes/res1_scpdsi.nc \
/home/grlo/LG_0/wrking/dendro/cl_indexes/res1/computed_scpdsi.nc
