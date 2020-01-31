#!/usr/bin/env bash

# See this article
#      https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html

# install gdal and python-wrapper are not that straight forward.
# In ubuntu, you can use the repository ubuntugis

sudo add-apt-repository ppa:ubuntugis/ppa 
sudo apt-get update
sudo apt-get install gdal-bin -y
sudo apt-get install libgdal-dev -y


# Now in a suitable virtual env,  you can do the pip install as follows: 

# first get the matching gdal version
ogrinfo --version

export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# Use the right version, otherwise it may have compile error 
# pip3 install GDAL==<GDAL VERSION FROM OGRINFO>

##############################################  Example run
# root@zubuntu1804:~# ogrinfo --version
# GDAL 2.4.2, released 2019/06/28

# root@zubuntu1804:~# pip3 install GDAL==2.4.2

# Collecting GDAL==2.4.2
#  Downloading https://files.pythonhosted.org/packages/dc/d5/90339b48bdcabc76124eaa058a32d796963a05624eb418ed5ea5af7db9fa/GDAL-2.4.2.tar.gz (564kB)
# Building wheels for collected packages: GDAL
#  Running setup.py bdist_wheel for GDAL ... done
#  Stored in directory: /root/.cache/pip/wheels/00/f6/51/91bbf2ad21dac494c3d0d83231f1d6d06d797bc4ed5cacd6b8
# Successfully built GDAL
# Installing collected packages: GDAL
# Successfully installed GDAL-2.4.2


# testing with the following command, to confirm gdal is installed inside python3
$ python3 -c "import gdal; print(gdal.VersionInfo())"
2040200

#################################################
