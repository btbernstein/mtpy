#!/usr/bin/env bash

# See this article
#      https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html

# install gdal and python-wrapper are not that straight forward.
# In ubuntu, you can use the repository ubuntugis

sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev


# Now in a suitable virtual env,  you can do the pip install as follows: 

export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

gdalver=`ogrinfo --version`
pip install GDAL==$gdalver

#pip install GDAL==<GDAL VERSION FROM OGRINFO>

