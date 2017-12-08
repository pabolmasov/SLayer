# Short installation guide for Titan cluster

## load libraries; predefined in modules.sh
source modules.sh


## installing shtns
wget https://bitbucket.org/nschaeff/shtns/downloads/shtns-2.9-r597.tar.gz
tar xf shtns-2.9-r597.tar.gz
mv shtns-2.9-r597.tar.gz sthns
./configure --enable-openmp --enable-python
make
python setup.py install --user



