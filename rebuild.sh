make clean
rm -rf build
make -j 20
python setup.py build
python setup.py install
