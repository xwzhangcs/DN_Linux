# DN_Linux

# creat build folder
mkdir build
cd build
# cmake
cmake -DCMAKE_PREFIX_PATH=/home/zhan2597/libtorch -DCUDA_HOST_COMPILER=/usr/bin/gcc-5 ..
# build our application
make
# once the build is complete, run as follows
./dn_lego_syn ../lego_model.pt ../0001_0000.json
