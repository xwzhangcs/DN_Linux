# DN_Linux

# creat build folder
mkdir build
cd build
# cmake
cmake -DCMAKE_PREFIX_PATH=/home/zhan2597/libtorch -DCUDA_HOST_COMPILER=/usr/bin/gcc-5 ..
# build our application
make
# once the build is complete, it will generate exe file in build\Release directory
./dn_lego_syn ../lego_model.pt ../data/1.png ../data/1_output.png
