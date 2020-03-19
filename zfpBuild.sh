cd zfp
git checkout tags/0.5.5
git pull
mkdir build
cd build
emcmake cmake .. -DBIT_STREAM_WORD_TYPE=uint64
cmake --build . --config Release
