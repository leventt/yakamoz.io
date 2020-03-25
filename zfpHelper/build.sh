cd zfp
git checkout tags/0.5.5
git pull
cd ..

mkdir build
cd build
emcmake cmake ..
cmake --build . --parallel 4

echo "copying stuff to ../../static"
cp zfpHelper.wasm ../../static/.
cp zfpHelper.js ../../static/.

cd ..