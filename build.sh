# conda env create -f environment.yml
# conda activate yakamoz
# python torchScriptGen.py

# cd zfp
# git checkout tags/0.5.5
# git pull

# cd ..
mkdir build
cd build
emcmake cmake ..
cmake --build . --parallel 4

echo "copying stuff to ../static"
cp zfp/bin/zfp.wasm ../static/zfp/.
cp zfp/bin/zfp.js ../static/zfp/.
cp zfpHelper.wasm ../static/.
cp zfpHelper.js ../static/.
