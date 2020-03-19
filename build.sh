conda env create -f environment.yml
conda activate yakamoz
python torchScriptGen.py

cd zfp
git checkout tags/0.5.5
git pull

cd ..
mkdir build
cd build
emcmake cmake ..
cmake --build . --config Release

cp zfp/bin/zfp.wasm ../static/.
cp zfp/bin/zfp.js ../static/.
