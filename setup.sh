#!/usr/bin/env bash

# # Clone the required repositories
echo 'Cloning the required repositories...'
mkdir -p cutest
cd cutest
git clone --depth=1 https://github.com/ralna/SIFDecode ./sifdecode
git clone --depth=1 https://github.com/ralna/CUTEst ./cutest
git clone --depth=1 https://bitbucket.org/optrove/sif ./mastsif

# Install SIFDecode
echo 'Installing SIFDecode...'
cd sifdecode
meson setup builddir
meson compile -C builddir
sudo meson install -C builddir

# Install CUTEst
echo 'Installing CUTEst...'
cd ../cutest
meson setup builddir -Dmodules=false
meson compile -C builddir
sudo meson install -C builddir

# Setup Python environment
echo 'Installing Python environment...'
cd ../..
python3 -m venv .env
source .env/bin/activate
python3 -m pip install -r ./requirements.txt

# Export environment variables to activate file
# pycutest only works on Linux, MacOS and WSL
# So, only the bash activation file matters
echo 'Exporting environment variables...'
sed -i '/^export VIRTUAL_ENV/a\
export MASTSIF="\$VIRTUAL_ENV/../cutest/mastsif"\
export PYCUTEST_CACHE="\$VIRTUAL_ENV/../cutest"\
' .env/bin/activate
sed -i '/^deactivate () {/,/^}/{
/^}/i\
    unset MASTSIF\
    unset PYCUTEST_CACHE
}' .env/bin/activate

deactivate
echo 'Done!'
