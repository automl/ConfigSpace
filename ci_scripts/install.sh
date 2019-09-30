#!/usr/bin/env sh
# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
cd ..
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip
source activate testenv

pip install codecov pytest pytest-cov cython

if [[ "$INSTALL_FROM_SDIST" == "true" ]]; then
    python setup.py sdist
    # Find file which was modified last as done in https://stackoverflow.com/a/4561987
    dist=`find dist -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
    echo "Installing $dist"
    pip install "$dist"
else
    python setup.py install
fi

echo "###################"
echo "conda list"
conda list
