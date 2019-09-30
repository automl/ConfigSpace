set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cwd=`pwd`
test_dir=$cwd/test/

cd $TEST_DIR

echo "###################"
echo "Current directory and test directory"
echo `pwd`
echo "$test_dir"
which pytest
which python
python -c "import ConfigSpace"
python -c "import ConfigSpace.configuration_space"

pytest -sv --cov=ConfigSpace $test_dir

cd $cwd

source ci_scripts/create_doc.sh $TRAVIS_BRANCH "doc_result"

if [ "$RUN_FLAKE8" == "true" ]; then
  pip install flake8 mypy
  ./ci_scripts/flake8_diff.sh
fi
