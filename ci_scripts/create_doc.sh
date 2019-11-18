#!/usr/bin/env sh

set -euo pipefail

# Check if DOCPUSH is set
if ! [[ -z ${DOCPUSH+x} ]]; then

    if [[ "$DOCPUSH" == "true" ]]; then

        # install documentation building dependencies
        pip install --upgrade matplotlib pillow sphinx sphinx_bootstrap_theme

        # $1 is the branch name
        # $2 is the global variable where we set the script status

        # delete any previous documentation folder
        if [ -d docs/$1 ]; then
            rm -rf docs/$1
        fi

        # create the documentation
        cd docs && make html 2>&1

        # Run the doctests from the documentation
        make doctest

        # create directory with branch name
        # the documentation for dev/stable from git will be stored here
        mkdir $1

        # get previous documentation from github
        git clone https://github.com/automl/ConfigSpace.git --branch gh-pages --single-branch

        # copy previous documentation
        cp -r ConfigSpace/. $1
        rm -rf ConfigSpace

        # if the documentation for the branch exists, remove it
        if [ -d $1/$1 ]; then
            rm -rf $1/$1
        fi

        # copy the updated documentation for this branch
        mkdir $1/$1
        cp -r build/html/. $1/$1


        if ! { [ $1 = "master" ]; }; then
            { echo "Not one of the allowed branches"; exit 0; }
        fi

        # takes a variable name as an argument and assigns the script outcome to a
        # variable with the given name. If it got this far, the script was successful
        function set_return() {
            # $1 is the variable where we save the script outcome
            local __result=$1
            local  status='success'
            eval $__result="'$status'"
        }

        set_return "$2"
    fi
fi
# Workaround for travis failure
set +u