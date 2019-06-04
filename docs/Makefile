# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ConfigSpace
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

buildapi:
	sphinx-apidoc -fePTMo apidoc ../ConfigSpace/
	for f in apidoc/*;   \
	do                   \
	  sed -i '/Submodules/d' $$f; \
	  sed -i '/----------/d' $$f; \
	  sed -i 's/    :show-inheritance:/    :show-inheritance:\n    :inherited-members:/g' $$f; \
	done;
	@echo
	@echo "Auto-generation of API documentation finished. " \
	      "The generated files are in 'apidoc/'"
		  
d:
	make doctest SPHINX_BUILD="/home/philipp/.virtualenvs/ConfigSpaceEnv/bin/python" > output_doctest.log

build_clean:
	cd /mnt/c/Users/Philipp/Documents/Code/ConfigSpace/ && \
	/home/philipp/.virtualenvs/ConfigSpaceEnv/bin/pip uninstall ConfigSpace && \
	/home/philipp/.virtualenvs/ConfigSpaceEnv/bin/python /mnt/c/Users/Philipp/Documents/Code/ConfigSpace/setup.py install && \
	  cd docs
	
b: build_clean
	make html SPHINX_BUILD="/home/philipp/.virtualenvs/ConfigSpaceEnv/bin/python"

a:
	make b
	make d


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
