SPHINXBUILD = sphinx-build
BUILDDIR = build
SPHINXOPTS =
ALLSPHINXOPTS = $(SPHINXOPTS) .

.PHONY: clean buildapi linkcheck html docs html-noexamples

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf ../build/

linkcheck:
	SPHINX_GALLERY_PLOT=False $(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

html: clean linkcheck
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	$(SPHINXBUILD) -M doctest $(ALLSPHINXOPTS) $(BUILDDIR)/html || true
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

docs: html linkcheck
