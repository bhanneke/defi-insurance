# Local build (Overleaf compiles main.tex directly and ignores this file).
# Requires latexmk (TeX Live / MacTeX).
all:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml

.PHONY: all clean
