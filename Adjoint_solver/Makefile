# Makefile for the analysis reports. Run from the Adjoint_solver/ directory.
#
#   make                 compile every report PDF from existing data
#   make REPORT          compile analysis/REPORT.pdf        (e.g. make gradient)
#   make REPORT-data     (re)run that report's Julia study  (e.g. make gradient-data)
#   make REPORT-all      rerun the study, then compile      (e.g. make gradient-all)
#   make data            rerun every report's Julia study
#   make all             rerun every study, then compile every PDF
#   make clean           remove the build folder and all compiled PDFs
#   make distclean       also remove the generated *_data/ folders
#
# REPORT is one of: gradient hessian optimizers regularization.
# LaTeX aux/logs and Julia run logs are kept out of the source tree, in build/
# (gitignored); only the finished PDFs are copied back next to their .tex. A report
# whose .tex has no \documentclass yet (an empty stub) is skipped, not an error.

JULIA   ?= julia
LATEX   ?= pdflatex

TEXDIR  := analysis
BUILD   := $(CURDIR)/build
REPORTS := gradient hessian optimizers regularization

# Convention: report NAME has source analysis/NAME.tex, script analysis/NAME.jl,
# and data analysis/NAME_data/. Adding a report is just appending its name above.

LATEXFLAGS := -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)
PDFS       := $(addprefix $(TEXDIR)/,$(addsuffix .pdf,$(REPORTS)))

.PHONY: all pdf data clean distclean $(REPORTS) \
        $(addsuffix -data,$(REPORTS)) $(addsuffix -all,$(REPORTS))

pdf: $(PDFS)
all: data pdf

# Compile a report twice so cleveref/hyperref cross-references resolve. The data
# files are prerequisites (via secondary expansion of the %_data/ folder), so the
# PDF rebuilds when its tables/figures change. Reports use \IfFileExists
# placeholders, so they compile even before their data has been generated.
.SECONDEXPANSION:
$(TEXDIR)/%.pdf: $(TEXDIR)/%.tex $$(wildcard $(TEXDIR)/$$*_data/*) | $(BUILD)
	@if grep -q '\\documentclass' $<; then \
	  ( cd $(TEXDIR) && $(LATEX) $(LATEXFLAGS) $*.tex && $(LATEX) $(LATEXFLAGS) $*.tex ) \
	    && cp $(BUILD)/$*.pdf $@; \
	else \
	  echo ">> skipping $*: $< has no \\documentclass yet"; \
	fi

# Per-report convenience targets: `make gradient`, `make gradient-data`, `make gradient-all`.
define REPORT_rules
$(1): $(TEXDIR)/$(1).pdf

$(1)-data: | $$(BUILD)
	$$(JULIA) --project=. $$(TEXDIR)/$(1).jl 2>&1 | tee $$(BUILD)/$(1)_julia.log

$(1)-all: $(1)-data
	$$(MAKE) $$(TEXDIR)/$(1).pdf
endef
$(foreach r,$(REPORTS),$(eval $(call REPORT_rules,$(r))))

data: $(addsuffix -data,$(REPORTS))

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) $(PDFS)

distclean: clean
	rm -rf $(addprefix $(TEXDIR)/,$(addsuffix _data,$(REPORTS)))
