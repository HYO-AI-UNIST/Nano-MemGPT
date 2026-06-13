# Nano-MemGPT LaTeX Paper Draft

This directory contains an ACL-style LaTeX draft for the Nano-MemGPT study.

## Files

- `final_paper.tex`: main paper draft
- `references.bib`: bibliography entries used by the draft
- `acl.sty`: official *ACL style file from `acl-org/acl-style-files`
- `acl_natbib.bst`: official *ACL bibliography style from `acl-org/acl-style-files`
- `acl_latex_template.tex`: official *ACL example template kept as a local reference

## Build

If a TeX distribution is installed, compile from this directory with:

```bash
pdflatex final_paper
bibtex final_paper
pdflatex final_paper
pdflatex final_paper
```

The current local environment did not have `pdflatex` or `bibtex` available on `PATH`, so PDF compilation was not verified here.

## Template Notes

The draft now uses the official ACL style package:

```latex
\documentclass[11pt]{article}
\usepackage[review]{acl}
```

For a camera-ready version, change `review` to `final`. For a non-anonymous preprint with page numbers, change `review` to `preprint`.
