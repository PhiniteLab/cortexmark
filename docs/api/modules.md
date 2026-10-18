# API Reference

This page is split into two groups. **Core pipeline modules** are reachable from the
`cortexmark` console script — either directly from `cortexmark run` (the primary bundle
artifacts) or from a `cortexmark <stage>` subcommand / the legacy `python -m
cortexmark.run_pipeline` runner. **Standalone utilities** have no importer anywhere in
`cortexmark/` and are not reached by any `cortexmark ...` invocation; they are usable only
as a library import or via their own `python -m cortexmark.<module>` entry point.

## Core pipeline modules

::: cortexmark.cli
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.common
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.paths
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.pdfutil
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.convert
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.ocr
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.clean
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.chunk
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.page_ir
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.consensus
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.metadata
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.topics
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.grounding
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.semantic_chunk
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.cross_ref
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.algorithm_extract
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.notation_glossary
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.scientific_ir
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.citation_ir
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.qa_pipeline
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.formula_validate
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.scientific_qa
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.citation_context
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.rag_export
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.render_templates
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.run_pipeline
    options:
      show_root_heading: true
      show_source: true

## Standalone utilities (not wired into any entrypoint)

`cortexmark.cleanup` implements the repository's `make clean-outputs` and `make clean-local-artifacts` maintenance commands. These commands honor `CONFIG` and validate project/data boundaries before deleting generated files; see [Configuration](../guide/configuration.md#clean-generated-files).

::: cortexmark.cleanup
    options:
      show_root_heading: true
      show_source: true

---

These modules are not imported by `cli.py` or `run_pipeline.py` and are not reached by
`cortexmark run`, any `cortexmark <stage>` subcommand, or the legacy pipeline. Each carries
its own `if __name__ == "__main__":` block, so it can still be run directly with `python -m
cortexmark.<module>`, and `cortexmark.plugin` is additionally documented on the
[Plugin System](../architecture/plugins.md) page as an experimental, unwired hook mechanism.

::: cortexmark.plugin
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.parallel
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.ocr_quality
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.figures
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.formula_score
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.doc_type
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.citations
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.multi_format
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.ghpages
    options:
      show_root_heading: true
      show_source: true

---

::: cortexmark.diff
    options:
      show_root_heading: true
      show_source: true
