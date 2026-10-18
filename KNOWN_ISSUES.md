# Known issues

This early preview is published with the unresolved findings below from a previous audit. The IDs, priorities, and affected components are retained for tracking. These are prior reported findings, not new validation results for the publication snapshot; cleanup and repository migration do not establish fixes. All entries remain unresolved at publication until a scoped fix and regression verification explicitly close them.

P1, P2, and P3 are the priorities supplied by that audit, in descending order. Line numbers are omitted because source locations may move.

| ID | Priority | Affected components | Previously reported issue |
|---|---|---|---|
| R01 | P1 | `multi_format.py`, `ghpages.py` | HTML event-attribute injection |
| R02 | P1 | `clean.py` | Display-math delimiters deleted |
| R03 | P1 | `page_ir.py` | Sparse digital text lost from IR |
| R04 | P1 | `page_ir.py` | Rotated-page evidence boxes are wrong |
| R05 | P1 | `rag_export.py` | Document record IDs collide across nested paths |
| R06 | P1 | `rag_export.py`, `semantic_chunk.py` | Scientific-object IDs collide across sibling chunks |
| R07 | P1 | `grounding.py` | Grounding approves incorrect identifiers and names |
| R08 | P2 | `cli.py` | raw.pdf destroys its raw artifact |
| R09 | P2 | `convert.py` | New word repair modifies code and math |
| R10 | P2 | `convert.py` | New word repair emits partial letter runs and misses punctuation |
| R11 | P2 | `rag_export.py` | Resolved graph edges point to missing exported objects |
| R12 | P2 | `rag_export.py` | RAG export destroys code indentation |
| R13 | P2 | `semantic_chunk.py` | One-line proof state swallows a later proof |
| R14 | P2 | `semantic_chunk.py` | Inferred provenance becomes explicit after serialization |
| R15 | P2 | `chunk.py` | Chunk character bound is violated |
| R16 | P2 | `scientific_qa.py` | Missing scientific cross-references evade QA |
| R17 | P2 | `scientific_qa.py` | Recovered equations omitted from fidelity denominator |
| R18 | P2 | `citations.py` | Citation identities collide after path slugification |
| R19 | P2 | `clean.py` | Table normalization changes fenced code |
| R20 | P2 | `metadata.py` | Markdown abstract headings are ignored |
| R21 | P2 | `ghpages.py` | Site output filenames collide |
| R22 | P2 | `ghpages.py` | Nested site navigation resolves incorrectly |
| R23 | P2 | `parallel.py` | Parallel CLI returns success for complete failure |
| R24 | P2 | `common.py` | Independent manifest writers lose cache records |
| R25 | P2 | `convert.py`, `common.py` | Output collision leaves stale cache content |
| R26 | P2 | `consensus.py`, `cli.py` | Consensus can assess text different from delivered Markdown |
| R27 | P2 | `ocr.py`, `page_ir.py` | Empty-page OCR eligibility diverges across artifacts |
| R28 | P2 | `diff.py` | Diff counts and unified output omit changes |
| R29 | P3 | `citation_context.py` | Co-citation counts collapse repeated sentence occurrences |

For classroom work, use Markdown and JSON outputs until the HTML event-attribute injection issue R01 is fixed. Do not use HTML exports for classroom sharing while this issue remains open.

Review extracted text, mathematics, tables, code, and citations against the original PDF. Treat diagnostic scores as heuristics. Inspect retrieval identities and references before indexing. See the [student workflow](https://github.com/PhiniteLab/cortexmark#convert-your-papers).
