# Unverified entries

None. All 38 fact sheets in `research/` (excluding `_index.md`) plus the 6
additional canonical tool/method papers required by the methodology (SHAP,
Integrated Gradients, DistilBERT, TinyBERT, Liao et al. 2020, Sharafaldin et
al. 2018) were successfully verified against Crossref or the arXiv API in
this session and added to `refs.bib` — see `verification-log.md` for the
full per-entry method/status table.

Two fact sheets originally carried `DOI: [VERIFICAR]`
(`2023-meng-netgpt.md`, `2025-cui-trafficllm.md`) because no journal DOI
exists for either paper. Both are legitimate arXiv preprints, independently
confirmed via the arXiv API (title/author/date match), so they were added to
`refs.bib` as `@misc` preprint entries and the two fact sheets were updated
in place to record the confirmed arXiv DOI instead of `[VERIFICAR]`. This is
not the same as an unresolved/unverifiable reference — no paper was excluded
from the bibliography.

No entry in this pass met the "not found on Crossref, title search
inconclusive" criterion that would require exclusion from `refs.bib`.
