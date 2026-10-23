<p class="small">Research prototype report, revision 16 · generated from saved records by
<code>scripts/build_report_round14.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; Part A is a manuscript
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 14: the manuscript draft
(L37; D110, D111, D131) followed by its verification, decisions and tests. Part B is revision 15's Part A (Round 13);
Part C is revision 14's Part A (Round 12); Parts D–E are the L39 and L38 parts; Parts F–P are the round-11 to round-1
reports.</p>

## 0. Round-14 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Deliverable | The manuscript draft below (also `docs/manuscript_round14.md` and `reports/manuscript_round14.pdf`, {{V:ms_pages}} pages): framing (a) = Rounds 10–13 (direct functional KAN survival models against AFCM and other functional baselines), framing (b) = Rounds 1–9 plus the Round-12 S-Int/P0 re-test (KAN blocks inside a functional autoencoder) |
| New analyses | **None.** Every number is read from a saved record; the only new record is a descriptive cohort count (`reports/round14/cohort_summary.json`, no model fitted), which reproduces the D129 counts |
| Number provenance | {{V:ms_values}} values logged with their sources; all re-read and re-formatted independently; the manuscript re-renders byte for byte; every other number is a registered design constant or convention in a registered context ({{V:ms_check}}) |
| References | {{V:ms_refs}} references, all cited; the method references that R1.17 had listed as unverified were checked against Crossref, JMLR or arXiv (`docs/references_round14.md`) |
| Claims | As D111: no novelty or "first" claim; no superiority claim; H-R12 reported as inconclusive against an unvalidated comparator |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the owner needs to do

1. **Complete the author block**, affiliations, funding, competing interests and author contributions (left as
   placeholders; nothing was invented).
2. **Decide on the target venue** and its format; the draft uses a generic structure (abstract, six numbered sections,
   declarations, references, provenance appendix).
3. **Review three interpretations** that the draft marks as interpretations rather than findings: why small cohorts
   favour simple baselines (§4.2), why the unconstrained refit fails (§3.6), and the framing-(b) statement that the
   autoencoder's KAN edges are not clinically readable (§3.7).

# Part A: Manuscript draft (Round 14)

## {{V:ms_title}} {#manuscript-title}

{{MANUSCRIPT}}

## R14.1 Owner decisions (D131) and fact check

- **Approval.** The owner approved Round 13 and report revision 15 and confirmed L36 and L41 closed.
- **Scope.** A self-contained manuscript under D110's two-part framing and D111's claim policy, embedded here as
  Part A; Parts B–P are revision 15's Parts A–O, unchanged.
- **Fact check of the owner's instruction.** Every number quoted in the instruction matches the records to its stated
  precision; the clarifications (orientation of one contrast, rounding of two values, the finite-refit averaging and
  two statements worded as interpretations) are recorded in D131 (R14.3).

## R14.2 Manuscript verification

{{T:ms_verification}}

## R14.3 Decisions (D131–D132)

{{T:decisions}}

## R14.4 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R14.5 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–P: historical records

Part B is revision 15's Part A (Round 13); Part C is revision 14's Part A (Round 12); Parts D and E are the L39 and
L38 parts; Parts F–P are the round-11 to round-1 reports, each rendered from its own records. Embedded living
documents show their current version.
