You are grading a research report produced by a multi-agent research system.

## Question

{{input}}

## Report

{{output}}

## Task

Judge whether the report's own claims are supported by citations or named
sources *within the report itself*. You are not given the sources the
system actually retrieved -- judge only whether the report's citation
structure is internally consistent and plausible, not whether a citation is
factually correct against the real source.

Score from 0.0 to 1.0:
- 1.0: every load-bearing claim (a specific fact, number, or named finding)
  is attributed to a source, and the cited sources are topically consistent
  with both the claim and the question.
- 0.5: some load-bearing claims are cited, others are asserted with no
  attribution at all.
- 0.0: the report makes specific factual claims with no citations
  anywhere, or cites sources that are clearly unrelated to the claim they
  are attached to.

Return the numeric score only, with one sentence of reasoning.
