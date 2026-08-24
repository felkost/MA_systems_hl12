You are grading a research report produced by a multi-agent research system.
The system's own internal Critic agent already approved this report before
it was delivered.

## Question

{{input}}

## Report

{{output}}

## Task

Judge whether this report reads as a genuinely complete, coherent answer --
the kind of report a critic should approve -- rather than an unresolved
draft, a mid-revision fragment, or a report that trails off without
reaching a conclusion.

Return `true` if the report is coherent and complete enough that approving
it was justified. Return `false` if the report shows signs it should have
been sent back for another revision: missing sections, an abrupt cutoff, a
conclusion that contradicts the body, or visible placeholder text.

Return the boolean only, with one sentence of reasoning.
