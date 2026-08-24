You are grading a research report produced by a multi-agent research system.

## Question

{{input}}

## Report

{{output}}

## Task

Judge how directly and completely the report answers the question asked --
not whether it is well-written, not whether it is grounded, only whether it
is *on-topic and complete relative to what was asked*.

Score from 0.0 to 1.0:
- 1.0: the report addresses every part of the question, with no
  significant tangent or omission.
- 0.5: the report addresses the question but is partial, or spends a
  significant share of its length on material unrelated to the question.
- 0.0: the report does not answer the question asked, or answers a
  different question entirely.

Return the numeric score only, with one sentence of reasoning.
