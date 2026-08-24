# Findings: single-agent vs. multi-agent AI architectures

**Goal:** Explain what distinguishes single-agent from multi-agent AI
architectures and when the literature says each tends to work better.

## Single-agent architectures

A single-agent design uses one language model to plan, reason, and act on a
task end to end, calling tools directly rather than delegating sub-tasks to
another model. According to *"The Landscape of Emerging AI Agent
Architectures for Reasoning, Planning, and Tool Calling: A Survey"*
(Masterman, Besen, Sawtell & Chao, 2024), single-agent architectures excel
when a problem is well-defined and does not require feedback from other
agents -- the same source, `2404.11584v1.pdf`.

## Multi-agent architectures

Multi-agent designs coordinate several agents or personas, each carrying
its own role and tool access, to divide labor and combine feedback across
participants. The same survey frames this as trading extra coordination
overhead for the ability to specialize each participant's role, and notes
that solutions using multiple distinct personas can outperform
single-agent chain-of-thought prompting on some tasks.

## When each tends to work better

- Single-agent: well-scoped tasks with a clear success criterion, where the
  coordination overhead of a multi-agent design would not pay for itself.
- Multi-agent: complex tasks that benefit from intelligent division of
  labor by skill, or where independent verification (a second agent
  checking the first's work) reduces the chance of an unnoticed error.

## Sources

- Masterman, Besen, Sawtell & Chao, "The Landscape of Emerging AI Agent
  Architectures for Reasoning, Planning, and Tool Calling: A Survey" (2024),
  `2404.11584v1.pdf`, sections 2-4.
