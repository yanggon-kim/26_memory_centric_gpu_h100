# Publication-only index reader

The checkpoint secret gate now reads immutable staged Git objects through one persistent `git cat-file --batch` reader rather than launching one `git show` process per staged path. NUL-delimited raw diff records retain unusual path names; deleted paths have no new object. Both candidate-file and staged-object secret scans remain enforced. Numerical scripts, trial counts and scientific gates are unchanged by this publication helper update.

Three CPU contracts passed in0.261s: staged secret remains rejected after the working file is cleaned; deletions/renames/newline paths/binary blobs preserve index semantics and a newly staged synthetic secret is rejected; empty index succeeds. No real credential was used or printed. Whole-checkpoint timing remains to be observed; no speedup is claimed from the CPU contracts alone.
