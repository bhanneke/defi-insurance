# Submissions

Frozen artifacts per journal submission — one subfolder per event, never
edited after the fact:

```
submissions/
  2026-MS-initial/        # example: first Management Science submission
    manuscript.pdf        # exact PDF submitted
    paper-src.zip         # snapshot of paper/ at submission
    pipeline-commit.txt   # git SHA of the pipeline that produced the numbers
    cover-letter.pdf
  2027-MS-R1/             # revision round: + response-to-referees.pdf
```

The living manuscript is `paper/` (synced to Overleaf); the code that
generates every number and figure is `pipeline/`. Record the pipeline
commit SHA in each snapshot so any submitted number can be reproduced.
