# Results

Show the latest run or choose one:

```sh
benchmark results
benchmark results --list
benchmark results --run RUN_ID
benchmark inspect --run RUN_ID --task task-01
benchmark results --run RUN_ID --csv scores.csv
```

`inspect` shows delivered messages and each judge's reasons. The table keeps
both scores and averages only complete, compatible pairs. A dash means no valid
score. Zero is a valid score. The mean reports how many selected tasks were
scored. Category results and the silence case are shown separately.

These eight selected examples are not a score for the full benchmark.

Continue an interrupted run:

```sh
benchmark resume --run RUN_ID
```

Completed agents and judges stay saved. An interrupted unit starts a new
attempt, keeping its earlier trace. A finished technical failure stays unscored.
To retry those failures explicitly:

```sh
benchmark resume --run RUN_ID --retry-errors
```

Use `inspect` to see saved failure details. For a rate limit (HTTP 429), wait
for the provider's limit to reset before retrying. Repeated failures may need
a higher token quota for your deployment.

A genuine model-time budget stop can be judged. Infrastructure failures do not
receive zero points. Resume uses saved model settings and rejects changed task
inputs or runtime code. For another model or configuration, start a new run.

Each `results/RUN_ID/` contains:

| Path | Contents |
| --- | --- |
| `settings.json`, `inputs/` | Models, task pins, limits and runtime identity |
| `progress.json` | Attempts started for each stage |
| `summary.json` | Current scores, coverage and errors |
| `artifacts/` | Verified agent and judge bundles, messages and traces |

Viewing results needs no API keys or company data. `--json` returns the summary
as JSON. With a custom results location, add `--results-dir PATH`. If the data
moves, resume accepts `--data-dir PATH` for the same pinned snapshot.

Exit codes: 0 for success, 1 for an input or storage error, 2 for failed or
unfinished work, 130 for interruption.

[Quickstart](quickstart.md)
