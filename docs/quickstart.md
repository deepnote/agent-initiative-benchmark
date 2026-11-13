# Quickstart

Use Python 3.13 on Linux or macOS, with its development headers and a C++
compiler for `hnswlib`. On macOS, run `xcode-select --install` if needed.
On Ubuntu, install `build-essential` and the headers for your Python 3.13.
Clone the repository and install:

```sh
git clone https://github.com/deepnote/agent-initiative-benchmark.git
cd agent-initiative-benchmark
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
benchmark init
```

If installation fails, see [installation help](installation.md).

`benchmark init` creates `benchmark.json` in your current folder. Edit this
file to choose an agent model and two distinct judge models.
Each profile has an API format, full endpoint, model ID and key variable name.
[Model settings](models.md) explains other providers.

Set the keys named in your profiles. For the default profiles:

```sh
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
benchmark data download
benchmark check
benchmark run --task task-01
benchmark results
```

`check` makes no model call. `run` shows your models and time budgets, then
calls the agent and judges. Scores range from 0 to 10.

Run all eight examples with `benchmark run --task all`. Use `benchmark tasks`
to see their names. Each run saves its own ID and results.

- [Dataset](data.md): local archives, existing data and storage location.
- [Results](results.md): saved messages, scores, CSV and resume.
