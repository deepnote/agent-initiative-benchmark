# Dataset

Download the [PageMantle Company Dataset](https://huggingface.co/datasets/deepnote/pagemantle-company-dataset)
from the repository folder:

```sh
benchmark data download
benchmark data check
```

Data goes into `data/pagemantle/`. All tasks share this folder. Choose another
location with `--dir /path/to/data` on either command.
With a custom location, also set `data_dir` in `benchmark.json` to that folder.

The download is about 6 GB. Allow 13.1 GB free during setup. Data and search
weights use about 10.2 GB afterward. Each service archive is checked, extracted
and removed before the next one. Run the same command after an interruption.
Completed services stay in place and partial downloads resume.

Already have the release archives? Keep `manifest.json`, `file-index.jsonl.gz`
and `data/*.tar.gz` together, then use:

```sh
benchmark data prepare --from /path/to/release
```

Already extracted them? Keep the manifest and file index beside the service
folders. Prepare search weights and reuse the files in place:

```sh
benchmark data prepare --from /path/to/extracted --dir /path/to/extracted
```

`data check` verifies changed files and reuses prior hash checks for unchanged
files. Add `--full` to read and hash every file again. `data download --dry-run`
shows sizes and the destination without downloading anything.

The dataset uses a [fixed revision](https://huggingface.co/datasets/deepnote/pagemantle-company-dataset/tree/eb412af9643bd081b7ec781d22305bbe66ba8fbf).
Dataset and search-model checksums are pinned in the installed `assets.lock.json`.
For a Hugging Face mirror, pass `--repository ORG/DATASET --revision FULL_COMMIT`.
The files must match the same checksums. No Hugging Face account is needed for
public downloads. Search weights download separately from their pinned model
repository. To reuse them offline, add `--search-from /path/to/model-files`.

The PageMantle Company Dataset by Deepnote is licensed separately under
[CC BY 4.0](../LICENSES/PageMantle-CC-BY-4.0.txt). See [NOTICE](../NOTICE).

[Quickstart](quickstart.md)
