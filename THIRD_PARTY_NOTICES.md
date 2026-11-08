# Third-party licenses

Python dependencies are installed separately. Their license files and notices
remain in those distributions. `requirements.lock` pins direct and transitive
dependencies. This repository does not bundle their source or binaries.

Direct dependencies:

| Package | License |
| --- | --- |
| [DuckDB](https://github.com/duckdb/duckdb-python) | MIT |
| [FastEmbed](https://github.com/qdrant/fastembed) | Apache-2.0 |
| [hnswlib](https://github.com/nmslib/hnswlib) | Apache-2.0 |
| [HTTPX](https://github.com/encode/httpx) | BSD-3-Clause |
| [ijson](https://github.com/ICRAR/ijson) | BSD-3-Clause AND ISC |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause, with separately licensed bundled components |
| [pypdf](https://github.com/py-pdf/pypdf) | BSD-3-Clause |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | BSD-3-Clause |

NumPy distributions include additional component notices. Preserve dependency
licenses and notices if you redistribute installed packages.

Search uses [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5),
licensed under MIT, through Qdrant's
[quantized ONNX model](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55).
The setup command downloads the pinned weights separately.

The PageMantle dataset and its bundled manifest have separate attribution in
[NOTICE](NOTICE).
