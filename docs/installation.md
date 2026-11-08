# Installation help

`hnswlib` builds a small C++ extension. It needs a compiler and the development
headers for Python 3.13.

On macOS, an `unknown architecture arm64e.x1` or `tapi error` during linking
can mean the compiler and selected SDK are incompatible. Choose a compatible
SDK already installed on your machine, then repeat installation. For example,
if you have the 26.5 SDK:

```sh
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
python -m pip install --require-hashes -r requirements.lock
```

This selects the SDK for your shell without changing the system configuration.

[Quickstart](quickstart.md)
