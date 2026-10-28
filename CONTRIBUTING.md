# Contributing

Create a virtual environment and install `python -m pip install -e ".[dev]"`.
Run `make check test train` from the repository root before proposing a change.
Keep test data synthetic and tiny; never commit downloaded datasets or trained binaries.
Preserve the split contract, add a regression test for changed behavior, and report
baseline and model metrics together. Do not select settings using test results.
