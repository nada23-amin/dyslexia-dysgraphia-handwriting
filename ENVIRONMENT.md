# Runtime environment record

The `requirements.txt` file declares supported dependency ranges; it is not a fully pinned lockfile.
For each final experiment run, record the actual interpreter and resolved packages from the same
virtual environment used to run the scripts:

```bash
python --version
python -m pip --version
python -m pip freeze > outputs/environment-freeze.txt
```

Archive `environment-freeze.txt` together with the run outputs, split manifests, and manuscript
version. Do not substitute a freeze file from a different environment or run. Runtime versions
should be reported from the actual environment rather than inferred from the allowed ranges in
`requirements.txt`.
