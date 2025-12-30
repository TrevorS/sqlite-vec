---
description: Run the sqlite-vec test suite
---

Run the sqlite-vec Python integration tests:

1. First ensure the extension is built: `make loadable` or run `/build`
2. Run the main test suite, skipping Python 3.12-specific tests:

```bash
python3 -m pytest -vv tests/test-loadable.py -k "not test_vec_npy_each_errors_files"
```

3. If all tests pass, run the additional test suites:

```bash
python3 -m pytest -vv tests/test-metadata.py tests/test-partition-keys.py tests/test-auxiliary.py tests/test-general.py
```

Expected results:
- test-loadable.py: 42+ passed, 4 skipped
- test-metadata.py: 14 passed  
- test-partition-keys.py: 4 passed
- test-auxiliary.py: 6 passed
- test-general.py: 2 passed

Report any failures with their error messages.
