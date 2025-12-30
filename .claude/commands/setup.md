---
description: Set up the development environment with all dependencies
---

Set up the sqlite-vec development environment:

1. Install system dependencies:
```bash
apt-get update && apt-get install -y sqlite3 gettext-base
```

2. Install Python test dependencies:
```bash
pip install pytest syrupy numpy tqdm
```

3. Generate the header file if needed (run /fix-header if this fails):
```bash
make sqlite-vec.h 2>/dev/null || echo "Header generation failed - run /fix-header"
```

4. Build the extension:
```bash
make loadable
```

5. Verify the build:
```bash
python3 -c "import sqlite3; db = sqlite3.connect(':memory:'); db.enable_load_extension(True); db.load_extension('dist/vec0'); print('Build successful!')"
```

After setup, run `/test` to verify everything works.
