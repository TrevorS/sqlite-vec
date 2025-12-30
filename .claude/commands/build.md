---
description: Build the sqlite-vec loadable extension
---

Build the sqlite-vec loadable extension:

1. Create the dist directory if needed
2. Compile sqlite-vec.c into dist/vec0.so
3. Report any compilation warnings or errors

Use this command:
```bash
mkdir -p dist && gcc -fPIC -shared -Wall -Wextra -Ivendor/ -O3 -lm sqlite-vec.c -o dist/vec0.so
```

If compilation fails due to missing sqlite-vec.h, create it from the template first.
