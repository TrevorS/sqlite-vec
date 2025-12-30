---
description: Regenerate sqlite-vec.h if missing or corrupted
---

The sqlite-vec.h header is generated from sqlite-vec.h.tmpl. If envsubst fails or the header is corrupted, regenerate it manually.

Read the VERSION file to get the current version, then create sqlite-vec.h with the proper version macros:

```bash
VERSION=$(cat VERSION | tr -d '\n')
MAJOR=$(echo $VERSION | cut -d. -f1)
MINOR=$(echo $VERSION | cut -d. -f2)  
PATCH=$(echo $VERSION | cut -d. -f3 | cut -d- -f1)
```

The header needs these defines:
- SQLITE_VEC_VERSION "v{VERSION}"
- SQLITE_VEC_VERSION_MAJOR {MAJOR}
- SQLITE_VEC_VERSION_MINOR {MINOR}
- SQLITE_VEC_VERSION_PATCH {PATCH}

Write the complete header file based on sqlite-vec.h.tmpl with these values substituted.
