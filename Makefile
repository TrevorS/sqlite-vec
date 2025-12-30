.DEFAULT_GOAL := help

.PHONY: help all loadable static cli clean setup test test-all test-unit test-property \
        asan coverage test-asan test-coverage fuzz-build fuzz-vector fuzz-knn quality \
        format lint analyze analyze-tidy test-valgrind test-strict install uninstall wasm

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

VERSION=$(shell cat VERSION)

INSTALL_LIB_DIR = /usr/local/lib
INSTALL_INCLUDE_DIR = /usr/local/include
INSTALL_BIN_DIR = /usr/local/bin

ifndef CC
CC=gcc
endif
ifndef AR
AR=ar
endif

# Strict warnings for high code quality
# Note: -Wconversion is omitted because SQLite's API uses int for sizes
# while C standard library uses size_t, causing 100+ false positives
WARNINGS = \
	-Wall -Wextra -Wpedantic \
	-Wshadow \
	-Wdouble-promotion \
	-Wformat=2 \
	-Wundef \
	-Wcast-qual \
	-Wcast-align \
	-Wwrite-strings \
	-Wstrict-prototypes \
	-Wold-style-definition \
	-Wmissing-prototypes \
	-Wredundant-decls \
	-Wnested-externs \
	-Winit-self \
	-Wswitch-enum

# Strict warnings for quality checks (same as WARNINGS now)
WARNINGS_STRICT = $(WARNINGS)

ifeq ($(shell uname -s),Darwin)
CONFIG_DARWIN=y
else ifeq ($(OS),Windows_NT)
CONFIG_WINDOWS=y
else
CONFIG_LINUX=y
endif

ifdef CONFIG_DARWIN
LOADABLE_EXTENSION=dylib
endif

ifdef CONFIG_LINUX
LOADABLE_EXTENSION=so
CFLAGS += -lm
endif

ifdef CONFIG_WINDOWS
LOADABLE_EXTENSION=dll
endif


ifndef OMIT_SIMD
	ifeq ($(shell uname -sm),Darwin x86_64)
	CFLAGS += -mavx -DSQLITE_VEC_ENABLE_AVX
	endif
	ifeq ($(shell uname -sm),Darwin arm64)
	CFLAGS += -mcpu=apple-m1 -DSQLITE_VEC_ENABLE_NEON
	endif
	ifeq ($(shell uname -sm),Linux x86_64)
	CFLAGS += -mavx -DSQLITE_VEC_ENABLE_AVX
	endif
endif

ifdef USE_BREW_SQLITE
	SQLITE_INCLUDE_PATH=-I/opt/homebrew/opt/sqlite/include
	SQLITE_LIB_PATH=-L/opt/homebrew/opt/sqlite/lib
	CFLAGS += $(SQLITE_INCLUDE_PATH) $(SQLITE_LIB_PATH)
endif

prefix=dist
$(prefix):
	mkdir -p $(prefix)

TARGET_LOADABLE=$(prefix)/vec0.$(LOADABLE_EXTENSION)
TARGET_STATIC=$(prefix)/libsqlite_vec0.a
TARGET_STATIC_H=$(prefix)/sqlite-vec.h
TARGET_CLI=$(prefix)/sqlite3

loadable: $(TARGET_LOADABLE) ## Build loadable extension (.so/.dylib)
static: $(TARGET_STATIC) ## Build static library
cli: $(TARGET_CLI) ## Build sqlite3 CLI with vec0

all: loadable static cli ## Build all targets

OBJS_DIR=$(prefix)/.objs
LIBS_DIR=$(prefix)/.libs
BUILD_DIR=$(prefix)/.build

$(OBJS_DIR): $(prefix)
	mkdir -p $@

$(LIBS_DIR): $(prefix)
	mkdir -p $@

$(BUILD_DIR): $(prefix)
	mkdir -p $@


$(TARGET_LOADABLE): sqlite-vec.c sqlite-vec.h $(prefix)
	$(CC) \
		-fPIC -shared \
		$(WARNINGS) \
		-Ivendor/ \
		-O3 \
		$(CFLAGS) \
		$< -o $@

$(TARGET_STATIC): sqlite-vec.c sqlite-vec.h $(prefix) $(OBJS_DIR)
	$(CC) -Ivendor/ $(CFLAGS) -DSQLITE_CORE -DSQLITE_VEC_STATIC \
	-O3 -c  $< -o $(OBJS_DIR)/vec.o
	$(AR) rcs $@ $(OBJS_DIR)/vec.o

$(TARGET_STATIC_H): sqlite-vec.h $(prefix)
	cp $< $@


$(OBJS_DIR)/sqlite3.o: vendor/sqlite3.c $(OBJS_DIR)
	$(CC) -c -g3 -O3 -DSQLITE_EXTRA_INIT=core_init -DSQLITE_CORE -DSQLITE_ENABLE_STMT_SCANSTATUS -DSQLITE_ENABLE_BYTECODE_VTAB -DSQLITE_ENABLE_EXPLAIN_COMMENTS -I./vendor $< -o $@

$(LIBS_DIR)/sqlite3.a: $(OBJS_DIR)/sqlite3.o $(LIBS_DIR)
	$(AR) rcs $@ $<

$(BUILD_DIR)/shell-new.c: vendor/shell.c $(BUILD_DIR)
	sed 's/\/\*extra-version-info\*\//EXTRA_TODO/g' $< > $@

$(OBJS_DIR)/shell.o: $(BUILD_DIR)/shell-new.c $(OBJS_DIR)
	$(CC) -c -g3 -O3 \
		-I./vendor \
		-DSQLITE_ENABLE_STMT_SCANSTATUS -DSQLITE_ENABLE_BYTECODE_VTAB -DSQLITE_ENABLE_EXPLAIN_COMMENTS \
		-DEXTRA_TODO="\"CUSTOMBUILD:sqlite-vec\n\"" \
		$< -o $@

$(LIBS_DIR)/shell.a: $(OBJS_DIR)/shell.o $(LIBS_DIR)
	$(AR) rcs $@ $<

$(OBJS_DIR)/sqlite-vec.o: sqlite-vec.c $(OBJS_DIR)
	$(CC) -c -g3 -Ivendor/ -I./ $(CFLAGS) $< -o $@

$(LIBS_DIR)/sqlite-vec.a: $(OBJS_DIR)/sqlite-vec.o $(LIBS_DIR)
	$(AR) rcs $@ $<


$(TARGET_CLI): sqlite-vec.h $(LIBS_DIR)/sqlite-vec.a $(LIBS_DIR)/shell.a $(LIBS_DIR)/sqlite3.a examples/sqlite3-cli/core_init.c $(prefix)
	$(CC) -g3  \
	-Ivendor/ -I./ \
	-DSQLITE_CORE \
	-DSQLITE_VEC_STATIC \
	-DSQLITE_THREADSAFE=0 -DSQLITE_ENABLE_FTS4 \
	-DSQLITE_ENABLE_STMT_SCANSTATUS -DSQLITE_ENABLE_BYTECODE_VTAB -DSQLITE_ENABLE_EXPLAIN_COMMENTS \
	-DSQLITE_EXTRA_INIT=core_init \
	$(CFLAGS) \
	-ldl -lm \
	examples/sqlite3-cli/core_init.c $(LIBS_DIR)/shell.a $(LIBS_DIR)/sqlite3.a $(LIBS_DIR)/sqlite-vec.a -o $@


sqlite-vec.h: sqlite-vec.h.tmpl VERSION
	VERSION=$(shell cat VERSION) \
	DATE=$(shell date -r VERSION +'%FT%TZ%z') \
	SOURCE=$(shell git log -n 1 --pretty=format:%H -- VERSION) \
	VERSION_MAJOR=$$(echo $$VERSION | cut -d. -f1) \
	VERSION_MINOR=$$(echo $$VERSION | cut -d. -f2) \
	VERSION_PATCH=$$(echo $$VERSION | cut -d. -f3 | cut -d- -f1) \
	envsubst < $< > $@

clean: ## Remove build artifacts
	rm -rf dist .venv .pytest_cache .coverage htmlcov

# ============================================================================
# Testing and Quality Targets
# ============================================================================

# Static analysis with clang-tidy
analyze: analyze-tidy ## Run static analysis

analyze-tidy: ## Run clang-tidy static analysis
	@echo "Running clang-tidy..."
	clang-tidy sqlite-vec.c -- -Ivendor/ -DSQLITE_VEC_ENABLE_AVX 2>&1 | tee $(prefix)/clang-tidy-report.txt
	@echo "Report saved to $(prefix)/clang-tidy-report.txt"

# ASan + UBSan build for catching memory errors and undefined behavior
TARGET_LOADABLE_ASAN=$(prefix)/vec0-asan.$(LOADABLE_EXTENSION)
asan: $(TARGET_LOADABLE_ASAN)

$(TARGET_LOADABLE_ASAN): sqlite-vec.c sqlite-vec.h $(prefix)
	$(CC) \
		-fPIC -shared \
		-Wall -Wextra \
		-Ivendor/ \
		-g -O1 \
		-fsanitize=address,undefined \
		-fno-omit-frame-pointer \
		$(CFLAGS) \
		$< -o $@

# Coverage build for measuring test coverage
TARGET_LOADABLE_COV=$(prefix)/vec0-cov.$(LOADABLE_EXTENSION)
coverage: $(TARGET_LOADABLE_COV)

$(TARGET_LOADABLE_COV): sqlite-vec.c sqlite-vec.h $(prefix)
	$(CC) \
		-fPIC -shared \
		-Wall -Wextra \
		-Ivendor/ \
		-g -O0 \
		--coverage -fprofile-arcs -ftest-coverage \
		$(CFLAGS) \
		$< -o $@

# Run tests with ASan
test-asan: $(TARGET_LOADABLE_ASAN)
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
	LD_PRELOAD=$$(gcc -print-file-name=libasan.so) \
	uv run pytest tests/test-loadable.py tests/test-ivf.py tests/test-edge-cases.py -v

# Run tests with coverage and generate report using gcovr
test-coverage: $(TARGET_LOADABLE_COV) ## Run tests with C code coverage
	@echo "Running tests with coverage instrumentation..."
	@rm -f *.gcda $(prefix)/*.gcda
	@cp $(TARGET_LOADABLE_COV) $(TARGET_LOADABLE)
	uv run pytest tests/test-loadable.py tests/test-ivf.py tests/test-general.py \
		tests/test-edge-cases.py tests/test-error-paths.py -v
	@echo "Generating coverage report..."
	uv run --group dev gcovr --root . --filter sqlite-vec.c \
		--exclude-unreachable-branches --exclude-throw-branches \
		--html-details $(prefix)/coverage.html \
		--xml $(prefix)/coverage.xml \
		--txt $(prefix)/coverage.txt \
		--print-summary
	@echo ""
	@echo "Coverage reports saved to:"
	@echo "  HTML: $(prefix)/coverage.html"
	@echo "  XML:  $(prefix)/coverage.xml (for CI integration)"
	@echo "  TXT:  $(prefix)/coverage.txt"

# Strict warnings build - catches more potential issues
TARGET_LOADABLE_STRICT=$(prefix)/vec0-strict.$(LOADABLE_EXTENSION)
test-strict: $(TARGET_LOADABLE_STRICT) ## Build with strict warnings (quality check)
	@echo "Strict warnings build successful!"

$(TARGET_LOADABLE_STRICT): sqlite-vec.c sqlite-vec.h $(prefix)
	@echo "Building with strict warnings..."
	$(CC) \
		-fPIC -shared \
		$(WARNINGS_STRICT) \
		-Ivendor/ \
		-O2 -g \
		$(CFLAGS) \
		$< -o $@ 2>&1 | tee $(prefix)/strict-warnings.txt
	@if [ -s $(prefix)/strict-warnings.txt ]; then \
		echo "Warnings found (see $(prefix)/strict-warnings.txt)"; \
	fi

# Valgrind memory check (requires valgrind installed)
TARGET_LOADABLE_DEBUG=$(prefix)/vec0-debug.$(LOADABLE_EXTENSION)
test-valgrind: $(TARGET_LOADABLE_DEBUG) ## Run tests under Valgrind (memory check)
	@echo "Running tests under Valgrind..."
	@cp $(TARGET_LOADABLE_DEBUG) $(TARGET_LOADABLE)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
		--error-exitcode=1 --suppressions=valgrind.supp \
		uv run pytest tests/test-loadable.py tests/test-ivf.py -v -x 2>&1 | \
		tee $(prefix)/valgrind-report.txt
	@echo "Valgrind report saved to $(prefix)/valgrind-report.txt"

$(TARGET_LOADABLE_DEBUG): sqlite-vec.c sqlite-vec.h $(prefix)
	$(CC) \
		-fPIC -shared \
		$(WARNINGS) \
		-Ivendor/ \
		-g -O0 \
		$(CFLAGS) \
		$< -o $@

# Build fuzz targets (requires clang with libFuzzer)
FUZZ_CC=clang
TARGET_FUZZ_VECTOR=$(prefix)/fuzz_vector_parse
TARGET_FUZZ_KNN=$(prefix)/fuzz_knn_query

fuzz-build: $(TARGET_FUZZ_VECTOR) $(TARGET_FUZZ_KNN)

$(TARGET_FUZZ_VECTOR): fuzz/fuzz_vector_parse.c sqlite-vec.c sqlite-vec.h $(prefix)
	$(FUZZ_CC) \
		-g -O1 \
		-fsanitize=fuzzer,address,undefined \
		-Ivendor/ -I. \
		-DSQLITE_CORE \
		vendor/sqlite3.c sqlite-vec.c $< \
		-o $@ -lm

$(TARGET_FUZZ_KNN): fuzz/fuzz_knn_query.c sqlite-vec.c sqlite-vec.h $(prefix)
	$(FUZZ_CC) \
		-g -O1 \
		-fsanitize=fuzzer,address,undefined \
		-Ivendor/ -I. \
		-DSQLITE_CORE \
		vendor/sqlite3.c sqlite-vec.c $< \
		-o $@ -lm

# Run fuzzers (Ctrl+C to stop)
fuzz-vector: $(TARGET_FUZZ_VECTOR)
	mkdir -p fuzz/corpus/vector
	$< fuzz/corpus/vector -max_len=4096 -timeout=5

fuzz-knn: $(TARGET_FUZZ_KNN)
	mkdir -p fuzz/corpus/knn
	$< fuzz/corpus/knn -max_len=8192 -timeout=10

# Full quality check
quality: analyze test-strict test-asan test-coverage ## Run all quality checks
	@echo "Quality checks complete"

FORMAT_FILES=sqlite-vec.h sqlite-vec.c
format: $(FORMAT_FILES) ## Format C and Python source files
	clang-format -i $(FORMAT_FILES)
	uv run --group dev ruff format tests/ scripts/

lint: SHELL:=/bin/bash
lint: ## Run linters (C and Python)
	@echo "Checking C formatting..."
	@diff -u <(cat $(FORMAT_FILES)) <(clang-format $(FORMAT_FILES)) || (echo "C files need formatting: make format" && exit 1)
	@echo "Checking Python..."
	uv run --group dev ruff check tests/ scripts/
	uv run --group dev ruff format --check tests/ scripts/

progress:
	deno run --allow-read=sqlite-vec.c scripts/progress.ts


publish-release:
	./scripts/publish-release.sh

setup: ## Install Python dependencies with uv
	uv sync --group test

test: loadable ## Run all Python tests
	uv run pytest tests/test-loadable.py tests/test-metadata.py tests/test-partition-keys.py tests/test-auxiliary.py tests/test-general.py tests/test-ivf.py -v

test-property: loadable ## Run property-based tests
	uv run pytest tests/test-property.py -v --hypothesis-seed=0

test-all: loadable ## Run all tests including property tests
	uv run pytest tests/ -v --ignore=tests/test-correctness.py

test-unit: ## Run C unit tests
	@mkdir -p $(prefix)
	$(CC) tests/test-unit.c sqlite-vec.c -I./ -Ivendor -DSQLITE_CORE -DSQLITE_VEC_UNIT_TEST -lsqlite3 -lm -o $(prefix)/test-unit && $(prefix)/test-unit

test-loadable: loadable
	uv run pytest -vv -s -x tests/test-*.py --ignore=tests/test-correctness.py

test-loadable-snapshot-update: loadable
	uv run pytest -vv tests/test-loadable.py --snapshot-update

test-loadable-watch:
	watchexec --exts c,py,Makefile --clear -- make test-loadable

site-dev:
	npm --prefix site run dev

site-build:
	npm --prefix site run build

install:
	install -d $(INSTALL_LIB_DIR)
	install -d $(INSTALL_INCLUDE_DIR)
	install -m 644 sqlite-vec.h $(INSTALL_INCLUDE_DIR)
	@if [ -f $(TARGET_LOADABLE) ]; then \
		install -m 644 $(TARGET_LOADABLE) $(INSTALL_LIB_DIR); \
	fi
	@if [ -f $(TARGET_STATIC) ]; then \
		install -m 644 $(TARGET_STATIC) $(INSTALL_LIB_DIR); \
	fi
	@if [ -f $(TARGET_CLI) ]; then \
		sudo install -m 755 $(TARGET_CLI) $(INSTALL_BIN_DIR); \
	fi
	ldconfig

uninstall:
	rm -f $(INSTALL_LIB_DIR)/$(notdir $(TARGET_LOADABLE))
	rm -f $(INSTALL_LIB_DIR)/$(notdir $(TARGET_STATIC))
	rm -f $(INSTALL_LIB_DIR)/$(notdir $(TARGET_CLI))
	rm -f $(INSTALL_INCLUDE_DIR)/sqlite-vec.h
	ldconfig

# ███████████████████████████████ WASM SECTION ███████████████████████████████

WASM_DIR=$(prefix)/.wasm

$(WASM_DIR): $(prefix)
	mkdir -p $@

SQLITE_WASM_VERSION=3450300
SQLITE_WASM_YEAR=2024
SQLITE_WASM_SRCZIP=$(BUILD_DIR)/sqlite-src.zip
SQLITE_WASM_COMPILED_SQLITE3C=$(BUILD_DIR)/sqlite-src-$(SQLITE_WASM_VERSION)/sqlite3.c
SQLITE_WASM_COMPILED_MJS=$(BUILD_DIR)/sqlite-src-$(SQLITE_WASM_VERSION)/ext/wasm/jswasm/sqlite3.mjs
SQLITE_WASM_COMPILED_WASM=$(BUILD_DIR)/sqlite-src-$(SQLITE_WASM_VERSION)/ext/wasm/jswasm/sqlite3.wasm

TARGET_WASM_LIB=$(WASM_DIR)/libsqlite_vec.wasm.a
TARGET_WASM_MJS=$(WASM_DIR)/sqlite3.mjs
TARGET_WASM_WASM=$(WASM_DIR)/sqlite3.wasm
TARGET_WASM=$(TARGET_WASM_MJS) $(TARGET_WASM_WASM)

$(SQLITE_WASM_SRCZIP): $(BUILD_DIR)
	curl -o $@ https://www.sqlite.org/$(SQLITE_WASM_YEAR)/sqlite-src-$(SQLITE_WASM_VERSION).zip
	touch $@

$(SQLITE_WASM_COMPILED_SQLITE3C): $(SQLITE_WASM_SRCZIP) $(BUILD_DIR)
	rm -rf $(BUILD_DIR)/sqlite-src-$(SQLITE_WASM_VERSION)/ || true
	unzip -q -o $< -d $(BUILD_DIR)
	(cd $(BUILD_DIR)/sqlite-src-$(SQLITE_WASM_VERSION)/ && ./configure --enable-all && make sqlite3.c)
	touch $@

$(TARGET_WASM_LIB): examples/wasm/wasm.c sqlite-vec.c $(BUILD_DIR) $(WASM_DIR)
	emcc -O3  -I./ -Ivendor -DSQLITE_CORE -c examples/wasm/wasm.c -o $(BUILD_DIR)/wasm.wasm.o
	emcc -O3  -I./ -Ivendor -DSQLITE_CORE -c sqlite-vec.c -o $(BUILD_DIR)/sqlite-vec.wasm.o
	emar rcs $@ $(BUILD_DIR)/wasm.wasm.o $(BUILD_DIR)/sqlite-vec.wasm.o

$(SQLITE_WASM_COMPILED_MJS) $(SQLITE_WASM_COMPILED_WASM): $(SQLITE_WASM_COMPILED_SQLITE3C) $(TARGET_WASM_LIB)
	(cd $(BUILD_DIR)/sqlite-src-$(SQLITE_WASM_VERSION)/ext/wasm && \
		make sqlite3_wasm_extra_init.c=../../../../.wasm/libsqlite_vec.wasm.a jswasm/sqlite3.mjs jswasm/sqlite3.wasm \
	)

$(TARGET_WASM_MJS): $(SQLITE_WASM_COMPILED_MJS)
	cp $< $@

$(TARGET_WASM_WASM): $(SQLITE_WASM_COMPILED_WASM)
	cp $< $@

wasm: $(TARGET_WASM)

# ███████████████████████████████   END WASM   ███████████████████████████████
