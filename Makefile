# abla-grad. COMPILER defaults to the installed ablac; the CUDA tests need a GPU and the shell (nix-shell shell.nix).
COMPILER ?= ablac
BUILD := build

.PHONY: test test-host
test: test-host

# Tests that need no device.
test-host:
	mkdir -p $(BUILD)
	$(COMPILER) build tests/backend_interface_test.ab -o $(BUILD)/backend_interface_test --fast --no-cache && $(BUILD)/backend_interface_test
	$(COMPILER) build tests/ops_test.ab -o $(BUILD)/ops_test --fast --no-cache && $(BUILD)/ops_test
	$(COMPILER) build tests/compare_test.ab -o $(BUILD)/compare_test --fast --no-cache && $(BUILD)/compare_test
	tests/compare_cli_test.sh $(COMPILER)
	$(COMPILER) build tests/gguf_reader_test.ab -o $(BUILD)/gguf_reader_test --fast --no-cache && $(BUILD)/gguf_reader_test $(BUILD)/gguf-fixture.gguf

# Tests that need the GPU (the CUDA backend and its kernels). Run in `nix-shell shell.nix`.
.PHONY: test-gpu
test-gpu:
	mkdir -p $(BUILD)
	$(COMPILER) build tests/kernels/matvec_f32_test.ab -o $(BUILD)/matvec_f32_test --fast --no-cache && $(BUILD)/matvec_f32_test
	$(COMPILER) build tests/kernels/elementwise_test.ab -o $(BUILD)/elementwise_test --fast --no-cache && $(BUILD)/elementwise_test
