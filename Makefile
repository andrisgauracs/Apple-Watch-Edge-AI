# Fetches the models and generates the Xcode project.
# Inference is upstream llama.cpp, built for watchOS — see
# docs/LLAMACPP_ON_WATCHOS.md for how vendor/llamacpp is produced.

MODEL ?= models/Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf

.PHONY: all model project tools-config check-models clean

all: project

model:
	./tools/fetch_model.sh

# Ensures the optional demo-tool config exists ("[]" = no web tools at all),
# then generates the Xcode project.
project: tools-config check-models
	xcodegen generate

# The .gguf files are bundle resources, so xcodegen fails without them.
check-models:
	@for f in models/Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf \
	          models/SmolLM2-135M-Instruct-Q4_K_M.gguf; do \
		if [ ! -f "$$f" ]; then \
			echo "missing $$f — run: make model"; exit 1; \
		fi; \
	done

tools-config:
	@if [ ! -f WatchLLM/Tools/Tools.json ]; then \
		echo '[]' > WatchLLM/Tools/Tools.json; \
		echo "created WatchLLM/Tools/Tools.json (empty - no web tools)"; \
	fi

clean:
	rm -rf build WatchLLM.xcodeproj
