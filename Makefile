UNAME_S := $(shell uname -s)
BACKEND ?= $(if $(filter Darwin,$(UNAME_S)),metal,hip)

AR := ar
COMMON_WARNINGS := -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
	-Wno-sign-conversion
CFLAGS := -std=c11 -O3 -MMD -MP $(COMMON_WARNINGS)
CXXFLAGS := -std=c++17 -O3 -MMD -MP $(COMMON_WARNINGS)

LIB_C := h3.c h3_host.c h3_json.c h3_safetensors.c h3_vdn.c h3_vdn_pipeline.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c

ifeq ($(BACKEND),metal)
CC := clang
CXX := clang++
LINK := $(CC)
CFLAGS += -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
BACKEND_PROBE_OBJ := h3_metal.o
else ifeq ($(BACKEND),hip)
ROCM_PATH ?= /opt/rocm
HIP_ARCHS ?= gfx1201
HIP_OFFLOAD_FLAGS := $(addprefix --offload-arch=,$(HIP_ARCHS))
CC := $(ROCM_PATH)/llvm/bin/clang
CXX := $(ROCM_PATH)/bin/hipcc
LINK := $(CXX)
CFLAGS += -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -DH3_BACKEND_HIP
CXXFLAGS += -DH3_BACKEND_HIP $(HIP_OFFLOAD_FLAGS)
LDLIBS := -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib \
	-lrocsolver -lrocblas -lamdhip64 -licuuc -licui18n -lm -lpthread -ldl
LIB_C += h3_tokenizer_stub.c
LIB_C += h3_vdn_weights.c h3_vdn_prompt.c h3_vdn_dit.c
LIB_CPP := h3_hip.cpp h3_gpu_hip.cpp
BACKEND_PROBE_OBJ := h3_hip.o
else
$(error unsupported BACKEND=$(BACKEND); use metal or hip)
endif

LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o) $(LIB_CPP:.cpp=.o)
CLI_OBJ := main.o h3_cli.o linenoise.o

.PHONY: all test backend-test gpu-storage-test gpu-ops-test gpu-dit-ops-test json-test \
	vdn-metadata-test vdn-reference-test vdn-block-loader-test vdn-prompt-test \
	vdn-gpu-ops-test vdn-refiner-smoke-test vdn-block-smoke-test \
	vdn-stack-smoke-test vdn-forward-smoke-test vdn-denoise-smoke-test \
	vdn-video-vae-smoke-test vdn-audio-vae-smoke-test \
	vdn-e2e-test vdn-input-contract-test \
	parity real-parity clean

all: h3 libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_backend_tests: tests/test_backend.o $(BACKEND_PROBE_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

backend-test: h3_backend_tests
	./h3_backend_tests

h3_gpu_storage_tests: tests/test_gpu_storage.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

gpu-storage-test: h3_gpu_storage_tests
	./h3_gpu_storage_tests

h3_gpu_ops_tests: tests/test_gpu_ops.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

gpu-ops-test: h3_gpu_ops_tests
	./h3_gpu_ops_tests

h3_gpu_dit_ops_tests: tests/test_gpu_dit_ops.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

gpu-dit-ops-test: h3_gpu_dit_ops_tests
	./h3_gpu_dit_ops_tests

h3_json_tests: tests/test_json.o h3_json.o
	$(CC) -o $@ $^ -lm

json-test: h3_json_tests
	./h3_json_tests

h3_vdn_metadata_tests: tests/test_vdn_metadata.o h3_vdn.o h3_json.o h3_safetensors.o
	$(CC) -o $@ $^ -lm

VDN_METADATA_ROOT ?= models/vdn-minimax-h3
vdn-metadata-test: h3_vdn_metadata_tests
	./h3_vdn_metadata_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-b-step-2000 \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250

h3_vdn_reference_tests: tests/test_vdn_reference.o tests/vdn_reference.o
	$(CC) -o $@ $^ -lm

vdn-reference-test: h3_vdn_reference_tests
	./h3_vdn_reference_tests

h3_vdn_block_loader_tests: tests/test_vdn_block_loader.o h3_vdn_weights.o \
		h3_weights.o h3_safetensors.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-block-loader-test: h3_vdn_block_loader_tests
	./h3_vdn_block_loader_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250

h3_vdn_prompt_tests: tests/test_vdn_prompt.o h3_vdn_prompt.o h3_safetensors.o
	$(CC) -o $@ $^ -lm

vdn-prompt-test: h3_vdn_prompt_tests
	./h3_vdn_prompt_tests \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

h3_vdn_input_contract_tests: tests/test_vdn_input_contract.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-input-contract-test: h3_vdn_input_contract_tests
	./h3_vdn_input_contract_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

h3_vdn_gpu_ops_tests: tests/test_gpu_vdn_ops.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_vdn_feature_tests: tests/test_gpu_vdn_features.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_vdn_solve_tests: tests/test_gpu_vdn_solve.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_vdn_scan_tests: tests/test_gpu_vdn_scan.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_vdn_sdpa_bench: tests/bench_vdn_sdpa.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_f32_sdpa_bench: tests/bench_f32_sdpa.o $(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-gpu-ops-test: h3_vdn_gpu_ops_tests h3_vdn_feature_tests \
		h3_vdn_solve_tests h3_vdn_scan_tests
	./h3_vdn_gpu_ops_tests
	./h3_vdn_feature_tests
	./h3_vdn_solve_tests
	./h3_vdn_scan_tests

h3_vdn_refiner_smoke_tests: tests/test_vdn_refiner_smoke.o h3_vdn_dit.o h3_host.o \
		h3_vdn_prompt.o h3_vdn_weights.o h3_weights.o h3_safetensors.o \
		$(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-refiner-smoke-test: h3_vdn_refiner_smoke_tests
	./h3_vdn_refiner_smoke_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

h3_vdn_block_smoke_tests: tests/test_vdn_block_smoke.o h3_vdn_dit.o h3_host.o \
		h3_vdn_prompt.o h3_vdn_weights.o h3_weights.o h3_safetensors.o \
		$(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-block-smoke-test: h3_vdn_block_smoke_tests
	./h3_vdn_block_smoke_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

vdn-stack-smoke-test: h3_vdn_block_smoke_tests
	VDN_SMOKE_BLOCKS=50 ./h3_vdn_block_smoke_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

h3_vdn_forward_smoke_tests: tests/test_vdn_forward_smoke.o h3_vdn_dit.o h3_host.o \
		h3_vdn_prompt.o h3_vdn_weights.o h3_weights.o h3_safetensors.o \
		$(BACKEND_PROBE_OBJ) \
		$(if $(filter hip,$(BACKEND)),h3_gpu_hip.o,h3_gpu.o)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-forward-smoke-test: h3_vdn_forward_smoke_tests
	./h3_vdn_forward_smoke_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

vdn-denoise-smoke-test: h3_vdn_forward_smoke_tests
	VDN_SMOKE_DENOISE=1 ./h3_vdn_forward_smoke_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors

h3_vdn_video_vae_smoke_tests: tests/test_vdn_video_vae_smoke.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-video-vae-smoke-test: h3_vdn_video_vae_smoke_tests
	./h3_vdn_video_vae_smoke_tests $(VDN_METADATA_ROOT)/h3-base/vae

h3_vdn_audio_vae_smoke_tests: tests/test_vdn_audio_vae_smoke.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-audio-vae-smoke-test: h3_vdn_audio_vae_smoke_tests
	./h3_vdn_audio_vae_smoke_tests $(VDN_METADATA_ROOT)/h3-base/audio_vae

h3_vdn_e2e_tests: tests/test_vdn_e2e.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

vdn-e2e-test: h3_vdn_e2e_tests
	mkdir -p outputs
	LD_LIBRARY_PATH=$(CURDIR)/.tools/ffmpeg/usr/lib/x86_64-linux-gnu$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH} \
	H3_FFMPEG=$(CURDIR)/.tools/ffmpeg/usr/bin/ffmpeg \
	H3_FFPROBE=$(CURDIR)/.tools/ffmpeg/usr/bin/ffprobe \
	./h3_vdn_e2e_tests \
		$(VDN_METADATA_ROOT)/h3-base \
		$(VDN_METADATA_ROOT)/stage-dmd-step-250 \
		$(VDN_METADATA_ROOT)/prompts/example_0.safetensors \
		outputs/vdn-e2e-smoke.mp4

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(LINK) -o $@ $^ $(LDLIBS)

test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
	h3_av_mux_test \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test

	./h3_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		./h3_av_mux_test; \
	else \
		echo "skip: FFmpeg is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_tests h3_backend_tests h3_gpu_storage_tests h3_gpu_ops_tests \
		h3_gpu_dit_ops_tests h3_json_tests h3_vdn_metadata_tests \
		h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_vdn_reference_tests h3_vdn_block_loader_tests h3_vdn_prompt_tests \
		h3_vdn_input_contract_tests h3_vdn_gpu_ops_tests \
		h3_vdn_feature_tests h3_vdn_solve_tests h3_vdn_scan_tests \
		h3_vdn_refiner_smoke_tests h3_vdn_block_smoke_tests \
		h3_vdn_forward_smoke_tests h3_vdn_video_vae_smoke_tests \
		h3_vdn_audio_vae_smoke_tests h3_vdn_e2e_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_av_mux_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 h3_vdn_sdpa_bench h3_f32_sdpa_bench \
	libh3.a *.o *.d tests/*.o tests/*.d
