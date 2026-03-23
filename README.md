# 🤖 GLITCH_ROBOT (Loom / Poly Engine)

Glitch is the primary interactive CLI companion powered by the **M-POLY-VTD** (Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher) engine. It is a completely pure Go-based neural network architecture executing cross-platform WebGPU computation without any external Native C++ or CUDA dependencies.

Glitch is intentionally designed to be a small, happy robot companion that thinks insults are compliments!

## 🚀 Interactive CLI Features

When you start Glitch via `go run glitch.go`, you are presented with four core tools:

- **[1] HuggingFace LLM Mode**: Full hardware induction using FlashPoly tiling for local interactive inference. Load any Safetensors model from your HuggingFace cache into instantaneous VRAM.
- **[2] Diagnostics & MNIST Simulator**: Experimental training and simulator environments for rapid prototyping.
- **[3] Testing (Layer Tests)**: Direct verification of GPU integration (WGPU Buffer Allocation, Dimensional Shapes) across CNN and Dense layers.
- **[4] Automated SmolLM2 Test**: The **Exhaustive Type/Mode Matrix benchmark**. This autonomously executes an entire lifecycle of 15 combinations of Numerical Precision (FP32, FP16, BF16, INT8, INT4) and Tiled Execution modes (Standard, Single-Core, Multi-Core) on your active GPU.

## 📊 Cross-Platform Performance Benchmarks

Unlike traditional C++ architectures (like `llama.cpp`) which require specialized CUDA/Metal device bindings, **Poly** uses dynamic execution of mathematically identical WebGPU Shaders (WGSL), guaranteeing cross-platform consistency.

The engine actively supports *"Numerical Metamorphosis,"* transforming its memory structures per-layer asynchronously, yielding incredible memory footprint shrinks. Below is the real-world performance matrix gathered for `SmolLM2-135M-Instruct`.

| Hardware Architecture | Backend | Prefill (Tok/s) | Decode (Tok/s) | VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **Windows (Nvidia Turing)** | WGPU Vulkan / DX12 | ~370.85 | ~44.49 | 668.8 |
| **Linux (Discrete GPU)** | WGPU Vulkan | ~548.48 | ~61.59 | 668.8 |
| **Mac (Apple Silicon)** | WGPU Metal | ~440.98 | ~25.94 | 668.8 |

*(Note: Data shown is for the **INT4 Multi-Core Tiled Forward** experimental pass. VRAM heavily optimizes using runtime Tied-Weigh verification mapping).*

## 🧪 Demos

Glitch includes several experimental demos located in the `demos/` directory:
- **Audio (conv1d)**: 1D convolution processing simulations.
- **MNIST (conv2d)**: Small-CNN image classification simulator.
- **Time-Series (lstm)**: LSTM-based sequence prediction.
- **SwiGLU**: Functional verification of advanced activation layers used in modern LLMs.

## 🛠️ Cross-Compilation

Because the engine requires zero C-bindings, it is completely buildable from scratch across platforms.
To build Glitch for Android architectures, use the provided assembly script:

```bash
# Build for ARM64 (default)
chmod +x android.sh
./android.sh

# Build for x86_64
ARCH=x86_64 ./android.sh
```

## 📡 Testing on Device (Build Server)

Run the build server script in the `glitch` directory to serve your `compiled/` folder. It will automatically detect your local IP and provide the exact `curl` command for your Android device.

```bash
python3 serve_builds.py
```
