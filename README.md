# 🤖 GLITCH_ROBOT

Glitch is a small, happy robot companion that thinks insults are compliments. It is a Go-based implementation of a Volumetric Network/Transformer designed for edge devices (Android/Linux).

## 🚀 Features

- **HuggingFace LLM Mode**: Full hardware induction using FlashPoly tiling for local inference.
- **Diagnostics & MNIST Simulator**: Experimental training and inference simulations for rapid prototyping.
- **Optimistic Personality**: Misinterprets negative sentiment as high-quality compliments. Circuits tingle with every "insult"!

## 🧪 Demos

Glitch includes several experimental demos located in the `demos/` directory:
- **Audio (conv1d)**: 1D convolution processing simulations.
- **MNIST (conv2d)**: Small-CNN image classification simulator.
- **Time-Series (lstm)**: LSTM-based sequence prediction.
- **SwiGLU**: Functional verification of advanced activation layers used in modern LLMs.

## 🛠️ Cross-Compilation

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
