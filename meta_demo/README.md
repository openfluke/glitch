# Poly Metacognition: "The Impossible Task"

This directory contains demonstrations of **Metacognition** (thinking about thinking) within the Poly engine. Specifically, it showcases a task that is fundamentally impossible for standard backpropagation: **Autonomous Architectural Evolution**.

## What is "The Impossible Task"?

Standard neural networks are constrained by a **static computational graph**. During training, backpropagation can only adjust the *values* (weights) of existing nodes. It cannot:
1.  Change a `Dense` layer (Linear Algebra) into a `KMeans` layer (Geometric Clustering).
2.  Decide to change its own hardware state (CPU to GPU) based on perceived difficulty.
3.  Rewrite its own underlying dispatch logic on-the-fly.

In `impossible_main.go`, we demonstrate a network that does exactly this.

### How it Works

The demo uses a **Nested Metacognition** stack:

1.  **Level 0 (The Brain)**: Starts as a standard `LayerDense`. It struggles to process data that is naturally grouped into 4 distinct spatial clusters.
2.  **Level 1 (The Observer)**: A `LayerMetacognition` that monitors the "thoughts" (activations/stats) of the brain.
3.  **The Meta-Agent**: Inside Level 1, a sub-network analyzes the "Stream of Consciousness" (Event History).
4.  **Recursive Intervention**: Upon detecting that the data is better suited for a different mathematical paradigm, the Meta-Agent issues an `autonomous_command`: **`MorphToKMeans`**.

### The Topological Transition

When the command is issued, the Poly engine performs a "Metamorphosis":
- It changes the `Layer.Type` field.
- It re-initializes the layer using `InitKMeansCell`, transforming a weight matrix into cluster centroids.
- The next forward pass automatically routes through the `KMeans` kernel instead of the `Dense` kernel.

## Demo Files

- **`main.go`**: Basic demo showing noise injection and gating to suppress "glitches".
- **`autonomous_main.go`**: Shows a meta-agent monitoring its own history and deciding to "focus" by upgrading its own precision (Float32 -> Float64).
- **`impossible_main.go`**: The flagship demo of architectural self-evolution (Dense -> KMeans).

## Why this matters

True AGI requires the ability to recognize when its current "way of thinking" is insufficient for a problem. By enabling layers to monitor themselves and autonomously restructure their own "brains," we move beyond fixed-weight optimization and into the realm of **Dynamic Autonomous Intelligence**.
