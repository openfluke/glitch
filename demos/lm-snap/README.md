# CHAR-LM SNAP · Zero Backprop Language Model

A character-level language model that installs statistical knowledge directly into a neural network—**no gradient descent, no backpropagation, and zero training epochs.**

Inspired by the efficiency of statistical n-grams but implemented within the `poly` volumetric network architecture, this demo shows how to "snap" weights into place to achieve immediate, interpretable results.

## Overview

Traditional LSTMs or Transformers take hours to reach a respectable perplexity on the Tiny Shakespeare dataset. **CHAR-LM SNAP** reaches comparable results in **milliseconds** by bypassing SGD entirely.

The model uses a simple but powerful architecture:
1. **KMeans Layer**: Acts as a sharp context lookup. It maps one-hot character inputs (or concatenated multi-character windows) to specific "context clusters."
2. **Dense Layer**: Acts as a probability table. It maps the context clusters to a log-probability distribution (logits) for the next character.

By counting occurrences in the training corpus and calculating exact transition probabilities (with Laplace smoothing), we "install" these statistics directly as weights.

## How it Works

The demo progresses through four hypotheses:

### [GEN 1] Unigram Frequency
*   **Hypothesis**: Character frequency is non-uniform.
*   **Action**: Install log-unigram probabilities as Dense weights.
*   **Result**: Drastic improvement over random noise by simply guessing the most common characters.

### [GEN 2] Bigram Transitions
*   **Hypothesis**: Adjacent character pairs carry strong signal.
*   **Action**: Use KMeans to identify the previous character and Dense to store $P(\text{next} | \text{prev})$.
*   **Result**: Perplexity drops significantly as the model learns basic spelling and structure (e.g., `q` → `u`).

### [GEN 3] Temperature Sweep
*   **Hypothesis**: KMeans "sharpness" affects lookup quality.
*   **Action**: Sweep the `KMeansTemperature` to find the optimal balance between soft-assignment and hard-lookup.

### [GEN 4] Trigram Window
*   **Hypothesis**: Doubling the context window (2-char context) yields better predictions.
*   **Action**: KMeans with $V^2$ clusters and a $2 \times V$ input. Dense stores $P(\text{next} | c_{t-1}, c_{t-2})$.
*   **Result**: Further perplexity reduction, producing text that mimics Shakespeare's style and vocabulary much more closely.

## Results Summary

| Model Generation | Perplexity |
| :--- | :---: |
| Random (baseline) | 65.00 |
| Unigram | 27.02 |
| Bigram (T=0.30) | 12.21 |
| **Trigram (2-char ctx)** | **8.54** |

*Note: Perplexity for the Trigram model on a held-out test set is ~8.54.*

## Usage

Ensure you have the `poly` engine available in your Go path, then run:

```bash
go run main.go
```

The script will automatically download the Tiny Shakespeare corpus, build the network, "snap" the weights for each generation, and sample text at various temperatures.

---
*Built with the [Poly Engine](https://github.com/openfluke/loom/tree/main/poly).*
