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
### Context Scaling (Deep N-grams)

`test_compare_deep.go` explores the limits of context window size using **Sparse Snapping**. Instead of allocating $V^N$ clusters, we only create clusters for patterns actually observed in the corpus.

| Generation | Context Window | Unique Patterns | Perplexity |
| :--- | :---: | :---: | :---: |
| 3-gram | 2 chars | 1,360 | 7.22 |
| **4-gram** | **3 chars** | **10,899** | **6.15** |
| 5-gram | 4 chars | 46,224 | 12.04 |
| 6-gram | 5 chars | 125,229 | 36.30 |

#### The "Context Peak" Phenomenon
As the window size increases to **4-gram**, perplexity continues to drop as the model captures longer structural dependencies. However, at **5-gram and 6-gram**, we observe a sharp *increase* in perplexity. This is due to **data sparsity**: 6rd gen patterns are so specific to the training set that they rarely appear in the validation set, causing the model to miss context lookups and revert to uniform noise.

### Full Benchmark Output

```text
╔══════════════════════════════════════════════════════════════════╗
║      DEEP N-GRAM BENCHMARK  ·  CONTEXT SCALING                 ║
║  Testing 3, 4, 5, and 6-gram performance with Sparse Snapping  ║
╚══════════════════════════════════════════════════════════════════╝

[*] Corpus: 1115394 chars | Vocab: 65

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [EXPERIMENT]  3-gram (ctxLen=2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Analyzing corpus for 3-grams...
  Found 1360 unique patterns.
  Snap time: 3.64ms  |  Perplexity: 7.22

  Sample text:
   am of hind the reirst he pat le; wrayse prom sord.

Cord le frome glive
   sis me my wer:
That your, whictagume soliefted
But wily king ague comee
  liarposes;
Hat it mis uposs,
Untly fork is tholad ce my he

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [EXPERIMENT]  4-gram (ctxLen=3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Analyzing corpus for 4-grams...
  Found 10899 unique patterns.
  Snap time: 35.71ms  |  Perplexity: 6.15

  Sample text:
   are cousands, whence chart is above a cold fellows' liver comfree that
  the the wand tremies gent do you are lie any fell sent,
To leavens, tite
   breat than if I secome; and shuts I commany thould well gi

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [EXPERIMENT]  5-gram (ctxLen=4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Analyzing corpus for 5-grams...
  Found 46224 unique patterns.
  Snap time: 188.91ms  |  Perplexity: 12.04

  Sample text:
   are the Earl of out with, now them.

PRINCENTIO:
Two men of not
Upbraid
  ing from my looker-on
The coffices of York.
What not.

DUKE OF YORK:
It
  adds for I must been your judgmen.
Yet will;
Or else purpose

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [EXPERIMENT]  6-gram (ctxLen=5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Analyzing corpus for 6-grams...
  Found 125229 unique patterns.
  Snap time: 553.69ms  |  Perplexity: 36.30

  Sample text:
   are full of her from the house in France,
Edward and fish'd years were
  age better toward they seduce that thou stood prunes;
sir, by a charity
  sits,
Advantage shall not for we assured birds or bad? and o'

## Interactive Snap Chat (6-Gram)

The `chat_snap.go` utility provides an interactive interface to the best-performing sparse n-gram models.

```text
╔══════════════════════════════════════════════════════════════════╗
║               SNAP INTERACTIVE CHAT  ·  6-GRAM                 ║
║  Powered by the Poly Engine  ·  Zero-Backprop Architecture   ║
╚══════════════════════════════════════════════════════════════════╝

[*] Loading and Snapping 6-gram model (ctxLen=5)...
[*] Analyzing unique patterns...
[*] Found 141021 unique 6-gram patterns.
[*] Snap complete. System ready.

Prompt > hello how are you?
Answer >
Servant gardening house,
That's Montague; for Katharina shame homely.

BUCKINGHAM:
My life as my rats than she's most lose happy vice's birds would

[Stats] Gen time: 9.652s | Performance: 15.54 chars/sec

Prompt > my rats
Answer >  are person, and king pity nor for a little accuse his master 
it was not so mine own since him and knees
Or ears?

MARCIUS:
No less, friar is will not

[Stats] Gen time: 9.608s | Performance: 15.61 chars/sec
```

## Architecture Summary

| Model             | Context | Patterns | Perplexity | Notes |
|-------------------|---------|----------|------------|-------|
| 3-gram            | 2 chars | 1,360    | 7.22       | Stable baseline |
| 4-gram            | 3 chars | 10,899   | 6.15       | **Global Optimum** |
| 5-gram            | 4 chars | 46,224   | 12.04      | Sparsity onset |
| 6-gram            | 5 chars | 141,021  | 36.30      | Maximum coherence |

╔══════════════════════════════════════════════════════════════════╗
║               DEEP N-GRAM CONTEXT COMPARISON                     ║
╠══════════════════════════════════════════════════╦══════════════╣
║ Generation                                       ║  Perplexity  ║
╠══════════════════════════════════════════════════╬══════════════╣
║ 3-gram (ctxLen=2)                                ║      7.22    ║
║ 4-gram (ctxLen=3)                                ║      6.15    ║
║ 5-gram (ctxLen=4)                                ║     12.04    ║
║ 6-gram (ctxLen=5)                                ║     36.30    ║
╚══════════════════════════════════════════════════╩══════════════╝
```

## Architecture Showdown

While `main.go` focuses on n-gram progression, `test_compare.go` evaluates how different neural architectural primitives perform when "snapped" with identical statistical knowledge.

### Benchmark Results (Tiny Shakespeare)

| Architecture | Perplexity | Notes |
| :--- | :---: | :--- |
| **Sequential(KMeans, Dense) [Trigram]** | **7.65** | **The Winner.** Precise context-to-probability mapping. |
| Sequential(Dense, Dense) [Bigram MLP] | 11.85 | Equivalent to Bigram lookup. |
| Sequential(KMeans[Feats], Dense) [Trigram] | 11.85 | Features mode tends towards Bigram performance limits. |
| Parallel(Uni, Bi, Tri) | 27.57 | Simple addition of branches; needs better smoothing. |
| Sequential(MHA, Dense) [Attention] | 77.67 | Naive attention weights struggle without backprop. |
| Sequential(LSTM, Dense) [Recurrent] | 107.75 | Non-linearities (Sigmoid/Tanh) dampen one-hot signals. |
| Sequential(CNN1, Dense) [Conv] | 336.38 | Local filters need more complex weight installation. |

### Analysis

1.  **KMeans is the Zero-Backprop MVP**: Discrete context clustering (KMeas) perfectly matches the discrete nature of N-grams. It acts as a high-speed hash table in weight space, allowing for sharp, precise predictions.
2.  **The "Backprop Gap"**: Continuous architectures like **LSTM** and **MHA** are designed to be *trained*, not configured. Their internal activations (Sigmoid, Tanh, Softmax) are optimized for distributed, learned representations. Snapping them with "Identity" or "Pass-through" weights often leads to signal degradation (Perplexity > 65).
3.  **Features vs. Identity**: ARCH 4 (KMeans Features) didn't outperform ARCH 2 (Bigram MLP), suggesting that distance-based outputs in a one-hot character space don't provide useful "fuzzy" logic—a character is either a match or it isn't.

## Usage

Ensure you have the `poly` engine available in your Go path, then run the demos:

```bash
# Standard N-gram progression
go run main.go

# Comprehensive architecture benchmark
go run test_compare.go

# Deep context scaling (sparse)
go run test_compare_deep.go

# Universal matrix benchmark (Arch x Gen)
go run test_compare_deep_all_layers.go
```

The scripts will automatically download the Tiny Shakespeare corpus, build the networks, "snap" weights, and generate samples.

---
*Built with the [Poly Engine](https://github.com/openfluke/loom/tree/main/poly).*
