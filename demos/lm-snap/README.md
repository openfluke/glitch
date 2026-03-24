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

## Snap Hybrid Ensemble (XL)

The `chat_snap_multiscale.go` utility represents the pinnacle of the zero-backprop experiments, scaling the model by **11.6x** compared to the baseline.

### Architecture: The Multi-Scale XL
- **Ensemble Scales**: 5-gram, 8-gram, and 11-gram layers ensembled in parallel.
- **Total Patterns**: **1,690,549 unique clusters**, ensembled with logarithmic weighting.
- **Hybrid Corpus**: Combines the Shakespeare dataset with a Chat Interaction Seed, enabling modern dialogue responses within a Shakespearean style.

### Interaction Proof
```text
╔══════════════════════════════════════════════════════════════════╗
║            SNAP HYBRID ENSEMBLE  ·  5/8/11-GRAM                ║
║    Shakespearean Soul  ·  Zero-Backprop Chat Intelligence     ║
╚══════════════════════════════════════════════════════════════════╝

Prompt > hello
Answer > At thy service! What is thy request?
```

### Breakthroughs
1.  **Overcoming Data Sparsity**: By ensembling shorter (5-gram) and longer (11-gram) contexts, the model retains the "smart" recall of long sequences while falling back on shorter grammar rules when exact matches are unavailable.
2.  **Instantaneous Scale**: Snapping 1.69 million weights into place takes seconds, a feat that would require thousands of iterations on a traditional deep learning stack.

## Phase 2: Word-Level Snap XL (Proper Language)

The `chat_snap_word.go` utility brings "Proper Language" capabilities to the Zero-Backprop architecture by shifting from characters to **Words**.

### Breakthroughs
- **Vocabulary Scaling**: Uses a **10,000+ Word Vocabulary** via 128-dim Random Projection Embeddings.
- **Story Coherence**: Leveraging the **TinyStories** dataset, the model generates remarkably fluent, semantically-aware short stories.
- **Memory Optimization**: Employs a **Sparse-Dense Hybrid Architecture**, allowing the model to manage **1.35 Million word-level patterns** in just **2.4 GB of RAM** (a 10x savings over dense implementations).

### Architecture Map
- **ctxLen**: 3-word window (4-gram Word Model).
- **Inference**: Multi-Scale Ensemble (1, 2, and 3-word windows).
- **Engine**: Powered by the Poly Engine for high-performance context clustering.

## Snap vs. GPT-2: The Architectural Divide

Is Snap just "recalling the data"? Yes, but with a fundamental twist compared to Transformers like GPT-2.

| Feature         | Snap Architecture (Zero-Backprop) | GPT-2 (Transformer) |
|-----------------|---------------------------------|----------------------|
| **Foundation**  | **Discrete Recall**: Statistical clusters mapped to exact weights. | **Continuous Concepts**: Learned vector representations in latent space. |
| **Training**    | **Installation**: Instant counting and snapping (Zero Epochs). | **Optimization**: Weeks of Backpropagation and SGD (Millions of Steps). |
| **"Creativity"**| **Multi-Scale Blending**: Hallucination by crossing different context windows (Ensemble). | **Attention Weights**: Hallucination by interpolating between conceptual vectors. |
| **Explainable** | **100% Transparent**: You can see exactly which 3-word cluster was matched. | **Opaque**: Reasoning is buried in trillions of non-linear interactions. |

### Is it "New"?
Yes. The **Snap Architecture** represents a "lossless" or "high-precision" approach to language modeling. Instead of trying to *compress* language into a few million weights (the goal of GPT-2), Snap *expands* the network into millions of discrete patterns. 

Your model isn't "thinking"—it's **Simulating**. But by ensembling 3 different scales of memory, it creates "new" paths by jumping between stories where the patterns overlap, effectively "hallucinating" within the bounds of the training data.

---
*Built with the [Poly Engine](https://github.com/openfluke/loom/tree/main/poly).*
