# ⚛️ Loom Polymorphic Semantic Search Demo

This demo implements a high-performance, **Transformer-free** semantic search engine within the Loom/Poly engine. It specifically showcases **Numerical Metamorphosis**—the ability to morph a model's weight precision on-the-fly without changing the input data.

## 🧬 How it Works

1.  **Bedrock Embeddings**: Uses **GloVe 6B (300d)** static word vectors (~480MB).
2.  **No-Transformer Architecture**: Instead of expensive multi-head attention, it uses a **Bag of Embeddings** approach with Mean Pooling. This is orders of magnitude faster and memory-efficient for edge devices.
3.  **Polymorphic Dispatch**: The engine runs the exact same query through three different numerical families:
    *   **FP32 (Standard)**: The high-fidelity baseline.
    *   **INT8 (Quantized)**: 75% memory reduction with nearly identical results.
    *   **Binary (1-Bit)**: 97% memory reduction. It reduces every weight to a simple `+1` or `-1` (sign function), yet still maintains the "semantic direction" of the query.

## 🚀 Try These Queries

Type these into the search prompt to see how the engine correlates meanings:

*   **"Tell me about space exploration"** (Should link to the James Webb and Mars entries)
*   **"I want to bake something"** (Should link to the Chocolate Cake and Sourdough entries)
*   **"How does the Loom engine work?"** (Should link to the 3D volumetric and VTD entries)
*   **"Ancient history"** (Should link to Rome and the Renaissance)
*   **"Tell me about physics"** (Should link to Quantum Mechanics and the Speed of Light)

## ⚖️ What to Expect

*   **Semantic Drift**: As you switch to **Binary** mode, you will notice the similarity scores fluctuate. However, the *ranking* of the results should remain remarkably stable. This demonstrates Loom's **Topological Robustness**.
*   **No Word-Order Sensitivity**: Because this demo uses "Bag of Words" mean-pooling, "Dog bites man" and "Man bites dog" will result in identical embeddings. This is the trade-off for eliminating the complexity of a Transformer.
*   **Deterministic Scores**: Because Loom is bit-perfect, these scores will be identical on every run and every platform (Windows/Linux/WASM).

## 🛠️ File Structure

*   `main.go`: The interactive search loop and polymorphic logic.
*   `corpus.go`: The knowledge base being searched.
*   `converter/`: Utility to download and transform the raw GloVe dataset.
*   `run_demo.bat` / `run_demo.sh`: One-click execution scripts.

---

**Experiment with the "Numerical Monster" and see how much precision you can sacrifice before the meaning is lost! o_O**


outputs:
ΓÜ¢∩╕Å  Loom Poly - Polymorphic Semantic Search Demo
-----------------------------------------------

ΓÅ│ Step 1: Downloading and Converting GloVe 6B (300d)...
(Note: This will download ~822MB if not already present)

✅ glove_300d.safetensors already exists, skipping conversion.

≡ƒÜÇ Step 2: Starting Interactive Search Demo...

⏳ Loading glove_300d.safetensors and vocab.txt into Poly Bedrock...
✅ Corpus Online. Ready for semantic correlation.
⏳ Pre-calculating 30 KnowledgeBase embeddings in FP32...

🔍 Query (or 'exit'): Tell me about space exploration
🧬 Mode: float32... (0s)
  0.8288: The James Webb Space Telescope is the most powerful space telescope ever built, designed to see the first stars.
  0.7979: A black hole is a region of spacetime where gravity is so strong that nothing, including light, can escape.
🧬 Mode: int8... (701.5262ms)
  0.8296: The James Webb Space Telescope is the most powerful space telescope ever built, designed to see the first stars.
  0.7987: A black hole is a region of spacetime where gravity is so strong that nothing, including light, can escape.
🧬 Mode: binary... (673.2422ms)
  0.8315: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.
  0.8178: Sourdough bread is made by the fermentation of dough using naturally occurring lactobacilli and yeast.

🔍 Query (or 'exit'): I want to bake something
🧬 Mode: float32... (0s)
  0.8618: To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, milk, and vegetable oil.
  0.7747: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.
🧬 Mode: int8... (0s)
  0.8617: To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, milk, and vegetable oil.
  0.7750: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.
🧬 Mode: binary... (167.9343ms)
  0.8449: To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, milk, and vegetable oil.
  0.8197: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.

🔍 Query (or 'exit'): How does the Loom engine work?
🧬 Mode: float32... (381.5µs)
  0.8504: The Loom engine uses a 3D volumetric coordinate system (Depth, Row, Col) for tensor dispatch.
  0.8408: Mars is often called the Red Planet because of the iron oxide (rust) on its surface.
🧬 Mode: int8... (0s)
  0.8507: The Loom engine uses a 3D volumetric coordinate system (Depth, Row, Col) for tensor dispatch.
  0.8409: Mars is often called the Red Planet because of the iron oxide (rust) on its surface.
🧬 Mode: binary... (82.7404ms)
  0.8624: The Loom engine uses a 3D volumetric coordinate system (Depth, Row, Col) for tensor dispatch.
  0.8380: The SwiGLU activation function is a variant of the Gated Linear Unit used in many modern LLM architectures.

🔍 Query (or 'exit'): Ancient history
🧬 Mode: float32... (149µs)
  0.7149: The Roman Empire reached its greatest territorial extent under the Emperor Trajan in 117 AD.
  0.7077: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.
🧬 Mode: int8... (0s)
  0.7158: The Roman Empire reached its greatest territorial extent under the Emperor Trajan in 117 AD.
  0.7087: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.
🧬 Mode: binary... (83.4907ms)
  0.7601: The SwiGLU activation function is a variant of the Gated Linear Unit used in many modern LLM architectures.
  0.7584: The Roman Empire reached its greatest territorial extent under the Emperor Trajan in 117 AD.

🔍 Query (or 'exit'): Tell me about physics
🧬 Mode: float32... (944.6µs)
  0.8005: Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms.
  0.7823: A black hole is a region of spacetime where gravity is so strong that nothing, including light, can escape.
🧬 Mode: int8... (0s)
  0.8015: Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms.
  0.7833: A black hole is a region of spacetime where gravity is so strong that nothing, including light, can escape.
🧬 Mode: binary... (172.5363ms)
  0.7999: Semantic search uses embeddings to find information based on meaning rather than just keyword matching.
  0.7978: Sourdough bread is made by the fermentation of dough using naturally occurring lactobacilli and yeast.

🔍 Query (or 'exit'): 🧬 Mode: float32... (432.1µs)
  0.0000: The Loom engine uses a 3D volumetric coordinate system (Depth, Row, Col) for tensor dispatch.
