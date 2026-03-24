package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	DataDir    = "data"
	CorpusFile = "data/shakespeare.txt"
	CorpusURL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	EvalChars  = 20000 // chars to evaluate perplexity on (keep eval fast)
	GenLength  = 400   // chars to generate per text sample

	// ═══════════════════════════════════════════════════════════════
	// 🎛️  MODEL SCALE CONTROL — CHANGE THIS NUMBER TO SCALE THE MODEL
	// ═══════════════════════════════════════════════════════════════
	// Scale = 1  → original model (V→V→V)
	// Scale = 2  → 2× hidden capacity (V→2V→V)
	// Scale = 10 → 10× hidden capacity (V→10V→V)
	// Scale = N  → N× more clusters/neurons, same statistics installed
	ModelScale = 3
	// ═══════════════════════════════════════════════════════════════
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║      CHAR-LM SNAP  ·  ZERO BACKPROP LANGUAGE MODEL             ║")
	fmt.Printf("║  Tiny Shakespeare  ·  Model Scale: ×%d                          ║\n", ModelScale)
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("A trained LSTM on this dataset takes hours and reaches ~1.3 ppl.")
	fmt.Println("We install statistical knowledge directly — no SGD, no epochs.")
	fmt.Println()

	// ── 1. DATA ──
	if err := ensureCorpus(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	fmt.Println("[*] Loading corpus...")
	corpus, vocab, _ := loadCorpus()
	V := len(vocab)
	H := V * ModelScale // SCALED hidden dimension
	n := len(corpus)

	trainEnd := n * 8 / 10
	valEnd := n * 9 / 10
	train := corpus[:trainEnd]
	val := corpus[trainEnd:valEnd]
	test := corpus[valEnd:]
	fmt.Printf("[*] Corpus: %d chars  |  vocab: %d  |  hidden dim: %d (×%d)\n", n, V, H, ModelScale)
	fmt.Printf("[*] Split:  train=%d  |  val=%d  |  test=%d\n\n", len(train), len(val), len(test))

	// ── 2. BUILD NETWORK (SCALED) ──
	fmt.Printf("[*] Building network: Sequential([KMeans(%d→%d), Dense(%d→%d)])\n", V, H, H, V)
	net := buildNetworkScaled(V, H)
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	obs := net.Layers[0].MetaObservedLayer
	kmeansL := &obs.SequentialLayers[0]
	denseL := &obs.SequentialLayers[1]
	fmt.Printf("[*] KMeans: %d clusters × %d-dim  |  Dense: %d → %d\n\n", H, V, H, V)

	// ── HELPERS (SCALED) ──

	// KMeans: each vocab char → ModelScale sub-clusters (identity with slight diversity)
	installIdentityKMeansScaled := func(temp float64) {
		kmeansL.NumClusters = H
		kmeansL.InputHeight = V
		kmeansL.OutputHeight = H
		kmeansL.KMeansTemperature = temp
		kmeansL.WeightStore = poly.NewWeightStore(H * V)
		for v := 0; v < V; v++ {
			for s := 0; s < ModelScale; s++ {
				idx := v*ModelScale + s
				// Primary: one-hot at position v
				kmeansL.WeightStore.Master[idx*V+v] = 1.0
				// Optional: tiny noise to other dims for sub-cluster diversity
				for j := 0; j < V; j++ {
					if j != v {
						kmeansL.WeightStore.Master[idx*V+j] = float32(s) * 1e-4
					}
				}
			}
		}
	}

	// Dense unigram: replicate log-unigram across each vocab char's sub-clusters
	installDenseUnigramScaled := func(logProbs []float32) {
		denseL.InputHeight = H
		denseL.OutputHeight = V
		denseL.Activation = poly.ActivationLinear
		denseL.WeightStore = poly.NewWeightStore(V * H)
		for out := 0; out < V; out++ {
			for v := 0; v < V; v++ {
				for s := 0; s < ModelScale; s++ {
					inIdx := v*ModelScale + s
					denseL.WeightStore.Master[out*H+inIdx] = logProbs[out]
				}
			}
		}
	}

	// Dense bigram: for each prev-char, replicate its row across sub-clusters
	installDenseBigramScaled := func(bigram [][]float32) {
		denseL.InputHeight = H
		denseL.OutputHeight = V
		denseL.Activation = poly.ActivationLinear
		denseL.WeightStore = poly.NewWeightStore(V * H)
		for out := 0; out < V; out++ {
			for prev := 0; prev < V; prev++ {
				logP := float32(math.Log(float64(bigram[prev][out])))
				for s := 0; s < ModelScale; s++ {
					inIdx := prev*ModelScale + s
					denseL.WeightStore.Master[out*H+inIdx] = logP
				}
			}
		}
	}

	// Evaluation subset
	valEval := val
	if len(valEval) > EvalChars {
		valEval = valEval[:EvalChars]
	}
	testEval := test
	if len(testEval) > EvalChars {
		testEval = testEval[:EvalChars]
	}

	evalPPL := func(corpus []int) float64 {
		return evalPerplexityScaled(net, corpus, V, H)
	}

	rng := rand.New(rand.NewSource(42))
	seedChar := train[100]

	type GenResult struct {
		name   string
		valPPL float64
	}
	var history []GenResult

	// ── BASELINE: random weights (SCALED) ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  BASELINE  —  random weights (×%d scaled network)\n", ModelScale)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  Expected random perplexity ≈ %.0f  (vocab size)\n", float64(V))
	fmt.Println("  Evaluating...")
	pplxRandom := evalPPL(valEval)
	fmt.Printf("  Perplexity: %.2f\n\n", pplxRandom)
	fmt.Println("  Generated text:")
	printWrapped(generateTextScaled(net, seedChar, vocab, V, H, GenLength, 1.0, rng))
	fmt.Println()
	history = append(history, GenResult{fmt.Sprintf("Random (baseline, ×%d)", ModelScale), pplxRandom})

	// ── GEN 1: UNIGRAM (SCALED) ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  [GEN 1]  Hypothesis: character frequency is non-uniform.")
	fmt.Printf("           Action: install log-unigram, replicated across %d sub-clusters/char.\n", ModelScale)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	t0 := time.Now()
	unigram := computeUnigram(train, V)
	uniLogProbs := make([]float32, V)
	for j := 0; j < V; j++ {
		uniLogProbs[j] = float32(math.Log(float64(unigram[j])))
	}
	installIdentityKMeansScaled(0.1)
	installDenseUnigramScaled(uniLogProbs)
	gen1Time := time.Since(t0)
	pplxUnigram := evalPPL(valEval)
	fmt.Printf("  Snap time: %v  |  Perplexity: %.2f  (Δ=%.1f vs random)\n\n",
		gen1Time, pplxUnigram, pplxRandom-pplxUnigram)
	fmt.Println("  Generated text:")
	printWrapped(generateTextScaled(net, seedChar, vocab, V, H, GenLength, 1.0, rng))
	fmt.Println()
	history = append(history, GenResult{fmt.Sprintf("Unigram (×%d)", ModelScale), pplxUnigram})

	// Print top characters by frequency
	fmt.Println("  Top-8 chars by frequency:")
	type charFreq struct {
		c    byte
		freq float32
	}
	freqs := make([]charFreq, V)
	for i, f := range unigram {
		freqs[i] = charFreq{vocab[i], f}
	}
	sort.Slice(freqs, func(a, b int) bool { return freqs[a].freq > freqs[b].freq })
	for i := 0; i < 8 && i < len(freqs); i++ {
		label := fmt.Sprintf("%q", freqs[i].c)
		bar := progressBar(float64(freqs[i].freq), float64(freqs[0].freq), 30)
		fmt.Printf("    %-5s %.3f  %s\n", label, freqs[i].freq, bar)
	}
	fmt.Println()

	// ── GEN 2: BIGRAM (SCALED) ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  [GEN 2]  Hypothesis: adjacent character pairs carry strong signal.")
	fmt.Printf("           Action: install log bigram, %d sub-clusters per prev-char.\n", ModelScale)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	t0 = time.Now()
	bigram := computeBigram(train, V)
	installIdentityKMeansScaled(0.1)
	installDenseBigramScaled(bigram)
	gen2Time := time.Since(t0)
	pplxBigram := evalPPL(valEval)
	fmt.Printf("  Snap time: %v  |  Perplexity: %.2f  (Δ=%.1f vs unigram)\n\n",
		gen2Time, pplxBigram, pplxUnigram-pplxBigram)
	fmt.Println("  Generated text:")
	printWrapped(generateTextScaled(net, seedChar, vocab, V, H, GenLength, 1.0, rng))
	fmt.Println()
	history = append(history, GenResult{fmt.Sprintf("Bigram (T=0.10, ×%d)", ModelScale), pplxBigram})

	// Print top bigram transitions
	fmt.Println("  Top-8 bigram transitions (highest P(next|prev)):")
	type bigEntry struct {
		prev, next byte
		prob       float32
	}
	var topBi []bigEntry
	for i := 0; i < V; i++ {
		for j := 0; j < V; j++ {
			topBi = append(topBi, bigEntry{vocab[i], vocab[j], bigram[i][j]})
		}
	}
	sort.Slice(topBi, func(a, b int) bool { return topBi[a].prob > topBi[b].prob })
	for k := 0; k < 8 && k < len(topBi); k++ {
		fmt.Printf("    %q → %q  p=%.3f\n", topBi[k].prev, topBi[k].next, topBi[k].prob)
	}
	fmt.Println()

	// ── GEN 3: TEMPERATURE SWEEP (SCALED) ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  [GEN 3]  Hypothesis: KMeans sharpness affects context lookup quality.")
	fmt.Println("           Action: sweep temperatures, pick best on validation set.")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	temps := []float64{0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
	bestT := 0.1
	bestPPL := pplxBigram
	for _, t := range temps {
		installIdentityKMeansScaled(t)
		installDenseBigramScaled(bigram)
		p := evalPPL(valEval)
		marker := ""
		if p < bestPPL {
			bestPPL = p
			bestT = t
			marker = " ← best"
		}
		fmt.Printf("    T=%.3f  ppl=%.4f%s\n", t, p, marker)
	}
	fmt.Printf("\n  Best temperature: T=%.3f  val ppl=%.4f\n\n", bestT, bestPPL)

	installIdentityKMeansScaled(bestT)
	installDenseBigramScaled(bigram)
	history = append(history, GenResult{fmt.Sprintf("Bigram (T=%.2f, ×%d)", bestT, ModelScale), bestPPL})

	// ── GEN 4: TRIGRAM (SCALED, 2-char context) ──
	K := V * V * ModelScale // scaled trigram clusters
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  [GEN 4]  Hypothesis: doubling context + scaling capacity improves predictions.")
	fmt.Printf("           Action: 2-char context → %d clusters (%d²×%d), input=%d.\n", K, V, ModelScale, 2*V)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	t0 = time.Now()

	trigram := computeTrigram(train, V)

	// KMeans centers: each (c1,c2) pair → ModelScale sub-clusters
	kmeansL.NumClusters = K
	kmeansL.InputHeight = 2 * V
	kmeansL.OutputHeight = K
	kmeansL.KMeansTemperature = 0.1
	kmeansL.WeightStore = poly.NewWeightStore(K * 2 * V)
	for c1 := 0; c1 < V; c1++ {
		for c2 := 0; c2 < V; c2++ {
			base := (c1*V + c2) * ModelScale
			for s := 0; s < ModelScale; s++ {
				idx := base + s
				kmeansL.WeightStore.Master[idx*2*V+c1] = 1.0
				kmeansL.WeightStore.Master[idx*2*V+V+c2] = 1.0
				// Tiny noise for diversity
				for j := 0; j < 2*V; j++ {
					if j != c1 && j != V+c2 {
						kmeansL.WeightStore.Master[idx*2*V+j] = float32(s) * 1e-4
					}
				}
			}
		}
	}

	// Dense: replicate trigram probs across sub-clusters
	trigramW := make([]float32, V*K)
	for c1 := 0; c1 < V; c1++ {
		for c2 := 0; c2 < V; c2++ {
			ctx := c1*V + c2
			for s := 0; s < ModelScale; s++ {
				inIdx := ctx*ModelScale + s
				for out := 0; out < V; out++ {
					trigramW[out*K+inIdx] = float32(math.Log(float64(trigram[ctx][out])))
				}
			}
		}
	}
	denseL.InputHeight = K
	denseL.OutputHeight = V
	denseL.Activation = poly.ActivationLinear
	denseL.WeightStore = poly.NewWeightStore(V * K)
	copy(denseL.WeightStore.Master, trigramW)

	gen4Time := time.Since(t0)
	pplxTrigram := evalPerplexityTrigramScaled(net, valEval, V, K)
	fmt.Printf("  Snap time: %v  |  Perplexity: %.2f  (Δ=%.2f vs bigram)\n\n",
		gen4Time, pplxTrigram, bestPPL-pplxTrigram)
	fmt.Println("  Generated text (T=1.0):")
	seed2 := train[101]
	printWrapped(generateTextTrigramScaled(net, seedChar, seed2, vocab, V, K, GenLength, 1.0, rng))
	fmt.Println()
	history = append(history, GenResult{fmt.Sprintf("Trigram (2-char ctx, ×%d)", ModelScale), pplxTrigram})

	// ── TEST SET ──
	fmt.Println("[*] Final evaluation on held-out test set (trigram, scaled)...")
	pplxTest := evalPerplexityTrigramScaled(net, testEval, V, K)
	fmt.Printf("    Test perplexity: %.4f\n\n", pplxTest)

	// ── TEXT GENERATION SHOWCASE ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  TEXT GENERATION  —  trigram model, scaled capacity (2-char context)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	for _, sTemp := range []float64{0.5, 1.0, 1.5} {
		fmt.Printf("\n  Sampling temperature %.1f:\n", sTemp)
		printWrapped(generateTextTrigramScaled(net, seedChar, seed2, vocab, V, K, GenLength, sTemp, rng))
	}
	fmt.Println()

	// ── RESULTS TABLE ──
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           CHAR-LM SNAP  ·  RESULTS SUMMARY                    ║")
	fmt.Println("╠══════════════════════════════════════════════════╦══════════════╣")
	fmt.Println("║ Generation                                       ║  Perplexity  ║")
	fmt.Println("╠══════════════════════════════════════════════════╬══════════════╣")
	for _, r := range history {
		fmt.Printf("║ %-48s ║  %8.2f    ║\n", r.name, r.valPPL)
	}
	fmt.Println("╠══════════════════════════════════════════════════╬══════════════╣")
	fmt.Printf("║ %-48s ║  %8.2f    ║\n", "FINAL TEST SET (trigram, scaled)", pplxTest)
	fmt.Printf("║ %-48s ║  %8.2f    ║\n", fmt.Sprintf("Random baseline (vocab=%d)", V), float64(V))
	fmt.Println("╠══════════════════════════════════════════════════╩══════════════╣")
	fmt.Println("║  Backprop epochs : ZERO   Optimizer : NONE   Loss : NONE       ║")
	fmt.Printf("║  Model scale     : ×%d (hidden=%d, trigram clusters=%d)          ║\n", ModelScale, H, K)
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ── PERPLEXITY BAR ──
	fmt.Println("  Perplexity improvement (lower = better):")
	for _, r := range history {
		bar := pplxBar(r.valPPL, pplxRandom)
		fmt.Printf("  %-32s  ppl=%6.2f  %s\n", r.name, r.valPPL, bar)
	}
	fmt.Printf("  %-32s  ppl=%6.2f  (test, scaled)\n\n", "Trigram final", pplxTest)
}

// ── NETWORK CONSTRUCTION (SCALED) ────────────────────────────────────────────

func buildNetworkScaled(V, H int) *poly.VolumetricNetwork {
	jsonStr := fmt.Sprintf(`{
		"id": "lm_snap_scaled",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"sequential_layers": [
				{
					"type": "kmeans",
					"input_height": %d,
					"output_height": %d,
					"num_clusters": %d
				},
				{
					"type": "dense",
					"activation": "linear",
					"input_height": %d,
					"output_height": %d
				}
			]
		}]
	}`, V, H, H, H, V)
	net, err := poly.BuildNetworkFromJSON([]byte(jsonStr))
	if err != nil {
		panic(fmt.Sprintf("buildNetworkScaled: %v", err))
	}
	return net
}

// ── DATA ────────────────────────────────────────────────────────────────────

func ensureCorpus() error {
	if _, err := os.Stat(CorpusFile); err == nil {
		return nil
	}
	if err := os.MkdirAll(DataDir, 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	fmt.Println("[*] Downloading Tiny Shakespeare (~1.1MB)...")
	resp, err := http.Get(CorpusURL)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	f, err := os.Create(CorpusFile)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()
	nb, err := io.Copy(f, resp.Body)
	if err != nil {
		return fmt.Errorf("write: %w", err)
	}
	fmt.Printf("[*] Downloaded %d bytes\n", nb)
	return nil
}

func loadCorpus() (corpus []int, vocab []byte, charToIdx map[byte]int) {
	data, err := os.ReadFile(CorpusFile)
	if err != nil {
		panic(err)
	}
	seen := make(map[byte]bool)
	for _, b := range data {
		seen[b] = true
	}
	vocab = make([]byte, 0, len(seen))
	for b := range seen {
		vocab = append(vocab, b)
	}
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx = make(map[byte]int, len(vocab))
	for i, c := range vocab {
		charToIdx[c] = i
	}
	corpus = make([]int, len(data))
	for i, b := range data {
		corpus[i] = charToIdx[b]
	}
	return
}

// ── STATISTICS ──────────────────────────────────────────────────────────────

func computeTrigram(corpus []int, V int) [][]float32 {
	K := V * V
	counts := make([][]int, K)
	for i := range counts {
		counts[i] = make([]int, V)
	}
	for i := 2; i < len(corpus); i++ {
		ctx := corpus[i-2]*V + corpus[i-1]
		counts[ctx][corpus[i]]++
	}
	trig := make([][]float32, K)
	for k := range trig {
		trig[k] = make([]float32, V)
		total := V
		for _, c := range counts[k] {
			total += c
		}
		for j := range trig[k] {
			trig[k][j] = float32(counts[k][j]+1) / float32(total)
		}
	}
	return trig
}

func evalPerplexityTrigramScaled(net *poly.VolumetricNetwork, corpus []int, V, K int) float64 {
	if len(corpus) < 3 {
		return math.Inf(1)
	}
	totalLogProb := 0.0
	input := poly.NewTensor[float32](1, 2*V)
	for i := 2; i < len(corpus); i++ {
		for k := range input.Data {
			input.Data[k] = 0
		}
		input.Data[corpus[i-2]] = 1.0
		input.Data[V+corpus[i-1]] = 1.0
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmax(output.Data)
		p := float64(probs[corpus[i]])
		if p < 1e-10 {
			p = 1e-10
		}
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-2))
}

func generateTextTrigramScaled(net *poly.VolumetricNetwork, seed1, seed2 int, vocab []byte, V, K, length int, sampTemp float64, rng *rand.Rand) string {
	result := make([]byte, 0, length+2)
	result = append(result, vocab[seed1], vocab[seed2])
	prev2, prev1 := seed1, seed2
	input := poly.NewTensor[float32](1, 2*V)
	for i := 0; i < length; i++ {
		for k := range input.Data {
			input.Data[k] = 0
		}
		input.Data[prev2] = 1.0
		input.Data[V+prev1] = 1.0
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTemp(output.Data, sampTemp)
		next := sampleCategorical(probs, rng)
		result = append(result, vocab[next])
		prev2 = prev1
		prev1 = next
	}
	return string(result)
}

func computeUnigram(corpus []int, V int) []float32 {
	counts := make([]int, V)
	for _, c := range corpus {
		counts[c]++
	}
	total := len(corpus) + V
	probs := make([]float32, V)
	for i, c := range counts {
		probs[i] = float32(c+1) / float32(total)
	}
	return probs
}

func computeBigram(corpus []int, V int) [][]float32 {
	counts := make([][]int, V)
	for i := range counts {
		counts[i] = make([]int, V)
	}
	for i := 1; i < len(corpus); i++ {
		counts[corpus[i-1]][corpus[i]]++
	}
	bigram := make([][]float32, V)
	for i := range bigram {
		bigram[i] = make([]float32, V)
		total := V
		for _, c := range counts[i] {
			total += c
		}
		for j := range bigram[i] {
			bigram[i][j] = float32(counts[i][j]+1) / float32(total)
		}
	}
	return bigram
}

// ── EVALUATION (SCALED) ──────────────────────────────────────────────────────

func evalPerplexityScaled(net *poly.VolumetricNetwork, corpus []int, V, H int) float64 {
	if len(corpus) < 2 {
		return math.Inf(1)
	}
	totalLogProb := 0.0
	input := poly.NewTensor[float32](1, V)
	for i := 1; i < len(corpus); i++ {
		for k := range input.Data {
			input.Data[k] = 0
		}
		input.Data[corpus[i-1]] = 1.0
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmax(output.Data)
		p := float64(probs[corpus[i]])
		if p < 1e-10 {
			p = 1e-10
		}
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-1))
}

// ── TEXT GENERATION (SCALED) ─────────────────────────────────────────────────

func generateTextScaled(net *poly.VolumetricNetwork, seed int, vocab []byte, V, H, length int, sampTemp float64, rng *rand.Rand) string {
	result := make([]byte, 0, length)
	result = append(result, vocab[seed])
	prev := seed
	input := poly.NewTensor[float32](1, V)
	for i := 0; i < length; i++ {
		for k := range input.Data {
			input.Data[k] = 0
		}
		input.Data[prev] = 1.0
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTemp(output.Data, sampTemp)
		next := sampleCategorical(probs, rng)
		result = append(result, vocab[next])
		prev = next
	}
	return string(result)
}

// ── MATH HELPERS ────────────────────────────────────────────────────────────

func softmax(v []float32) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v {
		if x > maxV {
			maxV = x
		}
	}
	sum := float32(0)
	for i, x := range v {
		out[i] = float32(math.Exp(float64(x - maxV)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func softmaxTemp(v []float32, temp float64) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v {
		if x > maxV {
			maxV = x
		}
	}
	sum := float32(0)
	for i, x := range v {
		out[i] = float32(math.Exp(float64(x-maxV) / temp))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func sampleCategorical(probs []float32, rng *rand.Rand) int {
	u := rng.Float32()
	cum := float32(0)
	for i, p := range probs {
		cum += p
		if u < cum {
			return i
		}
	}
	return len(probs) - 1
}

// ── DISPLAY HELPERS ─────────────────────────────────────────────────────────

func progressBar(val, max float64, width int) string {
	filled := int(val / max * float64(width))
	if filled > width {
		filled = width
	}
	bar := make([]byte, width)
	for i := range bar {
		if i < filled {
			bar[i] = '#'
		} else {
			bar[i] = '.'
		}
	}
	return string(bar)
}

func pplxBar(ppl, worst float64) string {
	const width = 30
	ratio := 1.0 - ppl/worst
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	n := int(ratio * width)
	bar := make([]byte, width)
	for i := range bar {
		if i < n {
			bar[i] = '#'
		} else {
			bar[i] = '.'
		}
	}
	return string(bar)
}

func printWrapped(text string) {
	const lineWidth = 72
	runes := []rune(text)
	for i := 0; i < len(runes); i += lineWidth {
		end := i + lineWidth
		if end > len(runes) {
			end = len(runes)
		}
		fmt.Printf("  %s\n", string(runes[i:end]))
	}
}
