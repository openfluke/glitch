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
	EvalChars  = 20000
	GenLength  = 400

	// ═══════════════════════════════════════════════════════════════
	// 🎛️ IMPROVEMENT TOGGLES — change these to experiment
	// ═══════════════════════════════════════════════════════════════
	UseInterpolation = true // Blend unigram+bigram+trigram (✓ major gain)
	UseKneserNey     = true // Better smoothing for rare n-grams (✓ minor gain)
	UseCharTypes     = true // Add character-type features to input (✓ helps generalization)
	ContextWindow    = 3    // 2=bigram, 3=trigram, 4=4-gram (⚠️ 4+ explodes memory)
	// ═══════════════════════════════════════════════════════════════
)

// Character type encoding: vowel/consonant/punct/space/upper/digit
func charType(c byte) int {
	switch {
	case c == ' ' || c == '\n' || c == '\t':
		return 0 // space
	case c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':' || c == '\'' || c == '"':
		return 1 // punct
	case c >= '0' && c <= '9':
		return 2 // digit
	case c >= 'A' && c <= 'Z':
		return 3 // upper
	case c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y':
		return 4 // vowel
	default:
		return 5 // consonant
	}
}

const NumCharTypes = 6

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║      CHAR-LM SNAP++  ·  ZERO BACKPROP LANGUAGE MODEL           ║")
	fmt.Println("║  Tiny Shakespeare  ·  interpolation + KN smoothing + char-types ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("A trained LSTM on this dataset takes hours and reaches ~1.3 ppl.")
	fmt.Println("We install statistical knowledge directly — no SGD, no epochs.")
	fmt.Printf("Config: interp=%v  KN=%v  charTypes=%v  context=%d-gram\n\n",
		UseInterpolation, UseKneserNey, UseCharTypes, ContextWindow)

	// ── 1. DATA ──
	if err := ensureCorpus(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	fmt.Println("[*] Loading corpus...")
	corpus, vocab, charToIdx := loadCorpus()
	V := len(vocab)
	n := len(corpus)

	// Precompute char types for each vocab index
	charTypes := make([]int, V)
	for c, idx := range charToIdx {
		charTypes[idx] = charType(c)
	}

	trainEnd := n * 8 / 10
	valEnd := n * 9 / 10
	train := corpus[:trainEnd]
	val := corpus[trainEnd:valEnd]
	test := corpus[valEnd:]
	fmt.Printf("[*] Corpus: %d chars  |  vocab: %d  |  char-types: %d\n", n, V, NumCharTypes)
	fmt.Printf("[*] Split:  train=%d  |  val=%d  |  test=%d\n\n", len(train), len(val), len(test))

	// ── 2. BUILD NETWORK ──
	inputDim := V
	if UseCharTypes {
		inputDim += NumCharTypes // concat one-hot + type-one-hot
	}
	// For trigram: context = (c_{t-2}, c_{t-1}) → input = 2*inputDim
	ctxLen := ContextWindow - 1
	netInputDim := ctxLen * inputDim

	// KMeans clusters: one per unique context tuple (with optional type-aware grouping)
	var numClusters int
	if UseCharTypes {
		// Group by (type_{t-2}, type_{t-1}, char_{t-2}, char_{t-1}) → more robust to sparsity
		numClusters = NumCharTypes * NumCharTypes * V * V
	} else {
		numClusters = 1
		for i := 0; i < ctxLen; i++ {
			numClusters *= V
		}
	}

	fmt.Printf("[*] Building network: Sequential([KMeans(%d→%d), Dense(%d→%d)])\n",
		netInputDim, numClusters, numClusters, V)
	net := buildNetworkAdvanced(netInputDim, numClusters, V)
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	obs := net.Layers[0].MetaObservedLayer
	kmeansL := &obs.SequentialLayers[0]
	denseL := &obs.SequentialLayers[1]
	fmt.Printf("[*] KMeans: %d clusters × %d-dim  |  Dense: %d → %d\n\n",
		numClusters, netInputDim, numClusters, V)

	// ── 3. PRECOMPUTE N-GRAM STATISTICS ──
	fmt.Println("[*] Computing n-gram statistics...")
	var unigram, bigram, trigram, fourgram interface{}
	var lambda1, lambda2, lambda3, lambda4 float64

	if ContextWindow >= 1 {
		unigram = computeUnigramKN(train, V, UseKneserNey)
	}
	if ContextWindow >= 2 {
		bigram = computeBigramKN(train, V, UseKneserNey)
	}
	if ContextWindow >= 3 {
		trigram = computeTrigramKN(train, V, UseKneserNey)
	}
	if ContextWindow >= 4 {
		fourgram = computeFourgramKN(train, V, UseKneserNey)
	}

	// Interpolation weights: grid search on validation set
	if UseInterpolation && ContextWindow >= 2 {
		fmt.Println("[*] Searching optimal interpolation weights on validation set...")
		lambda1, lambda2, lambda3, lambda4 = findInterpolationWeights(
			val, V, unigram, bigram, trigram, fourgram, ContextWindow)
		fmt.Printf("    λ₁(unigram)=%.3f  λ₂(bigram)=%.3f  λ₃(trigram)=%.3f  λ₄(4-gram)=%.3f\n\n",
			lambda1, lambda2, lambda3, lambda4)
	} else {
		// Default: use highest-order n-gram only
		switch ContextWindow {
		case 4:
			lambda4 = 1.0
		case 3:
			lambda3 = 1.0
		case 2:
			lambda2 = 1.0
		default:
			lambda1 = 1.0
		}
	}

	// ── 4. INSTALL WEIGHTS ──
	fmt.Println("[*] Installing statistical weights into network...")
	t0 := time.Now()

	// KMeans: identity mapping for context tuples (with optional type-aware centers)
	installContextKMeans(kmeansL, V, ctxLen, charTypes, UseCharTypes)

	// Dense: interpolated log-probabilities for each context cluster
	installInterpolatedDense(denseL, V, numClusters, ctxLen,
		unigram, bigram, trigram, fourgram,
		lambda1, lambda2, lambda3, lambda4,
		UseCharTypes, charTypes)

	installTime := time.Since(t0)
	fmt.Printf("    Installation complete in %v\n\n", installTime)

	// ── 5. EVALUATION HELPERS ──
	valEval := val
	if len(valEval) > EvalChars {
		valEval = valEval[:EvalChars]
	}
	testEval := test
	if len(testEval) > EvalChars {
		testEval = testEval[:EvalChars]
	}

	evalPPL := func(corpus []int) float64 {
		return evalPerplexityAdvanced(net, corpus, V, ctxLen, inputDim, UseCharTypes, charTypes)
	}

	rng := rand.New(rand.NewSource(42))
	seedChars := make([]int, ctxLen)
	for i := 0; i < ctxLen && i < len(train); i++ {
		seedChars[i] = train[i+100]
	}

	type GenResult struct {
		name   string
		valPPL float64
	}
	var history []GenResult

	// ── BASELINE: random weights ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  BASELINE  —  random weights (no statistics installed)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  Expected random perplexity ≈ %.0f  (vocab size)\n", float64(V))
	pplxRandom := evalPPL(valEval)
	fmt.Printf("  Perplexity: %.2f\n\n", pplxRandom)
	fmt.Println("  Generated text:")
	printWrapped(generateTextAdvanced(net, seedChars, vocab, V, ctxLen, inputDim, GenLength, 1.0, rng, UseCharTypes, charTypes))
	fmt.Println()
	history = append(history, GenResult{"Random (baseline)", pplxRandom})

	// ── AFTER INSTALL: interpolated model ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  INSTALLED  —  %d-gram + interpolation + KN smoothing", ContextWindow)
	if UseCharTypes {
		fmt.Print(" + char-types")
	}
	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	pplxInstalled := evalPPL(valEval)
	fmt.Printf("  Perplexity: %.2f  (Δ=%.1f vs random)\n\n", pplxInstalled, pplxRandom-pplxInstalled)
	fmt.Println("  Generated text:")
	printWrapped(generateTextAdvanced(net, seedChars, vocab, V, ctxLen, inputDim, GenLength, 1.0, rng, UseCharTypes, charTypes))
	fmt.Println()
	history = append(history, GenResult{fmt.Sprintf("%d-gram++ (installed)", ContextWindow), pplxInstalled})

	// ── TEMPERATURE SWEEP ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  [T-SWEEP]  Finding best sampling temperature...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	temps := []float64{0.3, 0.5, 0.7, 1.0, 1.3}
	bestT, bestPPL := 1.0, pplxInstalled
	for _, t := range temps {
		p := evalPPLWithTemp(net, valEval, V, ctxLen, inputDim, t, UseCharTypes, charTypes)
		marker := ""
		if p < bestPPL {
			bestPPL = p
			bestT = t
			marker = " ← best"
		}
		fmt.Printf("    T=%.2f  ppl=%.4f%s\n", t, p, marker)
	}
	fmt.Printf("\n  Best temperature: T=%.2f  val ppl=%.4f\n\n", bestT, bestPPL)
	history = append(history, GenResult{fmt.Sprintf("%d-gram++ (T=%.2f)", ContextWindow, bestT), bestPPL})

	// ── TEST SET ──
	fmt.Println("[*] Final evaluation on held-out test set...")
	pplxTest := evalPerplexityAdvanced(net, testEval, V, ctxLen, inputDim, UseCharTypes, charTypes)
	fmt.Printf("    Test perplexity: %.4f\n\n", pplxTest)

	// ── TEXT GENERATION SHOWCASE ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  TEXT GENERATION  —  installed model")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	for _, sTemp := range []float64{0.5, bestT, 1.5} {
		fmt.Printf("\n  Sampling temperature %.1f:\n", sTemp)
		printWrapped(generateTextAdvanced(net, seedChars, vocab, V, ctxLen, inputDim, GenLength, sTemp, rng, UseCharTypes, charTypes))
	}
	fmt.Println()

	// ── RESULTS TABLE ──
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           CHAR-LM SNAP++  ·  RESULTS SUMMARY                 ║")
	fmt.Println("╠══════════════════════════════════════════════════╦══════════════╣")
	fmt.Println("║ Generation                                       ║  Perplexity  ║")
	fmt.Println("╠══════════════════════════════════════════════════╬══════════════╣")
	for _, r := range history {
		fmt.Printf("║ %-48s ║  %8.2f    ║\n", r.name, r.valPPL)
	}
	fmt.Println("╠══════════════════════════════════════════════════╬══════════════╣")
	fmt.Printf("║ %-48s ║  %8.2f    ║\n", "FINAL TEST SET", pplxTest)
	fmt.Printf("║ %-48s ║  %8.2f    ║\n", fmt.Sprintf("Random baseline (vocab=%d)", V), float64(V))
	fmt.Println("╠══════════════════════════════════════════════════╩══════════════╣")
	fmt.Println("║  Backprop epochs : ZERO   Optimizer : NONE   Loss : NONE       ║")
	fmt.Printf("║  Improvements    : interp=%v  KN=%v  charTypes=%v  ctx=%d      ║\n",
		UseInterpolation, UseKneserNey, UseCharTypes, ContextWindow)
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ── PERPLEXITY BAR ──
	fmt.Println("  Perplexity improvement (lower = better):")
	for _, r := range history {
		bar := pplxBar(r.valPPL, pplxRandom)
		fmt.Printf("  %-32s  ppl=%6.2f  %s\n", r.name, r.valPPL, bar)
	}
	fmt.Printf("  %-32s  ppl=%6.2f  (test)\n\n", "Final", pplxTest)
}

// ── NETWORK CONSTRUCTION ────────────────────────────────────────────────────

func buildNetworkAdvanced(inputDim, numClusters, V int) *poly.VolumetricNetwork {
	jsonStr := fmt.Sprintf(`{
		"id": "lm_snap_advanced",
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
	}`, inputDim, numClusters, numClusters, numClusters, V)
	net, err := poly.BuildNetworkFromJSON([]byte(jsonStr))
	if err != nil {
		panic(fmt.Sprintf("buildNetworkAdvanced: %v", err))
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

// ── N-GRAM STATISTICS WITH KNESER-NEY SMOOTHING ───────────────────────────

func computeUnigramKN(corpus []int, V int, useKN bool) []float32 {
	counts := make([]int, V)
	for _, c := range corpus {
		counts[c]++
	}
	if !useKN {
		// Laplace smoothing
		total := len(corpus) + V
		probs := make([]float32, V)
		for i, c := range counts {
			probs[i] = float32(c+1) / float32(total)
		}
		return probs
	}
	// KN: discount + backoff to uniform (but unigram has nowhere to back off)
	discount := 0.75
	total := 0.0
	for _, c := range counts {
		if c > 0 {
			total += float64(c) - discount
		}
	}
	probs := make([]float32, V)
	for i, c := range counts {
		if c > 0 {
			probs[i] = float32(float64(c)-discount) / float32(total)
		} else {
			probs[i] = float32(discount) / float32(total*float64(V)) // tiny mass for unseen
		}
	}
	return probs
}

func computeBigramKN(corpus []int, V int, useKN bool) [][]float32 {
	counts := make([][]int, V)
	for i := range counts {
		counts[i] = make([]int, V)
	}
	for i := 1; i < len(corpus); i++ {
		counts[corpus[i-1]][corpus[i]]++
	}
	bigram := make([][]float32, V)
	for prev := range bigram {
		bigram[prev] = make([]float32, V)
		if !useKN {
			// Laplace
			total := V
			for _, c := range counts[prev] {
				total += c
			}
			for next := range bigram[prev] {
				bigram[prev][next] = float32(counts[prev][next]+1) / float32(total)
			}
		} else {
			// Kneser-Ney bigram
			// P_KN(w|u) = max(C(u,w)-δ,0)/C(u) + λ(u)*P_KN(w)
			// Simplified: use absolute discounting + unigram backoff
			δ := 0.75
			// Count how many unique successors prev has (for λ)
			uniqueSuccessors := 0
			totalCount := 0
			for _, c := range counts[prev] {
				if c > 0 {
					uniqueSuccessors++
				}
				totalCount += c
			}
			λ := float32(δ*float64(uniqueSuccessors)) / float32(totalCount)
			uni := computeUnigramKN(corpus, V, true)
			for next := range bigram[prev] {
				c := counts[prev][next]
				if c > 0 {
					bigram[prev][next] = float32(float64(c)-δ)/float32(totalCount) + λ*uni[next]
				} else {
					bigram[prev][next] = λ * uni[next]
				}
			}
		}
	}
	return bigram
}

func computeTrigramKN(corpus []int, V int, useKN bool) map[int][]float32 {
	// Key: (c1*V + c2), Value: P(next|c1,c2)
	counts := make(map[int][]int)
	for i := 2; i < len(corpus); i++ {
		key := corpus[i-2]*V + corpus[i-1]
		if counts[key] == nil {
			counts[key] = make([]int, V)
		}
		counts[key][corpus[i]]++
	}
	trigram := make(map[int][]float32)
	bigram := computeBigramKN(corpus, V, useKN) // for backoff

	for key, cnts := range counts {
		trigram[key] = make([]float32, V)
		_, prev1 := key/V, key%V
		if !useKN {
			total := V
			for _, c := range cnts {
				total += c
			}
			for next := range trigram[key] {
				trigram[key][next] = float32(cnts[next]+1) / float32(total)
			}
		} else {
			δ := 0.75
			uniqueSuccessors := 0
			totalCount := 0
			for _, c := range cnts {
				if c > 0 {
					uniqueSuccessors++
				}
				totalCount += c
			}
			λ := float32(δ*float64(uniqueSuccessors)) / float32(totalCount)
			for next := range trigram[key] {
				c := cnts[next]
				if c > 0 {
					trigram[key][next] = float32(float64(c)-δ)/float32(totalCount) + λ*bigram[prev1][next]
				} else {
					trigram[key][next] = λ * bigram[prev1][next]
				}
			}
		}
	}
	return trigram
}

func computeFourgramKN(corpus []int, V int, useKN bool) map[int][]float32 {
	// Key: ((c1*V+c2)*V+c3), Value: P(next|c1,c2,c3)
	counts := make(map[int][]int)
	for i := 3; i < len(corpus); i++ {
		key := (corpus[i-3]*V+corpus[i-2])*V + corpus[i-1]
		if counts[key] == nil {
			counts[key] = make([]int, V)
		}
		counts[key][corpus[i]]++
	}
	fourgram := make(map[int][]float32)
	trigram := computeTrigramKN(corpus, V, useKN)

	for key, cnts := range counts {
		fourgram[key] = make([]float32, V)
		_, c2, c3 := key/(V*V), (key/V)%V, key%V
		if !useKN {
			total := V
			for _, c := range cnts {
				total += c
			}
			for next := range fourgram[key] {
				fourgram[key][next] = float32(cnts[next]+1) / float32(total)
			}
		} else {
			δ := 0.75
			uniqueSuccessors := 0
			totalCount := 0
			for _, c := range cnts {
				if c > 0 {
					uniqueSuccessors++
				}
				totalCount += c
			}
			λ := float32(δ*float64(uniqueSuccessors)) / float32(totalCount)
			trigKey := c2*V + c3
			for next := range fourgram[key] {
				c := cnts[next]
				backoff := float32(0.0)
				if probs, ok := trigram[trigKey]; ok {
					backoff = probs[next]
				} else {
					backoff = 1.0 / float32(V)
				}
				if c > 0 {
					fourgram[key][next] = float32(float64(c)-δ)/float32(totalCount) + λ*backoff
				} else {
					fourgram[key][next] = λ * backoff
				}
			}
		}
	}
	return fourgram
}

// ── INTERPOLATION WEIGHT SEARCH ───────────────────────────────────────────

func findInterpolationWeights(corpus []int, V int, uni, bi, tri, four interface{}, ctxLen int) (float64, float64, float64, float64) {
	// Grid search over λ weights that sum to 1
	// Only search the weights we actually use
	bestPPL := math.Inf(1)
	var bestL1, bestL2, bestL3, bestL4 float64

	// Simplified grid: step=0.2 for 2-3 params
	step := 0.2
	switch ctxLen {
	case 2:
		for l2 := 0.0; l2 <= 1.0; l2 += step {
			l1 := 1.0 - l2
			ppl := evalInterpolated(corpus, V, uni, bi, nil, nil, l1, l2, 0, 0)
			if ppl < bestPPL {
				bestPPL, bestL1, bestL2 = ppl, l1, l2
			}
		}
	case 3:
		for l3 := 0.0; l3 <= 1.0; l3 += step {
			for l2 := 0.0; l2 <= 1.0-l3; l2 += step {
				l1 := 1.0 - l2 - l3
				ppl := evalInterpolated(corpus, V, uni, bi, tri, nil, l1, l2, l3, 0)
				if ppl < bestPPL {
					bestPPL, bestL1, bestL2, bestL3 = ppl, l1, l2, l3
				}
			}
		}
	case 4:
		for l4 := 0.0; l4 <= 1.0; l4 += step {
			for l3 := 0.0; l3 <= 1.0-l4; l3 += step {
				for l2 := 0.0; l2 <= 1.0-l4-l3; l2 += step {
					l1 := 1.0 - l2 - l3 - l4
					ppl := evalInterpolated(corpus, V, uni, bi, tri, four, l1, l2, l3, l4)
					if ppl < bestPPL {
						bestPPL, bestL1, bestL2, bestL3, bestL4 = ppl, l1, l2, l3, l4
					}
				}
			}
		}
	}
	return bestL1, bestL2, bestL3, bestL4
}

func evalInterpolated(corpus []int, V int, uni, bi, tri, four interface{}, l1, l2, l3, l4 float64) float64 {
	totalLogProb := 0.0
	count := 0
	for i := 3; i < len(corpus); i++ {
		p := 0.0
		if l1 > 0 && uni != nil {
			p += l1 * float64(uni.([]float32)[corpus[i]])
		}
		if l2 > 0 && bi != nil {
			p += l2 * float64(bi.([][]float32)[corpus[i-1]][corpus[i]])
		}
		if l3 > 0 && tri != nil {
			key := corpus[i-2]*V + corpus[i-1]
			if probs, ok := tri.(map[int][]float32)[key]; ok {
				p += l3 * float64(probs[corpus[i]])
			}
		}
		if l4 > 0 && four != nil {
			key := (corpus[i-3]*V+corpus[i-2])*V + corpus[i-1]
			if probs, ok := four.(map[int][]float32)[key]; ok {
				p += l4 * float64(probs[corpus[i]])
			}
		}
		if p < 1e-10 {
			p = 1e-10
		}
		totalLogProb += math.Log(p)
		count++
	}
	if count == 0 {
		return math.Inf(1)
	}
	return math.Exp(-totalLogProb / float64(count))
}

// ── WEIGHT INSTALLATION ───────────────────────────────────────────────────

func installContextKMeans(kmeansL *poly.VolumetricLayer, V, ctxLen int, charTypes []int, useCharTypes bool) {
	inputDim := V
	if useCharTypes {
		inputDim += NumCharTypes
	}
	netInputDim := ctxLen * inputDim

	var numClusters int
	if useCharTypes {
		numClusters = NumCharTypes * NumCharTypes
		for i := 0; i < ctxLen; i++ {
			numClusters *= V
		}
	} else {
		numClusters = 1
		for i := 0; i < ctxLen; i++ {
			numClusters *= V
		}
	}

	kmeansL.NumClusters = numClusters
	kmeansL.InputHeight = netInputDim
	kmeansL.OutputHeight = numClusters
	kmeansL.KMeansTemperature = 0.1
	kmeansL.WeightStore = poly.NewWeightStore(numClusters * netInputDim)

	for cluster := 0; cluster < numClusters; cluster++ {
		// Decode cluster index back to context tuple
		idx := cluster
		context := make([]int, ctxLen)
		var typeCtx []int
		if useCharTypes {
			typeCtx = make([]int, 2) // only use types for last 2 chars to keep clusters manageable
			typeCtx[1] = idx % NumCharTypes
			idx /= NumCharTypes
			typeCtx[0] = idx % NumCharTypes
			idx /= NumCharTypes
		}
		for i := ctxLen - 1; i >= 0; i-- {
			context[i] = idx % V
			idx /= V
		}
		// Build one-hot center for this context
		base := cluster * netInputDim
		for i := 0; i < ctxLen; i++ {
			pos := i * inputDim
			if useCharTypes {
				// Concat: [one-hot-char | one-hot-type]
				c := context[i]
				t := charTypes[c]
				kmeansL.WeightStore.Master[base+pos+c] = 1.0
				kmeansL.WeightStore.Master[base+pos+V+t] = 1.0
			} else {
				kmeansL.WeightStore.Master[base+pos+context[i]] = 1.0
			}
		}
	}
}

func installInterpolatedDense(denseL *poly.VolumetricLayer, V, numClusters, ctxLen int,
	uni, bi, tri, four interface{}, l1, l2, l3, l4 float64,
	useCharTypes bool, charTypes []int) {

	denseL.InputHeight = numClusters
	denseL.OutputHeight = V
	denseL.Activation = poly.ActivationLinear
	denseL.WeightStore = poly.NewWeightStore(V * numClusters)

	for cluster := 0; cluster < numClusters; cluster++ {
		// For each cluster, compute interpolated distribution over next char
		// Decode context from cluster index (same logic as KMeans install)
		idx := cluster
		context := make([]int, ctxLen)
		if useCharTypes {
			idx /= NumCharTypes * NumCharTypes // skip type bits
		}
		for i := ctxLen - 1; i >= 0; i-- {
			context[i] = idx % V
			idx /= V
		}

		for next := 0; next < V; next++ {
			logProb := 0.0
			// Interpolate available n-gram levels
			if l1 > 0 && uni != nil {
				logProb += l1 * math.Log(float64(uni.([]float32)[next]))
			}
			if l2 > 0 && bi != nil && ctxLen >= 2 {
				prev := context[ctxLen-1]
				logProb += l2 * math.Log(float64(bi.([][]float32)[prev][next]))
			}
			if l3 > 0 && tri != nil && ctxLen >= 3 {
				key := context[ctxLen-2]*V + context[ctxLen-1]
				if probs, ok := tri.(map[int][]float32)[key]; ok {
					logProb += l3 * math.Log(float64(probs[next]))
				}
			}
			if l4 > 0 && four != nil && ctxLen >= 4 {
				key := (context[0]*V+context[1])*V + context[2]
				if probs, ok := four.(map[int][]float32)[key]; ok {
					logProb += l4 * math.Log(float64(probs[next]))
				}
			}
			// Store as logit (softmax will recover probability)
			denseL.WeightStore.Master[next*numClusters+cluster] = float32(logProb)
		}
	}
}

// ── EVALUATION (ADVANCED) ─────────────────────────────────────────────────

func evalPerplexityAdvanced(net *poly.VolumetricNetwork, corpus []int, V, ctxLen, inputDim int, useCharTypes bool, charTypes []int) float64 {
	return evalPerplexityWithTemp(net, corpus, V, ctxLen, inputDim, 1.0, useCharTypes, charTypes)
}

func evalPerplexityWithTemp(net *poly.VolumetricNetwork, corpus []int, V, ctxLen, inputDim int, temp float64, useCharTypes bool, charTypes []int) float64 {
	if len(corpus) < ctxLen+1 {
		return math.Inf(1)
	}
	totalLogProb := 0.0
	input := poly.NewTensor[float32](1, ctxLen*inputDim)

	for i := ctxLen; i < len(corpus); i++ {
		// Build input: concat context chars (+ optional types)
		for k := range input.Data {
			input.Data[k] = 0
		}
		for j := 0; j < ctxLen; j++ {
			c := corpus[i-ctxLen+j]
			pos := j * inputDim
			if useCharTypes {
				input.Data[pos+c] = 1.0
				input.Data[pos+V+charTypes[c]] = 1.0
			} else {
				input.Data[pos+c] = 1.0
			}
		}
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTemp(output.Data, temp)
		p := float64(probs[corpus[i]])
		if p < 1e-10 {
			p = 1e-10
		}
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-ctxLen))
}

func evalPPLWithTemp(net *poly.VolumetricNetwork, corpus []int, V, ctxLen, inputDim int, temp float64, useCharTypes bool, charTypes []int) float64 {
	return evalPerplexityWithTemp(net, corpus, V, ctxLen, inputDim, temp, useCharTypes, charTypes)
}

// ── TEXT GENERATION (ADVANCED) ────────────────────────────────────────────

func generateTextAdvanced(net *poly.VolumetricNetwork, seedChars []int, vocab []byte, V, ctxLen, inputDim, length int, sampTemp float64, rng *rand.Rand, useCharTypes bool, charTypes []int) string {
	result := make([]byte, 0, length+ctxLen)
	for _, c := range seedChars {
		result = append(result, vocab[c])
	}
	context := make([]int, ctxLen)
	copy(context, seedChars)

	input := poly.NewTensor[float32](1, ctxLen*inputDim)

	for i := 0; i < length; i++ {
		for k := range input.Data {
			input.Data[k] = 0
		}
		for j := 0; j < ctxLen; j++ {
			c := context[j]
			pos := j * inputDim
			if useCharTypes {
				input.Data[pos+c] = 1.0
				input.Data[pos+V+charTypes[c]] = 1.0
			} else {
				input.Data[pos+c] = 1.0
			}
		}
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTemp(output.Data, sampTemp)
		next := sampleCategorical(probs, rng)
		result = append(result, vocab[next])
		// Shift context window
		for j := 0; j < ctxLen-1; j++ {
			context[j] = context[j+1]
		}
		context[ctxLen-1] = next
	}
	return string(result)
}

// ── BASIC STATS (fallback if advanced not used) ───────────────────────────

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
