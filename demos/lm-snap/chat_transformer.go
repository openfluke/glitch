package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	CorpusFile = "data/shakespeare.txt"
	CtxLen     = 32  // Long transformer-style context
	SnapLen    = 5   // N-gram length for the snap (6-gram)
	HiddenSize = 128 // Embedding/Hidden dimension
	V          = 65  // Vocab size for Shakespeare
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║             SNAP TRANSFORMER CHAT  ·  33-GRAM                ║")
	fmt.Println("║  Multi-Head Attention + KV Cache  ·  Zero-Backprop Engine    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// 1. Load and prepare corpus
	data, err := os.ReadFile(CorpusFile)
	if err != nil {
		fmt.Printf("[!] Error: shakespeare.txt not found. Run 'go run main.go' first.\n")
		return
	}
	// Extract unique vocab manually to handle any file
	seenChars := make(map[byte]bool)
	for _, b := range data { seenChars[b] = true }
	var vocab []byte
	for b := range seenChars { vocab = append(vocab, b) }
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx := make(map[byte]int)
	for i, c := range vocab { charToIdx[c] = i }
	
	corpus := make([]int, len(data))
	for i, b := range data { corpus[i] = charToIdx[b] }
	vActual := len(vocab)

	fmt.Printf("[*] Loading and Snapping transformer (ctxLen=%d, vocab=%d)...\n", CtxLen, vActual)
	
	// 2. Build Hybrid Architecture
	net := poly.NewVolumetricNetwork(1, 1, 1, 5)
	
	// Layer 0: Embedding (65 -> 128)
	lEmbed := net.GetLayer(0, 0, 0, 0)
	lEmbed.Type = poly.LayerEmbedding
	lEmbed.VocabSize = vActual
	lEmbed.EmbeddingDim = HiddenSize
	lEmbed.WeightStore = poly.NewWeightStore(vActual * HiddenSize)
	for i := 0; i < vActual; i++ { lEmbed.WeightStore.Master[i*HiddenSize+i] = 1.0 }

	// Layer 1: RMSNorm
	lNorm := net.GetLayer(0, 0, 0, 1)
	lNorm.Type = poly.LayerRMSNorm
	lNorm.InputHeight = HiddenSize
	lNorm.WeightStore = poly.NewWeightStore(HiddenSize)
	for i := 0; i < HiddenSize; i++ { lNorm.WeightStore.Master[i] = 1.0 }

	// Layer 2: MHA (Simulate sliding window + Receding memory)
	lMHA := net.GetLayer(0, 0, 0, 2)
	lMHA.Type = poly.LayerMultiHeadAttention
	lMHA.DModel = HiddenSize
	lMHA.NumHeads = 1
	lMHA.HeadDim = HiddenSize
	lMHA.MaxSeqLen = 512
	lMHA.WeightStore = poly.NewWeightStore(HiddenSize*HiddenSize*4 + HiddenSize*4)
	// Initialize MHA as a "Recent History Accumulator"
	// Q=Identity, K=Identity, V=Identity.
	// We'll rely on the default Attention Softmax to distribute weight.
	for i := 0; i < HiddenSize; i++ {
		lMHA.WeightStore.Master[i*HiddenSize+i] = 1.0 // Q
		lMHA.WeightStore.Master[HiddenSize*HiddenSize+i*HiddenSize+i] = 1.0 // K
		lMHA.WeightStore.Master[2*HiddenSize*HiddenSize+i*HiddenSize+i] = 1.0 // V
		lMHA.WeightStore.Master[3*HiddenSize*HiddenSize+i*HiddenSize+i] = 1.0 // O
	}

	// 3. Analyze Patterns (6-gram peaks)
	fmt.Println("[*] Analyzing 6-gram patterns for synaptic calibration...")
	patterns := analyzeNgrams(corpus, SnapLen+1, vActual)
	var prefixes []string
	for k := range patterns { prefixes = append(prefixes, k) }
	sort.Strings(prefixes)
	numClusters := len(prefixes)
	fmt.Printf("[*] Found %d unique patterns.\n", numClusters)

	// Layer 3: KMeans (Pattern Matcher)
	lKM := net.GetLayer(0, 0, 0, 3)
	lKM.Type = poly.LayerKMeans
	lKM.NumClusters = numClusters
	lKM.InputHeight = HiddenSize
	lKM.KMeansTemperature = 0.05
	lKM.WeightStore = poly.NewWeightStore(numClusters * HiddenSize)

	// Layer 4: Dense (Predictor)
	lDense := net.GetLayer(0, 0, 0, 4)
	lDense.Type = poly.LayerDense
	lDense.InputHeight = numClusters
	lDense.OutputHeight = vActual
	lDense.WeightStore = poly.NewWeightStore(numClusters * vActual)

	// Initialize Dense weights to a very low value (log(epsilon))
	for k := range lDense.WeightStore.Master { lDense.WeightStore.Master[k] = -100.0 }

	// Snap KMeans/Dense weights
	for i, pref := range prefixes {
		parts := strings.Split(pref, "|")
		for _, p := range parts {
			if p == "" { continue }
			var charIdx int
			fmt.Sscanf(p, "%d", &charIdx)
			lKM.WeightStore.Master[i*HiddenSize + charIdx] += 1.0
		}
		
		counts := patterns[pref]
		var total float64
		for _, c := range counts { total += float64(c) }
		for charIdx, count := range counts {
			if count > 0 {
				p := float64(count) / total
				lDense.WeightStore.Master[charIdx*numClusters+i] = float32(math.Log(p + 1e-10))
			}
		}
	}

	fmt.Println("[*] Snap complete. System ready.")
	fmt.Println("[*] Suggestion: 'it no'")

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("\nPrompt > ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if strings.ToLower(input) == "exit" { break }
		if len(input) < 5 { input = "     " + input }

		// Convert input to IDs
		inputIDs := make([]int, len(input))
		for i, c := range input { 
			idx, ok := charToIdx[byte(c)]
			if !ok { idx = 0 }
			inputIDs[i] = idx
		}

		// Reset KV Cache
		lMHA.KVOffset = 0
		lMHA.KVCacheK = nil
		lMHA.KVCacheV = nil

		fmt.Printf("Answer > ")
		start := time.Now()
		
		// 1. Prefill
		var currentInput *poly.Tensor[float32]
		for _, id := range inputIDs { currentInput = forwardToken(net, id) }

		// 2. Decode
		genCount := 0
		for i := 0; i < 200; i++ {
			probs := softmax(currentInput.Data, 1.0)
			nextID := sampleTopK(probs, 5, 1.0)
			fmt.Print(string(vocab[nextID]))
			genCount++
			currentInput = forwardToken(net, nextID)
			if vocab[nextID] == '\n' && genCount > 100 { break }
		}

		elapsed := time.Since(start)
		fmt.Printf("\n\n[Stats] Gen time: %v | Performance: %.2f chars/sec\n", 
			elapsed.Round(time.Millisecond), float64(genCount)/elapsed.Seconds())
	}
}

func forwardToken(net *poly.VolumetricNetwork, tokenID int) *poly.Tensor[float32] {
	lEmbed := net.GetLayer(0,0,0,0)
	hidden := poly.NewTensor[float32](1, lEmbed.EmbeddingDim)
	copy(hidden.Data, lEmbed.WeightStore.Master[tokenID*lEmbed.EmbeddingDim : (tokenID+1)*lEmbed.EmbeddingDim])

	_, hidden = poly.RMSNormForwardPolymorphic(net.GetLayer(0,0,0,1), hidden)
	
	// Layer 2: MHA with KV Cache
	_, hidden = poly.MHAForwardPolymorphic(net.GetLayer(0,0,0,2), hidden)
	
	// CRITICAL: L2 Normalize to match Snap clusters regardless of context length
	var sumSq float32
	for _, v := range hidden.Data { sumSq += v * v }
	mag := float32(math.Sqrt(float64(sumSq))) + 1e-10
	for i := range hidden.Data { hidden.Data[i] /= mag }

	// Layer 3: KMeans
	_, hidden = poly.KMeansForwardPolymorphic(net.GetLayer(0,0,0,3), hidden)
	
	// Layer 4: Dense
	_, hidden = poly.DenseForwardPolymorphic(net.GetLayer(0,0,0,4), hidden)
	return hidden
}

func analyzeNgrams(corpus []int, n int, vSize int) map[string][]int {
	patterns := make(map[string][]int)
	for i := n - 1; i < len(corpus); i++ {
		var prefixBuilder strings.Builder
		for j := 0; j < n-1; j++ { fmt.Fprintf(&prefixBuilder, "%03d|", corpus[i-(n-1)+j]) }
		prefix := prefixBuilder.String()
		if _, ok := patterns[prefix]; !ok { patterns[prefix] = make([]int, vSize) }
		patterns[prefix][corpus[i]]++
	}
	return patterns
}

func softmax(v []float32, temp float64) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v { if x > maxV { maxV = x } }
	sum := float32(0)
	for i, x := range v {
		out[i] = float32(math.Exp(float64(x-maxV) / temp))
		sum += out[i]
	}
	for i := range out { out[i] /= sum }
	return out
}

func sampleTopK(probs []float32, k int, temp float32) int {
	type kv struct { i int; p float32 }
	var top []kv
	for i, p := range probs { top = append(top, kv{i, p}) }
	sort.Slice(top, func(i, j int) bool { return top[i].p > top[j].p })
	if k > len(top) { k = len(top) }
	top = top[:k]
	var total float32
	for i := range top {
		top[i].p = float32(math.Pow(float64(top[i].p), 1.0/float64(temp)))
		total += top[i].p
	}
	r := float32(time.Now().UnixNano()%1000) / 1000.0 * total
	var cum float32
	for _, kv := range top {
		cum += kv.p
		if r <= cum { return kv.i }
	}
	return top[0].i
}
