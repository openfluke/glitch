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
	DataDirKSnap    = "data"
	CorpusFileKSnap = "data/shakespeare.txt"
	CorpusURLKSnap  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	EvalCharsKSnap  = 5000 // Faster eval for deep stacks
	GenLengthKSnap  = 150
)

type ArchKSnap struct {
	Name string
	Net  *poly.VolumetricNetwork
	Init func(*poly.VolumetricNetwork)
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║      KMEANS DEPTH BENCHMARK  ·  METACOGNITION ENABLED          ║")
	fmt.Println("║  Testing 1, 2, and 3-layered KMeans stacks for Snap LMs       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	if err := ensureCorpusKSnap(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	corpus, vocab, _ := loadCorpusKSnap()
	V := len(vocab)
	train := corpus[:len(corpus)*8/10]
	val := corpus[len(corpus)*8/10 : len(corpus)*9/10]
	if len(val) > EvalCharsKSnap { val = val[:EvalCharsKSnap] }

	fmt.Printf("[*] Corpus: %d chars | Vocab: %d\n\n", len(corpus), V)

	uni := computeUnigramKSnap(train, V)
	bi := computeBigramKSnap(train, V)
	tri := computeTrigramKSnap(train, V)

	archs := []ArchKSnap{
		{
			Name: "1-Layer Baseline (KMeans -> Dense)",
			Net:  buildKStackSnap(V, 1),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				layers := obs.SequentialLayers
				installContextKMeansKSnap(&layers[0], V, 2)
				installInterpolatedDenseKSnap(&layers[1], V, V*V, 2, uni, bi, tri)
			},
		},
		{
			Name: "2-Layer Stack (KMeans[F] -> KMeans[C] -> Dense)",
			Net:  buildKStackSnap(V, 2),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				layers := obs.SequentialLayers
				// L0: Map context to features
				installContextKMeansKSnap(&layers[0], V, 2)
				// L1: Cluster the features (Identity-like mapping for the demonstration)
				installFeatureKMeansKSnap(&layers[1], V*V, V*V)
				// L2: Dense prediction
				installInterpolatedDenseKSnap(&layers[2], V, V*V, 2, uni, bi, tri)
			},
		},
		{
			Name: "3-Layer Stack (KMeans[F] -> KMeans[F] -> KMeans[C] -> Dense)",
			Net:  buildKStackSnap(V, 3),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				layers := obs.SequentialLayers
				installContextKMeansKSnap(&layers[0], V, 2)
				installFeatureKMeansKSnap(&layers[1], V*V, V*V)
				installFeatureKMeansKSnap(&layers[2], V*V, V*V)
				installInterpolatedDenseKSnap(&layers[3], V, V*V, 2, uni, bi, tri)
			},
		},
	}

	results := make(map[string]float64)
	rng := rand.New(rand.NewSource(42))

	for i, arch := range archs {
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("  [EXPERIMENT %d]  %s\n", i+1, arch.Name)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		t0 := time.Now()
		arch.Init(arch.Net)
		snapTime := time.Since(t0)

		ppl := evalPerplexityKSnap(arch, val, V)
		results[arch.Name] = ppl

		fmt.Printf("  Snap time: %3.2fms  |  Perplexity: %.2f\n\n", float64(snapTime.Microseconds())/1000.0, ppl)

		fmt.Println("  Sample text:")
		ctxLen := 2
		seedChars := make([]int, ctxLen)
		for s := 0; s < ctxLen; s++ { seedChars[s] = train[100+s] }
		sample := generateTextKSnap(arch, seedChars, vocab, V, GenLengthKSnap, 1.0, rng)
		printWrappedKSnap(sample)
		fmt.Println()
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               KMEANS STACK DEPTH COMPARISON                      ║")
	fmt.Println("╠══════════════════════════════════════════════════╦══════════════╣")
	keys := make([]string, 0, len(results))
	for k := range results { keys = append(keys, k) }
	sort.Strings(keys)
	for _, k := range keys { fmt.Printf("║ %-48s ║  %8.2f    ║\n", k, results[k]) }
	fmt.Println("╚══════════════════════════════════════════════════╩══════════════╝\n")
}

// ── BUILD HELPERS ────────────────────────────────────────────────────────────

func buildKStackSnap(V, depth int) *poly.VolumetricNetwork {
	ctxLen := 2
	numClusters := V * V
	inputDim := ctxLen * V

	layersJSON := ""
	for i := 0; i < depth; i++ {
		mode := "clusters"
		outH := numClusters
		if i < depth-1 {
			mode = "features"
			outH = numClusters // Maintain dimension
		}
		inH := inputDim
		if i > 0 { inH = numClusters }
		
		layersJSON += fmt.Sprintf(`{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d, "output_mode": "%s"},`, inH, outH, numClusters, mode)
	}
	// Final Dense layer
	layersJSON += fmt.Sprintf(`{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}`, numClusters, V)

	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [%s]
		}]
	}`, inputDim, V, layersJSON)

	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildKStack: %v", err)) }

	// Recursively wrap internal layers with Metacognition
	wrapRecursivelySnap(net)
	return net
}

func wrapRecursivelySnap(net *poly.VolumetricNetwork) {
	// First wrap top-level layers
	poly.WrapWithMetacognition(net, nil)
	
	for i := range net.Layers {
		l := &net.Layers[i]
		if l.Type == poly.LayerMetacognition && l.MetaObservedLayer != nil {
			wrapLayerSnap(l.MetaObservedLayer)
		}
	}
}

func wrapLayerSnap(l *poly.VolumetricLayer) {
	if len(l.SequentialLayers) > 0 {
		for i := range l.SequentialLayers {
			sub := &l.SequentialLayers[i]
			if sub.Type == poly.LayerMetacognition { continue }
			
			observed := new(poly.VolumetricLayer)
			*observed = *sub
			
			sub.Type = poly.LayerMetacognition
			sub.MetaObservedLayer = observed
			sub.MetaRules = poly.DefaultStabilityRules()
			
			// Recurse
			wrapLayerSnap(observed)
		}
	}
	// Handle ParallelBranches too if needed
	if len(l.ParallelBranches) > 0 {
		for i := range l.ParallelBranches {
			wrapLayerSnap(&l.ParallelBranches[i])
		}
	}
}

// ── INSTALL HELPERS ──────────────────────────────────────────────────────────

func installContextKMeansKSnap(l *poly.VolumetricLayer, V, ctxLen int) {
	l.KMeansTemperature = 0.1
	inputDim := l.InputHeight
	numClusters := 1
	for i := 0; i < ctxLen; i++ { numClusters *= V }
	l.WeightStore = poly.NewWeightStore(numClusters * inputDim)
	for c := 0; c < numClusters; c++ {
		idx := c
		for i := ctxLen - 1; i >= 0; i-- {
			charIdx := idx % V
			offset := (inputDim - ctxLen*V) + i*V + charIdx
			l.WeightStore.Master[c*inputDim+offset] = 1.0
			idx /= V
		}
	}
}

func installFeatureKMeansKSnap(l *poly.VolumetricLayer, inDim, numClusters int) {
	l.KMeansTemperature = 0.5 // Softer features lookup
	l.WeightStore = poly.NewWeightStore(numClusters * inDim)
	// Identity mapping: each cluster j is "at" the j-th feature coordinate
	for j := 0; j < numClusters; j++ {
		if j < inDim {
			l.WeightStore.Master[j*inDim + j] = 1.0
		}
	}
}

func installInterpolatedDenseKSnap(l *poly.VolumetricLayer, V, numClusters, ctxLen int, uni []float32, bi [][]float32, tri map[int][]float32) {
	l.WeightStore = poly.NewWeightStore(numClusters * V)
	for c := 0; c < numClusters; c++ {
		idx := c
		context := make([]int, ctxLen)
		for i := ctxLen - 1; i >= 0; i-- {
			context[i] = idx % V
			idx /= V
		}
		for next := 0; next < V; next++ {
			p := 0.05 * float64(uni[next])
			if ctxLen >= 1 { p += 0.2 * float64(bi[context[ctxLen-1]][next]) }
			if ctxLen >= 2 {
				key := context[ctxLen-2]*V + context[ctxLen-1]
				if probs, ok := tri[key]; ok { p += 0.75 * float64(probs[next]) }
			}
			l.WeightStore.Master[next*numClusters+c] = float32(math.Log(p + 1e-10))
		}
	}
}

// ── STATS & EVAL ─────────────────────────────────────────────────────────────

func evalPerplexityKSnap(arch ArchKSnap, corpus []int, V int) float64 {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	if len(corpus) <= ctxLen { return 100 }
	totalLogProb, input := 0.0, poly.NewTensor[float32](1, inputDim)
	for i := ctxLen; i < len(corpus); i++ {
		for k := range input.Data { input.Data[k] = 0 }
		for j := 0; j < ctxLen; j++ { input.Data[j*V+corpus[i-ctxLen+j]] = 1.0 }
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTempKSnap(output.Data, 1.0)
		p := float64(probs[corpus[i]])
		if p < 1e-10 { p = 1e-10 }
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-ctxLen))
}

func generateTextKSnap(arch ArchKSnap, seedChars []int, vocab []byte, V, length int, temp float64, rng *rand.Rand) string {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	result := make([]byte, 0, length)
	for _, c := range seedChars { result = append(result, vocab[c]) }
	context := make([]int, ctxLen)
	copy(context, seedChars)
	input := poly.NewTensor[float32](1, inputDim)
	for i := 0; i < length; i++ {
		for k := range input.Data { input.Data[k] = 0 }
		for j := 0; j < ctxLen; j++ { input.Data[j*V+context[j]] = 1.0 }
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTempKSnap(output.Data, temp)
		next := sampleCategoricalKSnap(probs, rng)
		result = append(result, vocab[next])
		if ctxLen > 1 { copy(context, context[1:]); context[ctxLen-1] = next } else { context[0] = next }
	}
	return string(result)
}

func softmaxTempKSnap(v []float32, temp float64) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v { if x > maxV { maxV = x } }
	sum := float32(0)
	for i, x := range v { out[i] = float32(math.Exp(float64(x-maxV) / temp)); sum += out[i] }
	for i := range out { out[i] /= sum }
	return out
}

func sampleCategoricalKSnap(probs []float32, rng *rand.Rand) int {
	u := rng.Float32()
	cum := float32(0)
	for i, p := range probs { cum += p; if u < cum { return i } }
	return len(probs) - 1
}

func printWrappedKSnap(text string) {
	const lineWidth = 72
	runes := []rune(text)
	for i := 0; i < len(runes); i += lineWidth {
		end := i + lineWidth
		if end > len(runes) { end = len(runes) }
		fmt.Printf("  %s\n", string(runes[i:end]))
	}
}

func ensureCorpusKSnap() error {
	if _, err := os.Stat(CorpusFileKSnap); err == nil { return nil }
	os.MkdirAll(DataDirKSnap, 0755)
	resp, err := http.Get(CorpusURLKSnap)
	if err != nil { return err }
	defer resp.Body.Close()
	f, _ := os.Create(CorpusFileKSnap)
	defer f.Close()
	io.Copy(f, resp.Body)
	return nil
}

func loadCorpusKSnap() (corpus []int, vocab []byte, charToIdx map[byte]int) {
	data, _ := os.ReadFile(CorpusFileKSnap)
	seen := make(map[byte]bool)
	for _, b := range data { seen[b] = true }
	for b := range seen { vocab = append(vocab, b) }
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx = make(map[byte]int)
	for i, c := range vocab { charToIdx[c] = i }
	for _, b := range data { corpus = append(corpus, charToIdx[b]) }
	return
}

func computeUnigramKSnap(corpus []int, V int) []float32 {
	counts := make([]int, V)
	for _, c := range corpus { counts[c]++ }
	total := float64(len(corpus))
	probs := make([]float32, V)
	for i, c := range counts { probs[i] = float32(float64(c) / total) }
	return probs
}

func computeBigramKSnap(corpus []int, V int) [][]float32 {
	counts := make([][]int, V)
	for i := range counts { counts[i] = make([]int, V) }
	for i := 1; i < len(corpus); i++ { counts[corpus[i-1]][corpus[i]]++ }
	bigram := make([][]float32, V)
	for p := range bigram {
		bigram[p] = make([]float32, V)
		total := 0
		for _, c := range counts[p] { total += c }
		for n := range bigram[p] {
			if total > 0 { bigram[p][n] = float32(float64(counts[p][n]) / float64(total)) } else { bigram[p][n] = 1.0 / float32(V) }
		}
	}
	return bigram
}

func computeTrigramKSnap(corpus []int, V int) map[int][]float32 {
	counts := make(map[int][]int)
	for i := 2; i < len(corpus); i++ {
		key := corpus[i-2]*V + corpus[i-1]
		if counts[key] == nil { counts[key] = make([]int, V) }
		counts[key][corpus[i]]++
	}
	trigram := make(map[int][]float32)
	for key, cnts := range counts {
		trigram[key] = make([]float32, V)
		total := 0
		for _, c := range cnts { total += c }
		for n := range trigram[key] { trigram[key][n] = float32(float64(cnts[n]) / float64(total)) }
	}
	return trigram
}
