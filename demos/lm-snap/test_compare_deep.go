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
	DataDirDSnap    = "data"
	CorpusFileDSnap = "data/shakespeare.txt"
	CorpusURLDSnap  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	EvalCharsDSnap  = 5000
	GenLengthDSnap  = 200
)

type ArchDSnap struct {
	Name string
	Net  *poly.VolumetricNetwork
	Init func(*poly.VolumetricNetwork)
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║      DEEP N-GRAM BENCHMARK  ·  CONTEXT SCALING                 ║")
	fmt.Println("║  Testing 3, 4, 5, and 6-gram performance with Sparse Snapping  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	if err := ensureCorpusDSnap(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	corpus, vocab, _ := loadCorpusDSnap()
	V := len(vocab)
	train := corpus[:len(corpus)*8/10]
	val := corpus[len(corpus)*8/10 : len(corpus)*9/10]
	if len(val) > EvalCharsDSnap { val = val[:EvalCharsDSnap] }

	fmt.Printf("[*] Corpus: %d chars | Vocab: %d\n\n", len(corpus), V)

	rng := rand.New(rand.NewSource(42))
	results := make(map[string]float64)

	// Test different context lengths
	for ctxLen := 2; ctxLen <= 5; ctxLen++ {
		genName := fmt.Sprintf("%d-gram (ctxLen=%d)", ctxLen+1, ctxLen)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("  [EXPERIMENT]  %s\n", genName)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		
		fmt.Printf("  Analyzing corpus for %d-grams...\n", ctxLen+1)
		counts, prefixes := analyzeNgramsDSnap(train, V, ctxLen)
		numClusters := len(prefixes)
		fmt.Printf("  Found %d unique patterns.\n", numClusters)

		net := buildSparseNgramNetDSnap(V, ctxLen, numClusters)
		
		t0 := time.Now()
		installSparseNgramSnap(net, V, ctxLen, prefixes, counts)
		snapTime := time.Since(t0)

		arch := ArchDSnap{Name: genName, Net: net}
		ppl := evalPerplexityDSnap(arch, val, V)
		results[genName] = ppl

		fmt.Printf("  Snap time: %3.2fms  |  Perplexity: %.2f\n\n", float64(snapTime.Microseconds())/1000.0, ppl)

		fmt.Println("  Sample text:")
		seedChars := make([]int, ctxLen)
		for s := 0; s < ctxLen; s++ { seedChars[s] = train[100+s] }
		sample := generateTextDSnap(arch, seedChars, vocab, V, GenLengthDSnap, 1.0, rng)
		printWrappedDSnap(sample)
		fmt.Println()
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               DEEP N-GRAM CONTEXT COMPARISON                     ║")
	fmt.Println("╠══════════════════════════════════════════════════╦══════════════╣")
	fmt.Println("║ Generation                                       ║  Perplexity  ║")
	fmt.Println("╠══════════════════════════════════════════════════╬══════════════╣")
	keys := make([]string, 0, len(results))
	for k := range results { keys = append(keys, k) }
	sort.Strings(keys)
	for _, k := range keys { fmt.Printf("║ %-48s ║  %8.2f    ║\n", k, results[k]) }
	fmt.Println("╚══════════════════════════════════════════════════╩══════════════╝\n")
}

// ── ANALYSIS HELPERS ─────────────────────────────────────────────────────────

func analyzeNgramsDSnap(corpus []int, V, ctxLen int) (counts map[string][]int, prefixes []string) {
	counts = make(map[string][]int)
	for i := ctxLen; i < len(corpus); i++ {
		prefix := ""
		for j := 0; j < ctxLen; j++ {
			prefix += fmt.Sprintf("%03d|", corpus[i-ctxLen+j])
		}
		if _, ok := counts[prefix]; !ok {
			counts[prefix] = make([]int, V)
			prefixes = append(prefixes, prefix)
		}
		counts[prefix][corpus[i]]++
	}
	sort.Strings(prefixes)
	return
}

func parsePrefixDSnap(prefix string) []int {
	var res []int
	for i := 0; i < len(prefix); i += 4 {
		var n int
		fmt.Sscanf(prefix[i:i+3], "%d", &n)
		res = append(res, n)
	}
	return res
}

// ── BUILD HELPERS ────────────────────────────────────────────────────────────

func buildSparseNgramNetDSnap(V, ctxLen, numClusters int) *poly.VolumetricNetwork {
	inputDim := ctxLen * V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, inputDim, numClusters, numClusters, numClusters, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(err) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func installSparseNgramSnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	km := &seq[0]
	ds := &seq[1]

	numClusters := len(prefixes)
	inputDim := ctxLen * V

	km.KMeansTemperature = 0.05 // Very sharp
	km.WeightStore = poly.NewWeightStore(numClusters * inputDim)
	ds.WeightStore = poly.NewWeightStore(numClusters * V)

	for c, prefix := range prefixes {
		chars := parsePrefixDSnap(prefix)
		// KMeans: Map each char in context to cluster
		for i, charIdx := range chars {
			offset := i*V + charIdx
			km.WeightStore.Master[c*inputDim + offset] = 1.0
		}

		// Dense: Log-probabilities
		total := 0
		cnts := counts[prefix]
		for _, v := range cnts { total += v }
		for next, count := range cnts {
			p := float64(count) / float64(total)
			ds.WeightStore.Master[next*numClusters + c] = float32(math.Log(p + 1e-10))
		}
	}
}

// ── STATS & EVAL ─────────────────────────────────────────────────────────────

func evalPerplexityDSnap(arch ArchDSnap, corpus []int, V int) float64 {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	if len(corpus) <= ctxLen { return 100 }
	totalLogProb, input := 0.0, poly.NewTensor[float32](1, inputDim)
	for i := ctxLen; i < len(corpus); i++ {
		for k := range input.Data { input.Data[k] = 0 }
		for j := 0; j < ctxLen; j++ { input.Data[j*V+corpus[i-ctxLen+j]] = 1.0 }
		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmaxTempDSnap(output.Data, 1.0)
		p := float64(probs[corpus[i]])
		if p < 1e-10 { p = 1e-10 }
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-ctxLen))
}

func generateTextDSnap(arch ArchDSnap, seedChars []int, vocab []byte, V, length int, temp float64, rng *rand.Rand) string {
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
		probs := softmaxTempDSnap(output.Data, temp)
		next := sampleCategoricalDSnap(probs, rng)
		result = append(result, vocab[next])
		if ctxLen > 1 { copy(context, context[1:]); context[ctxLen-1] = next } else { context[0] = next }
	}
	return string(result)
}

func softmaxTempDSnap(v []float32, temp float64) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v { if x > maxV { maxV = x } }
	sum := float32(0)
	for i, x := range v { out[i] = float32(math.Exp(float64(x-maxV) / temp)); sum += out[i] }
	for i := range out { out[i] /= sum }
	return out
}

func sampleCategoricalDSnap(probs []float32, rng *rand.Rand) int {
	u := rng.Float32()
	cum := float32(0)
	for i, p := range probs { cum += p; if u < cum { return i } }
	return len(probs) - 1
}

func printWrappedDSnap(text string) {
	const lineWidth = 72
	runes := []rune(text)
	for i := 0; i < len(runes); i += lineWidth {
		end := i + lineWidth
		if end > len(runes) { end = len(runes) }
		fmt.Printf("  %s\n", string(runes[i:end]))
	}
}

func ensureCorpusDSnap() error {
	if _, err := os.Stat(CorpusFileDSnap); err == nil { return nil }
	os.MkdirAll(DataDirDSnap, 0755)
	resp, err := http.Get(CorpusURLDSnap)
	if err != nil { return err }
	defer resp.Body.Close()
	f, _ := os.Create(CorpusFileDSnap)
	defer f.Close()
	io.Copy(f, resp.Body)
	return nil
}

func loadCorpusDSnap() (corpus []int, vocab []byte, charToIdx map[byte]int) {
	data, _ := os.ReadFile(CorpusFileDSnap)
	seen := make(map[byte]bool)
	for _, b := range data { seen[b] = true }
	for b := range seen { vocab = append(vocab, b) }
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx = make(map[byte]int)
	for i, c := range vocab { charToIdx[c] = i }
	for _, b := range data { corpus = append(corpus, charToIdx[b]) }
	return
}
