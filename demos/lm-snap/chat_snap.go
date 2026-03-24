package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	CorpusFile = "data/shakespeare.txt"
	GenLength  = 150
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               SNAP INTERACTIVE CHAT  ·  6-GRAM                 ║")
	fmt.Println("║  Powered by the Poly Engine  ·  Zero-Backprop Architecture   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Printf("[*] Loading and Snapping 6-gram model (ctxLen=5)...\n")
	corpus, vocab, charToIdx := loadCorpus()
	V := len(vocab)
	ctxLen := 5

	fmt.Printf("[*] Analyzing unique patterns...\n")
	counts, prefixes := analyzeNgrams(corpus, V, ctxLen)
	numClusters := len(prefixes)
	fmt.Printf("[*] Found %d unique 6-gram patterns.\n", numClusters)

	net := buildNet(V, ctxLen, numClusters)
	installSnap(net, V, ctxLen, prefixes, counts)

	fmt.Println("[*] Snap complete. System ready.")
	fmt.Println("[*] Type at least 5 chars (e.g. from Shakespeare) and press Enter.")

	rand.Seed(time.Now().UnixNano())
	hintIdx := rand.Intn(len(corpus) - ctxLen - 1)
	hint := ""
	for i := 0; i < ctxLen; i++ {
		hint += string(vocab[corpus[hintIdx+i]])
	}
	fmt.Printf("[*] Suggestion: '%s'\n", hint)
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("Prompt > ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if strings.ToLower(input) == "exit" {
			break
		}

		if len(input) < ctxLen {
			input = strings.Repeat(" ", ctxLen-len(input)) + input
		}
		prompt := input[len(input)-ctxLen:]
		seed := make([]int, ctxLen)
		for i := 0; i < ctxLen; i++ {
			c := prompt[i]
			idx, ok := charToIdx[c]
			if !ok {
				idx = 0
			}
			seed[i] = idx
		}

		fmt.Print("Answer > ")
		generate(net, seed, vocab, V, ctxLen, GenLength)
		fmt.Println("\n")
	}
}

func generate(net *poly.VolumetricNetwork, seed []int, vocab []byte, V, ctxLen, length int) {
	context := make([]int, ctxLen)
	copy(context, seed)
	input := poly.NewTensor[float32](1, ctxLen*V)
	startTime := time.Now()

	for i := 0; i < length; i++ {
		for k := range input.Data {
			input.Data[k] = 0
		}
		for j := 0; j < ctxLen; j++ {
			input.Data[j*V+context[j]] = 1.0
		}

		output, _, _ := poly.ForwardPolymorphic(net, input)
		probs := softmax(output.Data, 1.0) // 1.0 as in test_compare_deep.go
		next := sample(probs)

		fmt.Printf("%c", vocab[next])
		copy(context, context[1:])
		context[ctxLen-1] = next
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\n\n[Stats] Gen time: %v | Performance: %.2f chars/sec", elapsed.Round(time.Millisecond), float64(length)/elapsed.Seconds())
}

func buildNet(V, ctxLen, numClusters int) *poly.VolumetricNetwork {
	inputDim := ctxLen * V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential", "input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d, "temperature": 0.05},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, inputDim, numClusters, numClusters, numClusters, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil {
		panic(err)
	}

	// Exact same wrap as in test_compare_deep.go
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func installSnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	km, ds := &seq[0], &seq[1]
	numClusters, inputDim := len(prefixes), ctxLen*V
	km.KMeansTemperature = 0.05 // CRITICAL: Ensure sharp matching as in benchmark
	km.WeightStore = poly.NewWeightStore(numClusters * inputDim)
	ds.WeightStore = poly.NewWeightStore(numClusters * V)

	for c, prefix := range prefixes {
		chars := parsePrefix(prefix)
		for i, charIdx := range chars {
			km.WeightStore.Master[c*inputDim+i*V+charIdx] = 1.0
		}
		total, cnts := 0, counts[prefix]
		for _, v := range cnts {
			total += v
		}
		for next, count := range cnts {
			p := float64(count) / float64(total)
			ds.WeightStore.Master[next*numClusters+c] = float32(math.Log(p + 1e-10))
		}
	}
}

func analyzeNgrams(corpus []int, V, n int) (map[string][]int, []string) {
	counts := make(map[string][]int)
	var prefixes []string
	for i := n; i < len(corpus); i++ {
		prefix := ""
		for j := 0; j < n; j++ {
			prefix += fmt.Sprintf("%03d|", corpus[i-n+j])
		}
		if _, ok := counts[prefix]; !ok {
			counts[prefix] = make([]int, V)
			prefixes = append(prefixes, prefix)
		}
		counts[prefix][corpus[i]]++
	}
	sort.Strings(prefixes)
	return counts, prefixes
}

func parsePrefix(prefix string) []int {
	var res []int
	for i := 0; i < len(prefix); i += 4 {
		var n int
		fmt.Sscanf(prefix[i:i+3], "%d", &n)
		res = append(res, n)
	}
	return res
}

func loadCorpus() (corpus []int, vocab []byte, charToIdx map[byte]int) {
	data, _ := os.ReadFile(CorpusFile)
	seen := make(map[byte]bool)
	for _, b := range data {
		seen[b] = true
	}
	for b := range seen {
		vocab = append(vocab, b)
	}
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx = make(map[byte]int)
	for i, c := range vocab {
		charToIdx[c] = i
	}
	for _, b := range data {
		corpus = append(corpus, charToIdx[b])
	}
	return
}

func softmax(v []float32, temp float64) []float32 {
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

func sample(probs []float32) int {
	u := rand.Float32()
	cum := float32(0)
	for i, p := range probs {
		cum += p
		if u < cum {
			return i
		}
	}
	return len(probs) - 1
}
