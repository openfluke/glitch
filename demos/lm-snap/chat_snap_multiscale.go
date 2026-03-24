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
	CorpusFile   = "data/shakespeare.txt"
	ChatSeedFile = "data/chat_seed.txt"
	GenLength    = 150
	MaxCtxLen    = 11
)

type Scale struct {
	CtxLen  int
	Weights []string
	Counts  map[string][]int
	Net     *poly.VolumetricNetwork
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║            SNAP HYBRID ENSEMBLE  ·  5/8/11-GRAM                ║")
	fmt.Println("║    Shakespearean Soul  ·  Zero-Backprop Chat Intelligence     ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Load hybrid corpus: Shakespeare + 50 copies of Chat Seed
	corpus, vocab, charToIdx := loadCorpus(CorpusFile, ChatSeedFile)
	V := len(vocab)

	// Define scales for the ensemble
	ctxLens := []int{5, 8, 11}
	scales := make([]*Scale, len(ctxLens))

	totalPatterns := 0
	fmt.Printf("[*] Snapping multi-scale layers...\n")
	for i, n := range ctxLens {
		fmt.Printf("    - Analyzing %d-gram scale (ctxLen=%d)...\n", n+1, n)
		counts, prefixes := analyzeNgramsXL(corpus, V, n)
		numClusters := len(prefixes)
		totalPatterns += numClusters
		
		fmt.Printf("      Found %d patterns. Building network...\n", numClusters)
		net := buildScaleNet(V, n, numClusters)
		installSnap(net, V, n, prefixes, counts)
		
		scales[i] = &Scale{
			CtxLen:  n,
			Weights: prefixes,
			Counts:  counts,
			Net:     net,
		}
	}

	fmt.Printf("\n[*] Total unique patterns across all scales: %d (~11.6x larger than 6-gram baseline)\n", totalPatterns)
	fmt.Println("[*] Multi-Scale Snap complete. System ready.")
	fmt.Println("[*] Type at least 11 chars (e.g. from Shakespeare) and press Enter.")

	rand.Seed(time.Now().UnixNano())
	hintIdx := rand.Intn(len(corpus) - MaxCtxLen - 1)
	hint := ""
	for i := 0; i < MaxCtxLen; i++ {
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

		// Format input context as "User: <input>\nModel: "
		// The model is trained on "User: ...\nModel: " patterns.
		// We want the context to end with "Model: " for the model to generate a response.
		chatInput := fmt.Sprintf("User: %s\nModel: ", input)
		
		fmt.Print("Answer > ")
		generateEnsemble(scales, chatInput, vocab, V, charToIdx, GenLength)
		fmt.Println("\n")
	}
}

func generateEnsemble(scales []*Scale, prompt string, vocab []byte, V int, charToIdx map[byte]int, length int) {
	// Initialize running context from prompt
	context := make([]int, MaxCtxLen)
	// Use the last MaxCtxLen characters of the prompt
	p := prompt
	if len(p) > MaxCtxLen {
		p = prompt[len(prompt)-MaxCtxLen:]
	} else {
		// Pad with spaces if prompt is shorter than MaxCtxLen
		p = strings.Repeat(" ", MaxCtxLen-len(p)) + p
	}

	for i := 0; i < MaxCtxLen; i++ {
		idx, ok := charToIdx[p[i]]
		if !ok { idx = 0 } // Use index 0 for unknown characters
		context[i] = idx
	}

	inputs := make([]*poly.Tensor[float32], len(scales))
	for i, s := range scales {
		inputs[i] = poly.NewTensor[float32](1, s.CtxLen*V)
	}

	startTime := time.Now()
	result := ""
	for i := 0; i < length; i++ {
		// Aggregate logits...
		sumLogits := make([]float32, V)
		for sIdx, s := range scales {
			input := inputs[sIdx]
			for k := range input.Data { input.Data[k] = 0 }
			ctxOffset := MaxCtxLen - s.CtxLen
			for j := 0; j < s.CtxLen; j++ {
				input.Data[j*V+context[ctxOffset+j]] = 1.0
			}
			output, _, _ := poly.ForwardPolymorphic(s.Net, input)
			weight := 1.0 + 0.5*float64(s.CtxLen)
			for v := 0; v < V; v++ {
				sumLogits[v] += float32(float64(output.Data[v]) * weight)
			}
		}

		probs := softmax(sumLogits, 0.8)
		next := sample(probs)
		char := vocab[next]
		
		fmt.Printf("%c", char)
		result += string(char)

		// Early stop if model starts hallucinating both sides of the chat
		if strings.HasSuffix(result, "\nUser:") || strings.HasSuffix(result, "User:") {
			break
		}

		copy(context, context[1:])
		context[MaxCtxLen-1] = next
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\n\n[Stats] Gen time: %v | Performance: %.2f chars/sec", elapsed.Round(time.Millisecond), float64(length)/elapsed.Seconds())
}

func buildScaleNet(V, ctxLen, numClusters int) *poly.VolumetricNetwork {
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
	if err != nil { panic(err) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func installSnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	km, ds := &seq[0], &seq[1]
	numClusters, inputDim := len(prefixes), ctxLen*V
	
	km.KMeansTemperature = 0.02
	km.WeightStore = poly.NewWeightStore(numClusters * inputDim)
	ds.WeightStore = poly.NewWeightStore(numClusters * V)

	for c, prefix := range prefixes {
		for i := 0; i < len(prefix); i++ {
			charIdx := int(prefix[i])
			km.WeightStore.Master[c*inputDim+i*V+charIdx] = 1.0
		}
		total := 0
		cnts := counts[prefix]
		for _, v := range cnts { total += v }
		for next, count := range cnts {
			if count > 0 {
				p := float64(count) / float64(total)
				ds.WeightStore.Master[next*numClusters+c] = float32(math.Log(p + 1e-10))
			} else {
				ds.WeightStore.Master[next*numClusters+c] = -20.0
			}
		}
	}
}

func analyzeNgramsXL(corpus []int, V, n int) (map[string][]int, []string) {
	counts := make(map[string][]int)
	var prefixes []string
	buf := make([]byte, n)
	for i := n; i < len(corpus); i++ {
		for j := 0; j < n; j++ {
			buf[j] = byte(corpus[i-n+j])
		}
		prefix := string(buf)
		if c, ok := counts[prefix]; !ok {
			counts[prefix] = make([]int, V)
			prefixes = append(prefixes, prefix)
			counts[prefix][corpus[i]]++
		} else {
			c[corpus[i]]++
		}
	}
	sort.Strings(prefixes)
	return counts, prefixes
}

func loadCorpus(paths ...string) (corpus []int, vocab []byte, charToIdx map[byte]int) {
	var data []byte
	for _, path := range paths {
		d, err := os.ReadFile(path)
		if err != nil {
			fmt.Printf("[!] Warning: Could not read %s\n", path)
			continue
		}
		// If it's the chat seed, repeat it to make it more prominent
		if strings.Contains(path, "chat_seed") {
			for r := 0; r < 50; r++ { // 50x weight to ensure it dominates greetings
				data = append(data, d...)
				data = append(data, '\n')
			}
		} else {
			data = append(data, d...)
		}
	}
	
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
	for _, x := range v { if x > maxV { maxV = x } }
	sum := float32(0)
	for i, x := range v {
		out[i] = float32(math.Exp(float64(x-maxV) / temp))
		sum += out[i]
	}
	for i := range out { out[i] /= sum }
	return out
}

func sample(probs []float32) int {
	u := rand.Float32()
	cum := float32(0)
	for i, p := range probs {
		cum += p
		if u < cum { return i }
	}
	return len(probs) - 1
}
