package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/openfluke/loom/poly"
)

const (
	CorpusFile    = "data/tinystories.txt"
	MaxVocab      = 10000
	EmbedDim      = 128
	GenLength     = 100
	DefaultCtxLen = 3
)

type WordScale struct {
	CtxLen  int
	Weights []string
	// Sparse Dense: ClusterID -> WordIdx -> Count
	Transitions []map[int]int
	Net         *poly.VolumetricNetwork
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          SNAP HYBRID WORD-LEVEL XL  ·  SPARSE OPTIMIZED        ║")
	fmt.Println("║    10k Vocab  ·  Random Projection  ·  1.6M Sparse Clusters    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Printf("[*] Loading and Tokenizing TinyStories (Word-Level)...\n")
	corpus, vocab, wordToIdx := loadWordCorpus(CorpusFile, MaxVocab)
	V := len(vocab)
	fmt.Printf("[*] Vocabulary Size: %d unique words.\n", V)

	// Build Random Projection Matrix (Frozen Embeddings)
	proj := make([][]float32, V)
	rand.Seed(42) // Deterministic randoms for consistent embeddings
	for i := range proj {
		proj[i] = make([]float32, EmbedDim)
		for j := range proj[i] {
			proj[i][j] = float32(rand.NormFloat64() * 0.1)
		}
	}

	ctxLens := []int{1, 2, 3}
	scales := make([]*WordScale, len(ctxLens))
	totalPatterns := 0

	for i, n := range ctxLens {
		fmt.Printf("[*] Snapping %d-word scale (ctxLen=%d)...\n", n+1, n)
		counts, prefixes := analyzeWordGrams(corpus, V, n)
		numClusters := len(prefixes)
		totalPatterns += numClusters

		fmt.Printf("    Found %d patterns. Building Sparse-Efficient network...\n", numClusters)
		// We ONLY use KMeans in the Poly network to save memory (Dense layer was 25GB)
		net := buildKMeansOnlyNet(n, numClusters)
		transitions := installSparseWeightSnap(net, n, prefixes, counts, proj, wordToIdx)

		scales[i] = &WordScale{
			CtxLen:      n,
			Weights:     prefixes,
			Transitions: transitions,
			Net:         net,
		}
	}

	fmt.Printf("\n[*] Total unique word patterns: %d\n", totalPatterns)
	fmt.Println("[*] Word-Level Snap complete (Memory: ~2.4 GB). System ready.")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nPrompt > ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if strings.ToLower(input) == "exit" {
			break
		}

		fmt.Print("Answer > ")
		generateSparseWordEnsemble(scales, input, vocab, wordToIdx, proj, GenLength)
		fmt.Println()
	}
}

func generateSparseWordEnsemble(scales []*WordScale, prompt string, vocab []string, wordToIdx map[string]int, proj [][]float32, length int) {
	words := tokenize(prompt)
	V := len(vocab)

	window := make([]int, DefaultCtxLen)
	for i := 0; i < DefaultCtxLen; i++ {
		// Fill window from the end of the words slice
		wordPos := len(words) - DefaultCtxLen + i
		if wordPos >= 0 && wordPos < len(words) {
			w := words[wordPos]
			if val, ok := wordToIdx[w]; ok {
				window[i] = val
			} else {
				window[i] = 0 // <UNK>
			}
		} else {
			window[i] = 0 // Padding / <UNK>
		}
	}

	for i := 0; i < length; i++ {
		sumProbs := make([]float32, V)

		for _, s := range scales {
			input := poly.NewTensor[float32](1, s.CtxLen*EmbedDim)
			ctxOffset := DefaultCtxLen - s.CtxLen
			for j := 0; j < s.CtxLen; j++ {
				wordIdx := window[ctxOffset+j]
				copy(input.Data[j*EmbedDim:], proj[wordIdx])
			}

			// Forward pass only runs KMeans. Output is cluster activations.
			output, _, _ := poly.ForwardPolymorphic(s.Net, input)

			// Find the best matching cluster (Top-1)
			bestCluster := 0
			maxAct := output.Data[0]
			for c, act := range output.Data {
				if act > maxAct {
					maxAct = act
					bestCluster = c
				}
			}

			// Aggregate word probabilities from the best cluster
			trans := s.Transitions[bestCluster]
			total := 0
			for _, count := range trans {
				total += count
			}

			weight := 1.0 + 0.2*float64(s.CtxLen) // Bias towards longer context
			for nextIdx, count := range trans {
				p := float32(float64(count) / float64(total))
				sumProbs[nextIdx] += p * float32(weight)
			}
		}

		next := sample(sumProbs)
		fmt.Printf("%s ", vocab[next])

		copy(window, window[1:])
		window[DefaultCtxLen-1] = next
	}
}

func buildKMeansOnlyNet(ctxLen, numClusters int) *poly.VolumetricNetwork {
	inputDim := ctxLen * EmbedDim
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d, "temperature": 0.01
		}]
	}`, inputDim, numClusters, numClusters)
	net, _ := poly.BuildNetworkFromJSON([]byte(json))
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func installSparseWeightSnap(net *poly.VolumetricNetwork, ctxLen int, prefixes []string, counts map[string]map[int]int, proj [][]float32, wordToIdx map[string]int) []map[int]int {
	km := net.Layers[0].MetaObservedLayer
	numClusters, inputDim := len(prefixes), ctxLen*EmbedDim

	km.WeightStore = poly.NewWeightStore(numClusters * inputDim)
	transitions := make([]map[int]int, numClusters)

	for c, prefix := range prefixes {
		pIdxStrings := strings.Split(prefix, " ")
		for i, idxStr := range pIdxStrings {
			if i >= ctxLen {
				break
			}
			var wordIdx int
			fmt.Sscanf(idxStr, "%d", &wordIdx)
			copy(km.WeightStore.Master[c*inputDim+i*EmbedDim:], proj[wordIdx])
		}
		transitions[c] = counts[prefix]
	}
	return transitions
}

func analyzeWordGrams(corpus []int, V, n int) (map[string]map[int]int, []string) {
	counts := make(map[string]map[int]int)
	var prefixes []string
	for i := n; i < len(corpus); i++ {
		var sb strings.Builder
		for j := 0; j < n; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(fmt.Sprintf("%d", corpus[i-n+j]))
		}
		prefix := sb.String()
		if _, ok := counts[prefix]; !ok {
			counts[prefix] = make(map[int]int)
			prefixes = append(prefixes, prefix)
		}
		counts[prefix][corpus[i]]++
	}
	return counts, prefixes
}

func loadWordCorpus(path string, maxVocab int) ([]int, []string, map[string]int) {
	data, _ := os.ReadFile(path)
	rawWords := tokenize(string(data))
	freqs := make(map[string]int)
	for _, w := range rawWords {
		freqs[w]++
	}
	type wordFreq struct {
		w string
		f int
	}
	var sorted []wordFreq
	for w, f := range freqs {
		sorted = append(sorted, wordFreq{w, f})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].f > sorted[j].f })
	if len(sorted) > maxVocab {
		sorted = sorted[:maxVocab]
	}
	vocab := make([]string, len(sorted))
	wordToIdx := make(map[string]int)
	for i, wf := range sorted {
		vocab[i] = wf.w
		wordToIdx[wf.w] = i
	}
	corpus := make([]int, 0, len(rawWords))
	for _, w := range rawWords {
		if idx, ok := wordToIdx[w]; ok {
			corpus = append(corpus, idx)
		} else {
			corpus = append(corpus, 0)
		}
	}
	return corpus, vocab, wordToIdx
}

func tokenize(text string) []string {
	re := regexp.MustCompile(`(\w+|[^\w\s])`)
	return re.FindAllString(strings.ToLower(text), -1)
}

func sample(probs []float32) int {
	u := rand.Float32()
	cum := float32(0)
	sum := float32(0)
	for _, p := range probs {
		sum += p
	}
	for i, p := range probs {
		cum += p / sum
		if u < cum {
			return i
		}
	}
	return len(probs) - 1
}
