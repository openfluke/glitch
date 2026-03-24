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
	CtxLen     = 32  
	SnapLen    = 5   
	HiddenSize = 128 
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║            ULTIMATE TRANSFORMER SNAP  ·  META-CHAT           ║")
	fmt.Println("║  Full Llama-Style Block + Metacognition  ·  Poly Engine v2   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// 1. Load data
	data, _ := os.ReadFile(CorpusFile)
	seen := make(map[byte]bool)
	for _, b := range data { seen[b] = true }
	var vocab []byte
	for b := range seen { vocab = append(vocab, b) }
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx := make(map[byte]int)
	for i, c := range vocab { charToIdx[c] = i }
	corpus := make([]int, len(data))
	for i, b := range data { corpus[i] = charToIdx[b] }
	V := len(vocab)

	fmt.Printf("[*] Analyzing patterns (SnapLen=%d)...\n", SnapLen)
	patterns := analyzeNgrams(corpus, SnapLen+1, V)
	var prefixes []string
	for k := range patterns { prefixes = append(prefixes, k) }
	sort.Strings(prefixes)
	numClusters := len(prefixes)

	// 2. Build Transformer Network
	// Architecture: [Embedding, Metacognition(RMSNorm), Metacognition(MHA), Metacognition(KMeans), Metacognition(Dense)]
	net := poly.NewVolumetricNetwork(1, 1, 1, 5)
	
	// Embedding
	l0 := net.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerEmbedding
	l0.VocabSize = V
	l0.EmbeddingDim = HiddenSize
	l0.WeightStore = poly.NewWeightStore(V * HiddenSize)
	for i := 0; i < V; i++ { l0.WeightStore.Master[i*HiddenSize+i] = 1.0 }

	// Norm
	l1 := net.GetLayer(0, 0, 0, 1)
	l1.Type = poly.LayerRMSNorm
	l1.InputHeight = HiddenSize
	l1.WeightStore = poly.NewWeightStore(HiddenSize)
	for i := 0; i < HiddenSize; i++ { l1.WeightStore.Master[i] = 1.0 }

	// MHA (Recency focus)
	l2 := net.GetLayer(0, 0, 0, 2)
	l2.Type = poly.LayerMultiHeadAttention
	l2.DModel = HiddenSize
	l2.NumHeads = 1
	l2.HeadDim = HiddenSize
	l2.MaxSeqLen = 512
	l2.WeightStore = poly.NewWeightStore(HiddenSize*HiddenSize*4 + HiddenSize*4)
	for i := 0; i < HiddenSize; i++ {
		l2.WeightStore.Master[i*HiddenSize+i] = 1.0 // Q
		l2.WeightStore.Master[HiddenSize*HiddenSize+i*HiddenSize+i] = 1.0 // K
		l2.WeightStore.Master[2*HiddenSize*HiddenSize+i*HiddenSize+i] = 1.0 // V
		l2.WeightStore.Master[3*HiddenSize*HiddenSize+i*HiddenSize+i] = 1.0 // O
	}

	// KMeans
	l3 := net.GetLayer(0, 0, 0, 3)
	l3.Type = poly.LayerKMeans
	l3.NumClusters = numClusters
	l3.InputHeight = HiddenSize
	l3.KMeansTemperature = 0.05
	l3.WeightStore = poly.NewWeightStore(numClusters * HiddenSize)

	// Dense
	l4 := net.GetLayer(0, 0, 0, 4)
	l4.Type = poly.LayerDense
	l4.InputHeight = numClusters
	l4.OutputHeight = V
	l4.WeightStore = poly.NewWeightStore(numClusters * V)
	for k := range l4.WeightStore.Master { l4.WeightStore.Master[k] = -100.0 }

	// SNAPPING
	fmt.Printf("[*] Snapping %d clusters into Transformer synapses...\n", numClusters)
	for i, pref := range prefixes {
		parts := strings.Split(pref, "|")
		for _, p := range parts {
			if p == "" { continue }
			var charIdx int
			fmt.Sscanf(p, "%d", &charIdx)
			l3.WeightStore.Master[i*HiddenSize+charIdx] += 1.0
		}
		counts := patterns[pref]
		var total float64
		for _, c := range counts { total += float64(c) }
		for charIdx, count := range counts {
			if count > 0 {
				p := float64(count) / total
				l4.WeightStore.Master[charIdx*numClusters+i] = float32(math.Log(p + 1e-10))
			}
		}
	}

	// WRAP WITH METACGONITION
	fmt.Println("[*] Engaging Metacognitive Heuristics...")
	poly.WrapWithMetacognition(net, []poly.MetaRule{
		{Condition: poly.MetaCondGainDrift, Threshold: 0.1, Command: 101, SelfOnly: true},
		{Condition: poly.MetaCondStdBelow, Threshold: 0.01, Command: 98, Param: float32(numClusters), SelfOnly: true}, // If KMeans dies, reboot it
	})

	fmt.Println("[*] System active. Listening to context...")
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("\nPrompt > ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if strings.ToLower(input) == "exit" { break }
		if len(input) < 5 { input = "     " + input }

		inputIDs := make([]int, len(input))
		for i, c := range input { 
			idx, ok := charToIdx[byte(c)]
			if !ok { idx = 0 }
			inputIDs[i] = idx
		}

		// Reset Context
		l2.KVOffset = 0
		l2.KVCacheK = nil
		l2.KVCacheV = nil

		fmt.Printf("Answer > ")
		start := time.Now()
		
		// 1. Prefill
		var currentInput *poly.Tensor[float32]
		for _, id := range inputIDs { 
			currentInput = poly.NewTensor[float32](1, HiddenSize)
			copy(currentInput.Data, l0.WeightStore.Master[id*HiddenSize : (id+1)*HiddenSize])
			
			// Layer 1: Norm
			_, currentInput = poly.DispatchLayer(l1, currentInput, nil)
			// Layer 2: MHA
			_, currentInput = poly.DispatchLayer(l2, currentInput, nil)
			
			// L2 Normalization (to match 5-char snap clusters)
			var sumSq float32
			for _, v := range currentInput.Data { sumSq += v * v }
			mag := float32(math.Sqrt(float64(sumSq))) + 1e-10
			for j := range currentInput.Data { currentInput.Data[j] /= mag }
			
			// Layer 3: KMeans
			_, currentInput = poly.DispatchLayer(l3, currentInput, nil)
			// Layer 4: Dense
			_, currentInput = poly.DispatchLayer(l4, currentInput, nil)
		}

		// 2. Decode
		genCount := 0
		for i := 0; i < 200; i++ {
			probs := softmax(currentInput.Data, 1.0)
			nextID := sampleTopK(probs, 5, 1.0)
			fmt.Print(string(vocab[nextID]))
			genCount++
			
			// Next step
			currentInput = poly.NewTensor[float32](1, HiddenSize)
			copy(currentInput.Data, l0.WeightStore.Master[nextID*HiddenSize : (nextID+1)*HiddenSize])
			
			_, currentInput = poly.DispatchLayer(l1, currentInput, nil)
			_, currentInput = poly.DispatchLayer(l2, currentInput, nil)
			
			var sumSq float32
			for _, v := range currentInput.Data { sumSq += v * v }
			mag := float32(math.Sqrt(float64(sumSq))) + 1e-10
			for j := range currentInput.Data { currentInput.Data[j] /= mag }
			
			_, currentInput = poly.DispatchLayer(l3, currentInput, nil)
			_, currentInput = poly.DispatchLayer(l4, currentInput, nil)
			
			if genCount%50 == 0 {
				fmt.Printf(" [META: std=%.3f avg=%.3f]", ComputeStats(currentInput.Data).Std, ComputeStats(currentInput.Data).Avg)
			}
			if vocab[nextID] == '\n' && genCount > 100 { break }
		}

		elapsed := time.Since(start)
		fmt.Printf("\n\n[Stats] Gen time: %v | Performance: %.2f chars/sec\n", 
			elapsed.Round(time.Millisecond), float64(genCount)/elapsed.Seconds())
	}
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

type Stat struct{ Avg, Std float32 }
func ComputeStats(v []float32) Stat {
	var sum, sqSum float32
	for _, x := range v { sum += x; sqSum += x * x }
	avg := sum / float32(len(v))
	return Stat{Avg: avg, Std: float32(math.Sqrt(float64(sqSum/float32(len(v)) - avg*avg)))}
}
