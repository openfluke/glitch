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
	DataDirSnap    = "data"
	CorpusFileSnap = "data/shakespeare.txt"
	CorpusURLSnap  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	EvalCharsSnap  = 10000
	GenLengthSnap  = 200
)

type ArchSnap struct {
	Name string
	Net  *poly.VolumetricNetwork
	Init func(*poly.VolumetricNetwork)
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║      CHAR-LM SNAP COMPARE  ·  ARCHITECTURE SHOWDOWN            ║")
	fmt.Println("║  Testing different layer configurations for zero-backprop LMs  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	if err := ensureCorpusSnap(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	corpus, vocab, _ := loadCorpusSnap()
	V := len(vocab)
	train := corpus[:len(corpus)*8/10]
	val := corpus[len(corpus)*8/10 : len(corpus)*9/10]
	if len(val) > EvalCharsSnap {
		val = val[:EvalCharsSnap]
	}

	fmt.Printf("[*] Corpus: %d chars | Vocab: %d\n\n", len(corpus), V)

	// Precompute stats
	uni := computeUnigramKNSnap(train, V)
	bi := computeBigramKNSnap(train, V)
	tri := computeTrigramKNSnap(train, V)

	archs := []ArchSnap{
		{
			Name: "Sequential(KMeans, Dense) [Standard Trigram]",
			Net:  buildKMeansDenseSnap(V, 2),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installContextKMeansSnap(&obs.SequentialLayers[0], V, 2)
				installInterpolatedDenseSnap(&obs.SequentialLayers[1], V, V*V, 2, uni, bi, tri)
			},
		},
		{
			Name: "Sequential(Dense, Dense) [Bigram MLP]",
			Net:  buildDenseDenseSnap(V, 1),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installDenseBigramSnap(&obs.SequentialLayers[0], &obs.SequentialLayers[1], V, bi)
			},
		},
		{
			Name: "Parallel(Unigram, Bigram, Trigram)",
			Net:  buildParallelSnap(V),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installParallelNgramsSnap(obs, V, uni, bi, tri)
			},
		},
		{
			Name: "Sequential(KMeans[Features], Dense) [Feature Trigram]",
			Net:  buildKMeansFeaturesDenseSnap(V, 2),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installContextKMeansSnap(&obs.SequentialLayers[0], V, 2)
				installInterpolatedDenseSnap(&obs.SequentialLayers[1], V, V*V, 2, uni, bi, tri)
			},
		},
		{
			Name: "Sequential(LSTM, Dense) [Recurrent Bigram]",
			Net:  buildLSTMDenseSnap(V),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installLSTMBigramSnap(&obs.SequentialLayers[0], &obs.SequentialLayers[1], V, bi)
			},
		},
		{
			Name: "Sequential(MHA, Dense) [Attention Bigram]",
			Net:  buildMHADenseSnap(V),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installMHABigramSnap(&obs.SequentialLayers[0], &obs.SequentialLayers[1], V, bi)
			},
		},
		{
			Name: "Sequential(CNN1, Dense) [Local Bigram Filter]",
			Net:  buildCNN1DenseSnap(V),
			Init: func(n *poly.VolumetricNetwork) {
				obs := n.Layers[0].MetaObservedLayer
				installCNN1BigramSnap(&obs.SequentialLayers[0], &obs.SequentialLayers[1], V, bi)
			},
		},
	}

	results := make(map[string]float64)
	rng := rand.New(rand.NewSource(42))

	// Baseline
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  BASELINE  —  Random uniform guess")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	pplRandom := float64(V)
	fmt.Printf("  Expected random perplexity: %.2f\n\n", pplRandom)

	for i, arch := range archs {
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("  [ARCH %d]  %s\n", i+1, arch.Name)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		t0 := time.Now()
		arch.Init(arch.Net)
		snapTime := time.Since(t0)

		ppl := evalPerplexitySnap(arch, val, V)
		results[arch.Name] = ppl

		improvement := pplRandom - ppl
		if ppl > 1e10 { improvement = 0 }

		fmt.Printf("  Snap time: %v  |  Perplexity: %.2f  (Δ=%.1f vs random)\n\n",
			snapTime, ppl, improvement)

		fmt.Println("  Generated text sample:")
		inputDim := arch.Net.Layers[0].InputHeight
		ctxLen := inputDim / V
		if ctxLen < 1 { ctxLen = 1 }
		seedChars := make([]int, ctxLen)
		for s := 0; s < ctxLen; s++ {
			seedChars[s] = train[100+s]
		}
		sample := generateTextSnap(arch, seedChars, vocab, V, GenLengthSnap, 1.0, rng)
		printWrappedSnap(sample)
		fmt.Println()
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                ARCHITECTURE COMPARISON SUMMARY                   ║")
	fmt.Println("╠══════════════════════════════════════════════════╦══════════════╣")
	fmt.Println("║ Architecture                                     ║  Perplexity  ║")
	fmt.Println("╠══════════════════════════════════════════════════╬══════════════╣")
	keys := make([]string, 0, len(results))
	for k := range results {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("║ %-48s ║  %8.2f    ║\n", k, results[k])
	}
	fmt.Println("╚══════════════════════════════════════════════════╩══════════════╝")
	fmt.Println()

	fmt.Println("  Perplexity improvement (longer bar = better):")
	for _, k := range keys {
		ppl := results[k]
		if ppl > 1000000 { ppl = 1000000 }
		bar := pplxBarSnap(ppl, pplRandom)
		fmt.Printf("  %-48s  ppl=%8.2f  %s\n", k, results[k], bar)
	}
}

// ── BUILD HELPERS ────────────────────────────────────────────────────────────

func buildKMeansDenseSnap(V, ctxLen int) *poly.VolumetricNetwork {
	numClusters := 1
	for i := 0; i < ctxLen; i++ {
		numClusters *= V
	}
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
	if err != nil { panic(fmt.Sprintf("buildKMeansDense: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func buildKMeansFeaturesDenseSnap(V, ctxLen int) *poly.VolumetricNetwork {
	numClusters := 1
	for i := 0; i < ctxLen; i++ {
		numClusters *= V
	}
	inputDim := ctxLen * V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d, "output_mode": "features"},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, inputDim, inputDim, numClusters, inputDim, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildKMeansFeaturesDense: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func buildDenseDenseSnap(V, ctxLen int) *poly.VolumetricNetwork {
	inputDim := ctxLen * V
	hidden := V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "relu"},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, inputDim, hidden, hidden, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildDenseDense: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func buildParallelSnap(V int) *poly.VolumetricNetwork {
	inputDim := 2 * V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "parallel",
			"combine_mode": "add",
			"input_height": %d, "output_height": %d,
			"parallel_branches": [
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"},
				{
					"type": "sequential", 
					"input_height": %d, "output_height": %d,
					"sequential_layers": [
						{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d},
						{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
					]
				},
				{
					"type": "sequential", 
					"input_height": %d, "output_height": %d,
					"sequential_layers": [
						{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d},
						{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
					]
				}
			]
		}]
	}`, inputDim, V, inputDim, V, inputDim, V, inputDim, V, V, V, V, inputDim, V, inputDim, V*V, V*V, V*V, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildParallel: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func buildLSTMDenseSnap(V int) *poly.VolumetricNetwork {
	ctxLen := 2
	inputDim := V * ctxLen
	hidden := V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "lstm", "input_height": %d, "output_height": %d, "seq_length": %d},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, V, hidden, ctxLen, hidden, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildLSTMDense: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func buildMHADenseSnap(V int) *poly.VolumetricNetwork {
	ctxLen := 2
	inputDim := V * ctxLen
	dModel := V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "mha", "input_height": %d, "output_height": %d, "d_model": %d, "num_heads": 1, "head_dim": %d},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, inputDim, dModel, dModel, dModel, dModel, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildMHADense: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

func buildCNN1DenseSnap(V int) *poly.VolumetricNetwork {
	ctxLen := 3
	inputDim := V * ctxLen
	filters := V
	json := fmt.Sprintf(`{
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"input_height": %d, "output_height": %d,
			"sequential_layers": [
				{"type": "cnn1", "input_height": %d, "input_channels": %d, "output_height": 1, "filters": %d, "kernel_size": %d, "stride": 1, "padding": 0},
				{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
			]
		}]
	}`, inputDim, V, ctxLen, V, filters, ctxLen, filters, V)
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(fmt.Sprintf("buildCNN1Dense: %v", err)) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

// ── INSTALL HELPERS ──────────────────────────────────────────────────────────

func installContextKMeansSnap(l *poly.VolumetricLayer, V, ctxLen int) {
	l.KMeansTemperature = 0.1 // Sharp lookup
	inputDim := l.InputHeight
	numClusters := 1
	for i := 0; i < ctxLen; i++ {
		numClusters *= V
	}

	l.NumClusters = numClusters
	l.OutputHeight = (func() int {
		if l.KMeansOutputMode == "features" { return inputDim }
		return numClusters
	})()
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

func installInterpolatedDenseSnap(l *poly.VolumetricLayer, V, numClusters, ctxLen int, uni []float32, bi [][]float32, tri map[int][]float32) {
	if l.InputHeight != numClusters {
		l.WeightStore = poly.NewWeightStore(l.InputHeight * V)
		for next := 0; next < V; next++ {
			logP := math.Log(float64(uni[next]) + 1e-10)
			for i := 0; i < l.InputHeight; i++ {
				l.WeightStore.Master[next*l.InputHeight+i] = float32(logP) / float32(l.InputHeight)
			}
			lastCharOffset := l.InputHeight - V
			for charIdx := 0; charIdx < V; charIdx++ {
				p := float64(bi[charIdx][next])
				l.WeightStore.Master[next*l.InputHeight + lastCharOffset + charIdx] = float32(math.Log(p + 1e-10))
			}
		}
		return
	}

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

func installDenseBigramSnap(l1, l2 *poly.VolumetricLayer, V int, bi [][]float32) {
	l1.WeightStore = poly.NewWeightStore(l1.InputHeight * l1.OutputHeight)
	for i := 0; i < V; i++ {
		if l1.InputHeight >= 2*V {
			l1.WeightStore.Master[i*l1.InputHeight + V + i] = 1.0
		} else {
			l1.WeightStore.Master[i*l1.InputHeight + i] = 1.0
		}
	}

	l2.WeightStore = poly.NewWeightStore(l2.InputHeight * l2.OutputHeight)
	for i := 0; i < l2.InputHeight; i++ {
		for j := 0; j < l2.OutputHeight; j++ {
			l2.WeightStore.Master[j*l2.InputHeight+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10))
		}
	}
}

func installParallelNgramsSnap(obs *poly.VolumetricLayer, V int, uni []float32, bi [][]float32, tri map[int][]float32) {
	lUni := &obs.ParallelBranches[0]
	lUni.WeightStore = poly.NewWeightStore(lUni.InputHeight * lUni.OutputHeight)
	for c := 0; c < lUni.InputHeight; c++ {
		for next := 0; next < V; next++ {
			lUni.WeightStore.Master[next*lUni.InputHeight+c] = float32(math.Log(float64(uni[next]) + 1e-10))
		}
	}

	lBiSeq := &obs.ParallelBranches[1]
	installContextKMeansSnap(&lBiSeq.SequentialLayers[0], V, 1)
	lBiDense := &lBiSeq.SequentialLayers[1]
	lBiDense.WeightStore = poly.NewWeightStore(V * V)
	for c := 0; c < V; c++ {
		for next := 0; next < V; next++ {
			lBiDense.WeightStore.Master[next*V+c] = float32(math.Log(float64(bi[c][next]) + 1e-10))
		}
	}

	lTriSeq := &obs.ParallelBranches[2]
	installContextKMeansSnap(&lTriSeq.SequentialLayers[0], V, 2)
	lTriDense := &lTriSeq.SequentialLayers[1]
	K := V * V
	lTriDense.WeightStore = poly.NewWeightStore(V * K)
	for c := 0; c < K; c++ {
		for next := 0; next < V; next++ {
			p := 0.001 / float32(V)
			if probs, ok := tri[c]; ok { p = probs[next] }
			lTriDense.WeightStore.Master[next*K+c] = float32(math.Log(float64(p) + 1e-10))
		}
	}
}

func installLSTMBigramSnap(l1, l2 *poly.VolumetricLayer, V int, bi [][]float32) {
	hidden, inputSize := l1.OutputHeight, l1.InputHeight
	ihSize, hhSize, bSize := hidden*inputSize, hidden*hidden, hidden
	gateSize := ihSize + hhSize + bSize
	l1.WeightStore = poly.NewWeightStore(4 * gateSize)
	for h := 0; h < hidden; h++ {
		l1.WeightStore.Master[gateSize + ihSize + hhSize + h] = 5.0
		l1.WeightStore.Master[ihSize + hhSize + h] = 5.0
		l1.WeightStore.Master[3*gateSize + ihSize + hhSize + h] = 5.0
		if h < inputSize { l1.WeightStore.Master[2*gateSize + h*inputSize + h] = 1.0 }
	}
	l2.WeightStore = poly.NewWeightStore(hidden * V)
	for i := 0; i < hidden; i++ {
		for j := 0; j < V; j++ {
			l2.WeightStore.Master[j*hidden+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10))
		}
	}
}

func installMHABigramSnap(l1, l2 *poly.VolumetricLayer, V int, bi [][]float32) {
	dModel := l1.DModel
	kvDim := dModel
	l1.WeightStore = poly.NewWeightStore(2*dModel*dModel + 2*dModel*kvDim + 2*dModel + 2*kvDim)
	for i := 0; i < dModel; i++ {
		l1.WeightStore.Master[i*dModel+i] = 1.0
		l1.WeightStore.Master[dModel*dModel + i*dModel+i] = 1.0
		l1.WeightStore.Master[dModel*dModel + dModel*kvDim + i*dModel+i] = 1.0
		l1.WeightStore.Master[dModel*dModel + 2*dModel*kvDim + i*dModel+i] = 1.0
	}
	l2.WeightStore = poly.NewWeightStore(dModel * V)
	for i := 0; i < dModel; i++ {
		for j := 0; j < V; j++ {
			l2.WeightStore.Master[j*dModel+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10))
		}
	}
}

func installCNN1BigramSnap(l1, l2 *poly.VolumetricLayer, V int, bi [][]float32) {
	l1.WeightStore = poly.NewWeightStore(l1.Filters * l1.InputChannels * l1.KernelSize)
	for f := 0; f < l1.Filters; f++ {
		l1.WeightStore.Master[f*l1.InputChannels*3 + f*3 + 1] = 1.0
	}
	l2.WeightStore = poly.NewWeightStore(l1.Filters * V)
	for i := 0; i < l1.Filters; i++ {
		for j := 0; j < V; j++ {
			l2.WeightStore.Master[j*l1.Filters+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10))
		}
	}
}

// ── STATS & EVAL ─────────────────────────────────────────────────────────────

func computeUnigramKNSnap(corpus []int, V int) []float32 {
	counts := make([]int, V)
	for _, c := range corpus { counts[c]++ }
	total := float64(len(corpus))
	probs := make([]float32, V)
	for i, c := range counts { probs[i] = float32(float64(c) / total) }
	return probs
}

func computeBigramKNSnap(corpus []int, V int) [][]float32 {
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

func computeTrigramKNSnap(corpus []int, V int) map[int][]float32 {
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

func evalPerplexitySnap(arch ArchSnap, corpus []int, V int) float64 {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	if ctxLen < 1 { ctxLen = 1 }
	if len(corpus) <= ctxLen { return 100 }

	// Prepare input tensor with correct shape
	var input *poly.Tensor[float32]
	if arch.Name == "Sequential(CNN1, Dense) [Local Bigram Filter]" {
		input = poly.NewTensor[float32](1, V, ctxLen)
	} else if arch.Name == "Sequential(LSTM, Dense) [Recurrent Bigram]" || 
	          arch.Name == "Sequential(MHA, Dense) [Attention Bigram]" {
		input = poly.NewTensor[float32](1, ctxLen, V)
	} else {
		input = poly.NewTensor[float32](1, inputDim)
	}

	totalLogProb := 0.0
	for i := ctxLen; i < len(corpus); i++ {
		fillInputSnap(input, corpus[i-ctxLen:i], V)
		output, _, _ := poly.ForwardPolymorphic(net, input)
		
		// If output is a sequence (MHA/LSTM output), take the last step
		data := output.Data
		if len(output.Shape) == 3 {
			// [Batch, Seq, Features] -> take last seq step
			features := output.Shape[2]
			data = data[len(data)-features:]
		}
		
		probs := softmaxTempSnap(data, 1.0)
		p := float64(probs[corpus[i]])
		if p < 1e-10 { p = 1e-10 }
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-ctxLen))
}

func generateTextSnap(arch ArchSnap, seedChars []int, vocab []byte, V, length int, temp float64, rng *rand.Rand) string {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	if ctxLen < 1 { ctxLen = 1 }

	var input *poly.Tensor[float32]
	if arch.Name == "Sequential(CNN1, Dense) [Local Bigram Filter]" {
		input = poly.NewTensor[float32](1, V, ctxLen)
	} else if arch.Name == "Sequential(LSTM, Dense) [Recurrent Bigram]" || 
	          arch.Name == "Sequential(MHA, Dense) [Attention Bigram]" {
		input = poly.NewTensor[float32](1, ctxLen, V)
	} else {
		input = poly.NewTensor[float32](1, inputDim)
	}

	result := make([]byte, 0, length)
	for _, c := range seedChars { result = append(result, vocab[c]) }
	context := make([]int, ctxLen)
	copy(context, seedChars)

	for i := 0; i < length; i++ {
		fillInputSnap(input, context, V)
		output, _, _ := poly.ForwardPolymorphic(net, input)
		
		data := output.Data
		if len(output.Shape) == 3 {
			features := output.Shape[2]
			data = data[len(data)-features:]
		}

		probs := softmaxTempSnap(data, temp)
		next := sampleCategoricalSnap(probs, rng)
		result = append(result, vocab[next])
		if ctxLen > 1 { copy(context, context[1:]); context[ctxLen-1] = next } else { context[0] = next }
	}
	return string(result)
}

func fillInputSnap(t *poly.Tensor[float32], context []int, V int) {
	for k := range t.Data { t.Data[k] = 0 }
	if len(t.Shape) == 3 {
		dim1, dim2 := t.Shape[1], t.Shape[2]
		if dim1 == V { // [Batch, Channels, SeqLen]
			for j, c := range context { t.Data[c*dim2 + j] = 1.0 }
		} else { // [Batch, SeqLen, Channels]
			for j, c := range context { t.Data[j*dim2 + c] = 1.0 }
		}
	} else {
		for j, c := range context { t.Data[j*V + c] = 1.0 }
	}
}

func softmaxTempSnap(v []float32, temp float64) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v { if x > maxV { maxV = x } }
	sum := float32(0)
	for i, x := range v { out[i] = float32(math.Exp(float64(x-maxV) / temp)); sum += out[i] }
	for i := range out { out[i] /= sum }
	return out
}

func sampleCategoricalSnap(probs []float32, rng *rand.Rand) int {
	u := rng.Float32()
	cum := float32(0)
	for i, p := range probs { cum += p; if u < cum { return i } }
	return len(probs) - 1
}

// ── DISPLAY HELPERS ─────────────────────────────────────────────────────────

func pplxBarSnap(ppl, worst float64) string {
	const width = 30
	ratio := 1.0 - (math.Log(ppl) / math.Log(worst))
	if ratio < 0 { ratio = 0 }
	if ratio > 1 { ratio = 1 }
	bar := make([]byte, width)
	for i := 0; i < width; i++ { if i < int(ratio*float64(width)) { bar[i] = '#' } else { bar[i] = '.' } }
	return string(bar)
}

func printWrappedSnap(text string) {
	const lineWidth = 72
	runes := []rune(text)
	for i := 0; i < len(runes); i += lineWidth {
		end := i + lineWidth
		if end > len(runes) { end = len(runes) }
		fmt.Printf("  %s\n", string(runes[i:end]))
	}
}

func ensureCorpusSnap() error {
	if _, err := os.Stat(CorpusFileSnap); err == nil { return nil }
	os.MkdirAll(DataDirSnap, 0755)
	resp, err := http.Get(CorpusURLSnap)
	if err != nil { return err }
	defer resp.Body.Close()
	f, _ := os.Create(CorpusFileSnap)
	defer f.Close()
	io.Copy(f, resp.Body)
	return nil
}

func loadCorpusSnap() (corpus []int, vocab []byte, charToIdx map[byte]int) {
	data, _ := os.ReadFile(CorpusFileSnap)
	seen := make(map[byte]bool)
	for _, b := range data { seen[b] = true }
	for b := range seen { vocab = append(vocab, b) }
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx = make(map[byte]int)
	for i, c := range vocab { charToIdx[c] = i }
	for _, b := range data { corpus = append(corpus, charToIdx[b]) }
	return
}
