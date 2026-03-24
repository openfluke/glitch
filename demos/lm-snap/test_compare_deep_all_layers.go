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
	DataDirUSnap    = "data"
	CorpusFileUSnap = "data/shakespeare.txt"
	CorpusURLUSnap  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	EvalCharsUSnap  = 4000
	GenLengthUSnap  = 150
)

type ArchUSnap struct {
	Name string
	Net  *poly.VolumetricNetwork
	Init func(*poly.VolumetricNetwork, int, int, []string, map[string][]int, [][]float32)
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║      DEEP UNIVERSAL BENCHMARK  ·  MATRIX SHOWDOWN              ║")
	fmt.Println("║  Testing All Layers across 3, 4, 5, and 6-gram Scaling        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	if err := ensureCorpusUSnap(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	corpus, vocab, _ := loadCorpusUSnap()
	V := len(vocab)
	train := corpus[:len(corpus)*8/10]
	val := corpus[len(corpus)*8/10 : len(corpus)*9/10]
	if len(val) > EvalCharsUSnap { val = val[:EvalCharsUSnap] }

	fmt.Printf("[*] Corpus: %d chars | Vocab: %d\n\n", len(corpus), V)

	bi := computeBigramUSnap(train, V)
	rng := rand.New(rand.NewSource(42))

	archSpecs := []ArchUSnap{
		{
			Name: "KMeans (Sparse)",
			Init: installSparseKMeansUSnap,
		},
		{
			Name: "LSTM (Recurrent)",
			Init: installLSTMUSnap,
		},
		{
			Name: "MHA (Attention)",
			Init: installMHAUSnap,
		},
		{
			Name: "CNN1 (Convolutional)",
			Init: installCNN1USnap,
		},
	}

	for ctxLen := 2; ctxLen <= 5; ctxLen++ {
		gen := ctxLen + 1
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("  [GENERATION]  %d-gram (ctxLen=%d)\n", gen, ctxLen)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		
		fmt.Printf("  Analyzing corpus for patterns...\n")
		counts, prefixes := analyzeNgramsUSnap(train, V, ctxLen)
		numClusters := len(prefixes)
		fmt.Printf("  Found %d unique patterns.\n\n", numClusters)

		for _, spec := range archSpecs {
			name := fmt.Sprintf("%dG-%s", gen, spec.Name)
			net := buildNetUSnap(spec.Name, V, ctxLen, numClusters)
			
			t0 := time.Now()
			spec.Init(net, V, ctxLen, prefixes, counts, bi)
			snapTime := time.Since(t0)

			arch := ArchUSnap{Name: name, Net: net}
			ppl := evalPerplexityUSnap(arch, val, V)

			fmt.Printf("  %-25s  ppl: %8.2f  time: %5.2fms\n", 
				spec.Name, ppl, float64(snapTime.Microseconds())/1000.0)

			// Generation sample for the first ctxLen
			if ctxLen == 2 && spec.Name == "KMeans (Sparse)" {
				fmt.Println("  Sample text:")
				seedChars := make([]int, ctxLen)
				for s := 0; s < ctxLen; s++ { seedChars[s] = train[100+s] }
				sample := generateTextUSnap(arch, seedChars, vocab, V, GenLengthUSnap, 1.0, rng)
				printWrappedUSnap(sample)
			}
		}
		fmt.Println()
	}
}

// ── ANALYSIS HELPERS ─────────────────────────────────────────────────────────

func analyzeNgramsUSnap(corpus []int, V, ctxLen int) (counts map[string][]int, prefixes []string) {
	counts = make(map[string][]int)
	for i := ctxLen; i < len(corpus); i++ {
		prefix := ""
		for j := 0; j < ctxLen; j++ { prefix += fmt.Sprintf("%03d|", corpus[i-ctxLen+j]) }
		if _, ok := counts[prefix]; !ok {
			counts[prefix] = make([]int, V)
			prefixes = append(prefixes, prefix)
		}
		counts[prefix][corpus[i]]++
	}
	sort.Strings(prefixes)
	return
}

func parsePrefixUSnap(prefix string) []int {
	var res []int
	for i := 0; i < len(prefix); i += 4 {
		var n int
		fmt.Sscanf(prefix[i:i+3], "%d", &n)
		res = append(res, n)
	}
	return res
}

// ── BUILD HELPERS ────────────────────────────────────────────────────────────

func buildNetUSnap(kind string, V, ctxLen, numClusters int) *poly.VolumetricNetwork {
	inputDim := ctxLen * V
	var json string
	
	switch kind {
	case "KMeans (Sparse)":
		json = fmt.Sprintf(`{
			"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
			"layers": [{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential", "input_height": %d, "output_height": %d,
				"sequential_layers": [
					{"type": "kmeans", "input_height": %d, "output_height": %d, "num_clusters": %d},
					{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
				]
			}]
		}`, inputDim, V, inputDim, numClusters, numClusters, numClusters, V)
	case "LSTM (Recurrent)":
		json = fmt.Sprintf(`{
			"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
			"layers": [{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential", "input_height": %d, "output_height": %d,
				"sequential_layers": [
					{"type": "lstm", "input_height": %d, "output_height": %d, "seq_length": %d},
					{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
				]
			}]
		}`, inputDim, V, V, V, ctxLen, V, V)
	case "MHA (Attention)":
		json = fmt.Sprintf(`{
			"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
			"layers": [{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential", "input_height": %d, "output_height": %d,
				"sequential_layers": [
					{"type": "mha", "input_height": %d, "output_height": %d, "d_model": %d, "num_heads": 1, "head_dim": %d},
					{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
				]
			}]
		}`, inputDim, V, inputDim, V, V, V, V, V)
	case "CNN1 (Convolutional)":
		json = fmt.Sprintf(`{
			"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
			"layers": [{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential", "input_height": %d, "output_height": %d,
				"sequential_layers": [
					{"type": "cnn1", "input_height": %d, "input_channels": %d, "output_height": 1, "filters": %d, "kernel_size": %d, "stride": 1, "padding": 0},
					{"type": "dense", "input_height": %d, "output_height": %d, "activation": "linear"}
				]
			}]
		}`, inputDim, V, ctxLen, V, V, ctxLen, V, V)
	}

	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil { panic(err) }
	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	return net
}

// ── INSTALL HELPERS ──────────────────────────────────────────────────────────

func installSparseKMeansUSnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int, bi [][]float32) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	km, ds := &seq[0], &seq[1]
	numClusters, inputDim := len(prefixes), ctxLen * V

	km.KMeansTemperature = 0.05
	km.WeightStore = poly.NewWeightStore(numClusters * inputDim)
	ds.WeightStore = poly.NewWeightStore(numClusters * V)

	for c, prefix := range prefixes {
		chars := parsePrefixUSnap(prefix)
		for i, charIdx := range chars { km.WeightStore.Master[c*inputDim + i*V + charIdx] = 1.0 }
		total, cnts := 0, counts[prefix]
		for _, v := range cnts { total += v }
		for next, count := range cnts {
			p := float64(count) / float64(total)
			ds.WeightStore.Master[next*numClusters + c] = float32(math.Log(p + 1e-10))
		}
	}
}

func installLSTMUSnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int, bi [][]float32) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	l1, l2 := &seq[0], &seq[1]
	hidden, inputSize := V, V
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
		for j := 0; j < V; j++ { l2.WeightStore.Master[j*hidden+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10)) }
	}
}

func installMHAUSnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int, bi [][]float32) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	l1, l2 := &seq[0], &seq[1]
	dModel, kvDim := V, V
	l1.WeightStore = poly.NewWeightStore(2*dModel*dModel + 2*dModel*kvDim + 2*dModel + 2*kvDim)
	for i := 0; i < dModel; i++ {
		l1.WeightStore.Master[i*dModel+i] = 1.0
		l1.WeightStore.Master[dModel*dModel + i*dModel+i] = 1.0
		l1.WeightStore.Master[dModel*dModel + dModel*kvDim + i*dModel+i] = 1.0
		l1.WeightStore.Master[dModel*dModel + 2*dModel*kvDim + i*dModel+i] = 1.0
	}
	l2.WeightStore = poly.NewWeightStore(dModel * V)
	for i := 0; i < dModel; i++ {
		for j := 0; j < V; j++ { l2.WeightStore.Master[j*dModel+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10)) }
	}
}

func installCNN1USnap(net *poly.VolumetricNetwork, V, ctxLen int, prefixes []string, counts map[string][]int, bi [][]float32) {
	seq := net.Layers[0].MetaObservedLayer.SequentialLayers
	l1, l2 := &seq[0], &seq[1]
	l1.WeightStore = poly.NewWeightStore(l1.Filters * l1.InputChannels * l1.KernelSize)
	for f := 0; f < l1.Filters; f++ {
		// Detect char f at the last position in the window
		l1.WeightStore.Master[f*l1.InputChannels*l1.KernelSize + f*l1.KernelSize + (l1.KernelSize-1)] = 1.0
	}
	l2.WeightStore = poly.NewWeightStore(l1.Filters * V)
	for i := 0; i < l1.Filters; i++ {
		for j := 0; j < V; j++ { l2.WeightStore.Master[j*l1.Filters+i] = float32(math.Log(float64(bi[i%V][j]) + 1e-10)) }
	}
}

// ── STATS & EVAL ─────────────────────────────────────────────────────────────

func evalPerplexityUSnap(arch ArchUSnap, corpus []int, V int) float64 {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	if len(corpus) <= ctxLen { return 100 }
	var input *poly.Tensor[float32]
	if arch.Name[3:] == "CNN1 (Convolutional)" { input = poly.NewTensor[float32](1, V, ctxLen)
	} else if arch.Name[3:] == "LSTM (Recurrent)" || arch.Name[3:] == "MHA (Attention)" { input = poly.NewTensor[float32](1, ctxLen, V)
	} else { input = poly.NewTensor[float32](1, inputDim) }
	
	totalLogProb := 0.0
	for i := ctxLen; i < len(corpus); i++ {
		fillInputUSnap(input, corpus[i-ctxLen:i], V)
		output, _, _ := poly.ForwardPolymorphic(net, input)
		data := output.Data
		if len(output.Shape) == 3 { data = data[len(data)-output.Shape[2]:] }
		probs := softmaxTempUSnap(data, 1.0)
		p := float64(probs[corpus[i]])
		if p < 1e-10 { p = 1e-10 }
		totalLogProb += math.Log(p)
	}
	return math.Exp(-totalLogProb / float64(len(corpus)-ctxLen))
}

func fillInputUSnap(t *poly.Tensor[float32], context []int, V int) {
	for k := range t.Data { t.Data[k] = 0 }
	if len(t.Shape) == 3 {
		dim1, dim2 := t.Shape[1], t.Shape[2]
		if dim1 == V { for j, c := range context { t.Data[c*dim2 + j] = 1.0 } } else { for j, c := range context { t.Data[j*dim2 + c] = 1.0 } }
	} else { for j, c := range context { t.Data[j*V + c] = 1.0 } }
}

func softmaxTempUSnap(v []float32, temp float64) []float32 {
	out := make([]float32, len(v))
	maxV := v[0]
	for _, x := range v { if x > maxV { maxV = x } }
	sum := float32(0)
	for i, x := range v { out[i] = float32(math.Exp(float64(x-maxV) / temp)); sum += out[i] }
	for i := range out { out[i] /= sum }
	return out
}

func generateTextUSnap(arch ArchUSnap, seedChars []int, vocab []byte, V, length int, temp float64, rng *rand.Rand) string {
	net := arch.Net
	inputDim := net.Layers[0].InputHeight
	ctxLen := inputDim / V
	result := make([]byte, 0, length)
	for _, c := range seedChars { result = append(result, vocab[c]) }
	context := make([]int, ctxLen)
	copy(context, seedChars)
	
	var input *poly.Tensor[float32]
	if arch.Name[3:] == "CNN1 (Convolutional)" { input = poly.NewTensor[float32](1, V, ctxLen)
	} else if arch.Name[3:] == "LSTM (Recurrent)" || arch.Name[3:] == "MHA (Attention)" { input = poly.NewTensor[float32](1, ctxLen, V)
	} else { input = poly.NewTensor[float32](1, inputDim) }

	for i := 0; i < length; i++ {
		fillInputUSnap(input, context, V)
		output, _, _ := poly.ForwardPolymorphic(net, input)
		data := output.Data
		if len(output.Shape) == 3 { data = data[len(data)-output.Shape[2]:] }
		probs := softmaxTempUSnap(data, temp)
		next := sampleCategoricalUSnap(probs, rng)
		result = append(result, vocab[next])
		if ctxLen > 1 { copy(context, context[1:]); context[ctxLen-1] = next } else { context[0] = next }
	}
	return string(result)
}

func sampleCategoricalUSnap(probs []float32, rng *rand.Rand) int {
	u := rng.Float32()
	cum := float32(0)
	for i, p := range probs { cum += p; if u < cum { return i } }
	return len(probs) - 1
}

func printWrappedUSnap(text string) {
	const lineWidth = 72
	runes := []rune(text)
	for i := 0; i < len(runes); i += lineWidth {
		end := i + lineWidth
		if end > len(runes) { end = len(runes) }
		fmt.Printf("  %s\n", string(runes[i:end]))
	}
}

func ensureCorpusUSnap() error {
	if _, err := os.Stat(CorpusFileUSnap); err == nil { return nil }
	os.MkdirAll(DataDirUSnap, 0755)
	resp, err := http.Get(CorpusURLUSnap)
	if err != nil { return err }
	defer resp.Body.Close()
	f, _ := os.Create(CorpusFileUSnap)
	defer f.Close()
	io.Copy(f, resp.Body)
	return nil
}

func loadCorpusUSnap() (corpus []int, vocab []byte, charToIdx map[byte]int) {
	data, _ := os.ReadFile(CorpusFileUSnap)
	seen := make(map[byte]bool)
	for _, b := range data { seen[b] = true }
	for b := range seen { vocab = append(vocab, b) }
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	charToIdx = make(map[byte]int)
	for i, c := range vocab { charToIdx[c] = i }
	for _, b := range data { corpus = append(corpus, charToIdx[b]) }
	return
}

func computeUnigramUSnap(corpus []int, V int) []float32 {
	counts := make([]int, V)
	for _, c := range corpus { counts[c]++ }
	total := float64(len(corpus))
	probs := make([]float32, V)
	for i, c := range counts { probs[i] = float32(float64(c) / total) }
	return probs
}

func computeBigramUSnap(corpus []int, V int) [][]float32 {
	counts := make([][]int, V)
	for i := range counts { counts[i] = make([]int, V) }
	for i := 1; i < len(corpus); i++ { counts[corpus[i-1]][corpus[i]]++ }
	bigram := make([][]float32, V)
	for p := range bigram {
		bigram[p] = make([]float32, V)
		total := 0
		for _, c := range counts[p] { total += c }
		for n := range bigram[p] { if total > 0 { bigram[p][n] = float32(float64(counts[p][n]) / float64(total)) } else { bigram[p][n] = 1.0 / float32(V) } }
	}
	return bigram
}
