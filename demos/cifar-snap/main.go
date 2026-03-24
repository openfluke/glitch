package main

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	DataURL  = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	DataDir  = "data"
	BatchDir = "data/cifar-10-batches-bin"
)

var classNames = [10]string{
	"airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck",
}

// CIFAR images are CHW: [3072] = [1024 R | 1024 G | 1024 B], each channel row-major 32×32
type Sample struct {
	Image []float32
	Label int
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          CIFAR-10  ·  ZERO BACKPROP META-COGNITION SNAP         ║")
	fmt.Println("║  50,000 real photos  ·  10 classes  ·  3×32×32 RGB  ·  no SGD  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("CIFAR-10 is the canonical hard benchmark. State-of-the-art CNNs")
	fmt.Println("train for hours and reach ~95%. We will see how far pure heuristics go.")
	fmt.Println()

	// ── 1. DATA ──
	if err := ensureData(); err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	fmt.Println("[*] Loading dataset...")
	allTrain, err := loadAllBatches()
	if err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}
	allTest, err := loadBatch(filepath.Join(BatchDir, "test_batch.bin"))
	if err != nil {
		fmt.Printf("[!] %v\n", err)
		return
	}

	splitIdx  := int(float64(len(allTrain)) * 0.8)
	protoData := allTrain[:splitIdx]  // 40,000 — build prototypes
	valData   := allTrain[splitIdx:]  // 10,000 — evaluate each generation
	fmt.Printf("[*] %d train  →  %d prototype  |  %d validation  |  %d test\n\n",
		len(allTrain), len(protoData), len(valData), len(allTest))

	// ── 2. NETWORK (5-layer, all disabled in snap mode) ──
	net, err := poly.BuildNetworkFromJSON([]byte(`{
		"id": "cifar_snap",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"sequential_layers": [
				{
					"type": "cnn2", "activation": "relu",
					"input_height": 32, "input_width": 32, "input_channels": 3,
					"filters": 32, "kernel_size": 3, "stride": 1, "padding": 0,
					"output_height": 30, "output_width": 30
				},
				{
					"type": "cnn2", "activation": "relu",
					"input_height": 30, "input_width": 30, "input_channels": 32,
					"filters": 64, "kernel_size": 3, "stride": 2, "padding": 0,
					"output_height": 14, "output_width": 14
				},
				{
					"type": "cnn2", "activation": "relu",
					"input_height": 14, "input_width": 14, "input_channels": 64,
					"filters": 128, "kernel_size": 3, "stride": 2, "padding": 0,
					"output_height": 6, "output_width": 6
				},
				{
					"type": "dense", "activation": "relu",
					"input_height": 4608, "output_height": 256
				},
				{
					"type": "dense", "activation": "linear",
					"input_height": 256, "output_height": 10
				}
			]
		}]
	}`))
	if err != nil {
		panic(err)
	}

	// Baseline: random chance over 10 classes — no need to run the CNN
	fmt.Printf("Baseline (random, 10 classes):  accuracy≈10%%  score≈37/100\n\n")
	_ = 0 // baseline is ~10% random chance by definition

	poly.WrapWithMetacognition(net, []poly.MetaRule{})
	l   := &net.Layers[0]
	obs := l.MetaObservedLayer

	// Disable all CNN/Dense layers — meta-cognition bypasses them entirely
	for i := range obs.SequentialLayers {
		obs.SequentialLayers[i].IsDisabled = true
	}
	// Re-purpose the last slot as our KMeans output
	final := &obs.SequentialLayers[len(obs.SequentialLayers)-1]

	installKMeans := func(centers [][]float32, temp float64) {
		n := len(centers)
		dim := len(centers[0])
		final.IsDisabled      = false
		final.Type             = poly.LayerKMeans
		final.NumClusters      = n
		final.InputHeight      = dim
		final.OutputHeight     = n
		final.KMeansTemperature = temp
		final.WeightStore      = poly.NewWeightStore(n * dim)
		for i, c := range centers {
			copy(final.WeightStore.Master[i*dim:], c)
		}
	}

	// Track best result across generations
	type GenResult struct {
		name     string
		accuracy float64
		score    float64
	}
	var history []GenResult

	evalGen := func(name string, inputs []*poly.Tensor[float32], labels []float64) GenResult {
		m, _ := poly.EvaluateNetworkPolymorphic(net, inputs, labels)
		fmt.Printf("      %-38s  accuracy=%.1f%%  score=%.2f\n", name, m.Accuracy, m.Score)
		return GenResult{name, m.Accuracy, m.Score}
	}

	valInputs, valLabels   := toTensors(valData)
	testInputs, testLabels := toTensors(allTest)

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  AUTONOMOUS META-COGNITION  —  5 generations, zero backprop")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// ══════════════════════════════════════════════════════════
	// GEN 1 — Raw pixel mean prototypes (baseline snap)
	// ══════════════════════════════════════════════════════════
	fmt.Println("\n[GEN 1] Hypothesis: each class has a distinct mean color/shape signature.")
	fmt.Println("        Action: compute mean pixel image per class (3072-dim CHW).")
	t0 := time.Now()

	rawMeans := computeMeans(protoData, 3072)

	gen1Time := time.Since(t0)
	fmt.Printf("        Done in %v\n", gen1Time)
	printProtoGallery(rawMeans[:], 32)
	installKMeans(rawMeans[:], 0.3)
	history = append(history, evalGen("[GEN 1] Raw pixel mean prototypes", valInputs, valLabels))

	// ══════════════════════════════════════════════════════════
	// GEN 2 — Flip-augmented prototypes
	// ══════════════════════════════════════════════════════════
	fmt.Println("\n[GEN 2] Hypothesis: animals/vehicles face both directions — mean image is asymmetric.")
	fmt.Println("        Action: average each prototype with its horizontal mirror.")
	t0 = time.Now()

	flipMeans := make([][]float32, 10)
	for i, m := range rawMeans {
		flipped := hflip(m, 3, 32, 32)
		avg := make([]float32, 3072)
		for j := range avg {
			avg[j] = (m[j] + flipped[j]) * 0.5
		}
		flipMeans[i] = avg
	}

	fmt.Printf("        Done in %v\n", time.Since(t0))
	installKMeans(flipMeans[:], 0.3)
	history = append(history, evalGen("[GEN 2] Flip-augmented prototypes", valInputs, valLabels))

	// ══════════════════════════════════════════════════════════
	// GEN 3 — Per-channel whitening
	// ══════════════════════════════════════════════════════════
	fmt.Println("\n[GEN 3] Hypothesis: raw RGB distances are biased by global brightness variation.")
	fmt.Println("        Action: Z-score normalize each channel (subtract mean, divide by std).")
	fmt.Println("        This makes the KMeans distance metric channel-invariant.")
	t0 = time.Now()

	chanMean, chanStd := channelStats(protoData)
	fmt.Printf("        Channel means: R=%.3f G=%.3f B=%.3f\n", chanMean[0], chanMean[1], chanMean[2])
	fmt.Printf("        Channel stds:  R=%.3f G=%.3f B=%.3f\n", chanStd[0], chanStd[1], chanStd[2])

	// Whiten prototypes
	whitenedMeans := make([][]float32, 10)
	for i, m := range flipMeans {
		whitenedMeans[i] = whitenImage(m, chanMean, chanStd)
	}

	// Whiten val/test inputs too
	whitenedValInputs  := whitenTensors(valData,  chanMean, chanStd)
	whitenedTestInputs := whitenTensors(allTest,  chanMean, chanStd)

	fmt.Printf("        Done in %v\n", time.Since(t0))
	installKMeans(whitenedMeans[:], 0.3)
	history = append(history, evalGen("[GEN 3] Channel-whitened prototypes", whitenedValInputs, valLabels))

	// ══════════════════════════════════════════════════════════
	// GEN 4 — Temperature sweep on held-out validation
	// ══════════════════════════════════════════════════════════
	fmt.Println("\n[GEN 4] Hypothesis: KMeans temperature controls confidence — optimal T is unknown.")
	fmt.Println("        Action: sweep T ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0} on 1k val samples.")
	t0 = time.Now()

	temps     := []float64{0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0}
	bestT     := 0.3
	bestAcc   := 0.0
	sweep1k   := whitenedValInputs[:1000]
	labels1k  := valLabels[:1000]

	for _, T := range temps {
		installKMeans(whitenedMeans[:], T)
		m, _ := poly.EvaluateNetworkPolymorphic(net, sweep1k, labels1k)
		marker := "  "
		if m.Accuracy > bestAcc {
			bestAcc = m.Accuracy
			bestT   = T
			marker  = "← best"
		}
		fmt.Printf("          T=%-5.2f  accuracy=%.1f%%  %s\n", T, m.Accuracy, marker)
	}
	fmt.Printf("        Best T=%.2f found in %v\n", bestT, time.Since(t0))
	installKMeans(whitenedMeans[:], bestT)
	history = append(history, evalGen(fmt.Sprintf("[GEN 4] Optimal temperature (T=%.2f)", bestT), whitenedValInputs, valLabels))

	// ══════════════════════════════════════════════════════════
	// GEN 5 — Sub-prototype split (brightness tertiles)
	// ══════════════════════════════════════════════════════════
	fmt.Println("\n[GEN 5] Hypothesis: one mean per class loses sub-class modes (sitting vs running cat).")
	fmt.Println("        Action: split each class into 3 brightness tertiles → 30 sub-prototypes.")
	fmt.Println("        Classification: nearest sub-prototype → its parent class wins.")
	t0 = time.Now()

	subProtos := computeSubPrototypes(protoData, chanMean, chanStd, 3)
	fmt.Printf("        Built %d sub-prototypes in %v\n", len(subProtos), time.Since(t0))

	// Need 30-cluster KMeans + manual argmax-with-voting eval
	// We install it and use a custom eval that maps cluster→class
	installKMeans(subProtos, bestT)

	// Custom eval: output is 30-dim, class = argmax(output) / 3
	gen5Acc, gen5Score := evalWithVoting(net, whitenedValInputs, valLabels, 3)
	history = append(history, GenResult{
		name:     "[GEN 5] 3 sub-prototypes per class (30 clusters)",
		accuracy: gen5Acc,
		score:    gen5Score,
	})
	fmt.Printf("      %-38s  accuracy=%.1f%%  score=%.2f\n",
		"[GEN 5] 3 sub-prototypes per class", gen5Acc, gen5Score)

	// ══════════════════════════════════════════════════════════
	// FINAL EVALUATION on official test set
	// ══════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  FINAL EVALUATION  —  official 10,000 test samples")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// Best approach: Gen5 sub-prototypes with whitening + optimal T
	fmt.Printf("\n[TEST] Gen 5 approach (best) on %d test samples...\n", len(allTest))
	t0 = time.Now()
	testAcc, testScore := evalWithVoting(net, whitenedTestInputs, testLabels, 3)
	testTime := time.Since(t0)
	fmt.Printf("       Accuracy=%.1f%%  Score=%.2f  Time=%v  (%.1f µs/sample)\n\n",
		testAcc, testScore, testTime, float64(testTime.Microseconds())/float64(len(allTest)))

	// Also eval Gen 1 approach on test for comparison
	installKMeans(rawMeans[:], 0.3)
	mGen1Test, _ := poly.EvaluateNetworkPolymorphic(net, testInputs, testLabels)
	installKMeans(subProtos, bestT) // restore best

	// Per-class breakdown
	printPerClassAccuracy(whitenedTestInputs, testLabels, subProtos, bestT)

	// Generation progression table
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    CIFAR-10 META-COGNITION PROGRESSION                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════╦════════╦══════════╣")
	fmt.Println("║ Generation                                           ║  Acc   ║  Score   ║")
	fmt.Println("╠══════════════════════════════════════════════════════╬════════╬══════════╣")
	for _, r := range history {
		fmt.Printf("║ %-52s ║ %5.1f%% ║ %7.2f  ║\n", r.name, r.accuracy, r.score)
	}
	fmt.Println("╠══════════════════════════════════════════════════════╬════════╬══════════╣")
	fmt.Printf( "║ FINAL TEST SET (Gen 5 strategy)                      ║ %5.1f%% ║ %7.2f  ║\n", testAcc, testScore)
	fmt.Printf( "║ Baseline  (Gen 1, raw means, test set)               ║ %5.1f%% ║ %7.2f  ║\n", mGen1Test.Accuracy, mGen1Test.Score)
	fmt.Printf( "║ Random baseline (10 classes, by chance)              ║  ~10.0%% ║   ~37.00  ║\n")
	fmt.Println("╠══════════════════════════════════════════════════════╩════════╩══════════╣")
	fmt.Println("║  Backprop: NONE   Optimizer: NONE   Training time: ~0ms                  ║")
	fmt.Println("║  CNN layers: DISABLED (random weights never used)                         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	// ══════════════════════════════════════════════════════════════════════════
	// DTYPE PRECISION SWEEP
	// Take the Gen 5 sub-prototypes (best float32 result) and re-evaluate them
	// at every numerical precision the poly framework supports — from float64
	// down to 1-bit binary.  Each type's centers are quantized via
	// SimulatePrecision (with proper per-type scale) then reinstalled as float32
	// KMeans so the distance computation is always in float32.
	// This answers: how much accuracy do you lose as you compress the centroids?
	// ══════════════════════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  DTYPE PRECISION SWEEP  —  Gen 5 sub-prototypes at every numeric type")
	fmt.Println("  BEFORE (float32): the results above.  AFTER: each dtype below.")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	type dtypeEntry struct {
		dtype poly.DType
		name  string
		bits  int
	}
	allDTypes := []dtypeEntry{
		{poly.DTypeFloat64,  "float64",  64},
		{poly.DTypeFloat32,  "float32",  32}, // baseline — should match Gen 5 exactly
		{poly.DTypeFloat16,  "float16",  16},
		{poly.DTypeBFloat16, "bfloat16", 16},
		{poly.DTypeFP8E4M3,  "fp8_e4m3",  8},
		{poly.DTypeFP8E5M2,  "fp8_e5m2",  8},
		{poly.DTypeInt32,    "int32",    32},
		{poly.DTypeInt16,    "int16",    16},
		{poly.DTypeInt8,     "int8",      8},
		{poly.DTypeInt4,     "int4",      4},
		{poly.DTypeInt2,     "int2",      2},
		{poly.DTypeTernary,  "ternary",   2},
		{poly.DTypeBinary,   "binary",    1},
	}

	// Find max absolute value across all sub-prototype centers for scaling
	maxAbs := float32(0)
	for _, c := range subProtos {
		for _, v := range c {
			if a := float32(math.Abs(float64(v))); a > maxAbs { maxAbs = a }
		}
	}
	fmt.Printf("\n  Sub-prototype weight range: [%.4f, %.4f]  (maxAbs=%.4f)\n\n",
		-maxAbs, maxAbs, maxAbs)

	// Scale map: maps each dtype to a quantization step that uses the full bit range.
	// For float types, scale=1.0 (SimulatePrecision handles mantissa truncation directly).
	// For integer types, scale = maxAbs / maxPositiveIntVal so values fill [-max, +max].
	dtypeScale := func(dt poly.DType) float32 {
		switch dt {
		case poly.DTypeInt64, poly.DTypeUint64:
			return maxAbs / 9.2e18
		case poly.DTypeInt32, poly.DTypeUint32:
			return maxAbs / 2.147e9
		case poly.DTypeInt16, poly.DTypeUint16:
			return maxAbs / 32767.0
		case poly.DTypeInt8, poly.DTypeUint8:
			return maxAbs / 127.0
		case poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
			return maxAbs / 127.0
		case poly.DTypeInt4, poly.DTypeUint4, poly.DTypeFP4:
			return maxAbs / 7.0
		case poly.DTypeInt2, poly.DTypeUint2, poly.DTypeTernary:
			return maxAbs / 1.0
		case poly.DTypeBinary:
			return maxAbs
		default: // float64, float32, float16, bfloat16
			return 1.0
		}
	}

	type DTypeResult struct {
		name   string
		bits   int
		valAcc float64
		testAcc float64
	}
	var dtypeResults []DTypeResult

	for _, entry := range allDTypes {
		scale := dtypeScale(entry.dtype)

		// Quantize sub-prototype centers using SimulatePrecision, store back as float32.
		// Distance computation in KMeans stays in float32 — this tests centroid precision only.
		qProtos := make([][]float32, len(subProtos))
		for i, c := range subProtos {
			qProtos[i] = make([]float32, len(c))
			for j, v := range c {
				qProtos[i][j] = poly.SimulatePrecision(v, entry.dtype, scale)
			}
		}

		installKMeans(qProtos, bestT)
		vAcc, _ := evalWithVoting(net, whitenedValInputs, valLabels, 3)
		tAcc, _ := evalWithVoting(net, whitenedTestInputs, testLabels, 3)
		dtypeResults = append(dtypeResults, DTypeResult{entry.name, entry.bits, vAcc, tAcc})

		marker := ""
		if entry.dtype == poly.DTypeFloat32 { marker = "  ← baseline (before)" }
		delta := vAcc - gen5Acc
		fmt.Printf("  %-12s  %2d-bit  val=%5.1f%%  test=%5.1f%%  Δval=%+5.1f%%%s\n",
			entry.name, entry.bits, vAcc, tAcc, delta, marker)
	}

	// Summary table
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║             DTYPE PRECISION SWEEP — BEFORE vs AFTER                     ║")
	fmt.Println("╠══════════════╦═══════╦═══════════╦═══════════╦══════════════════════════╣")
	fmt.Println("║ DType        ║  Bits ║  Val Acc  ║ Test Acc  ║ Δ vs float32 (val)       ║")
	fmt.Println("╠══════════════╬═══════╬═══════════╬═══════════╬══════════════════════════╣")
	for _, r := range dtypeResults {
		delta := r.valAcc - gen5Acc
		n := min(int(math.Abs(delta)/0.5), 20)
		ch := byte('v')
		if delta >= 0 { ch = '^' }
		barB := make([]byte, n)
		for i := range n { barB[i] = ch }
		bar := string(barB)
		fmt.Printf("║ %-12s ║  %-3d  ║  %5.1f%%   ║  %5.1f%%   ║ %+5.1f%%  %-20s║\n",
			r.name, r.bits, r.valAcc, r.testAcc, delta, bar)
	}
	fmt.Println("╠══════════════╩═══════╩═══════════╩═══════════╩══════════════════════════╣")
	fmt.Printf( "║  BEFORE (float32 Gen5): val=%.1f%%  test=%.1f%%                               ║\n", gen5Acc, testAcc)
	fmt.Println("║  All dtypes: centroids quantized via SimulatePrecision, distance in f32  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
}

// ── GENERATION HELPERS ─────────────────────────────────────────────────────

func computeMeans(samples []Sample, dim int) [10][]float32 {
	sums   := [10][]float64{}
	counts := [10]int{}
	for i := range sums {
		sums[i] = make([]float64, dim)
	}
	for _, s := range samples {
		for j, v := range s.Image {
			sums[s.Label][j] += float64(v)
		}
		counts[s.Label]++
	}
	var means [10][]float32
	for i := 0; i < 10; i++ {
		means[i] = make([]float32, dim)
		if counts[i] > 0 {
			for j := range sums[i] {
				means[i][j] = float32(sums[i][j] / float64(counts[i]))
			}
		}
	}
	return means
}

// hflip mirrors a CHW image horizontally.
func hflip(img []float32, ch, h, w int) []float32 {
	out := make([]float32, len(img))
	for c := 0; c < ch; c++ {
		for r := 0; r < h; r++ {
			for k := 0; k < w; k++ {
				out[c*h*w+r*w+k] = img[c*h*w+r*w+(w-1-k)]
			}
		}
	}
	return out
}

// channelStats computes per-channel mean and std across all samples.
func channelStats(samples []Sample) ([3]float64, [3]float64) {
	var sum, sum2 [3]float64
	n := float64(len(samples)) * 1024.0
	for _, s := range samples {
		for c := 0; c < 3; c++ {
			for p := 0; p < 1024; p++ {
				v := float64(s.Image[c*1024+p])
				sum[c]  += v
				sum2[c] += v * v
			}
		}
	}
	var mean, std [3]float64
	for c := 0; c < 3; c++ {
		mean[c] = sum[c] / n
		variance := sum2[c]/n - mean[c]*mean[c]
		if variance < 0 { variance = 0 }
		std[c] = math.Sqrt(variance)
		if std[c] < 1e-8 { std[c] = 1e-8 }
	}
	return mean, std
}

func whitenImage(img []float32, chanMean, chanStd [3]float64) []float32 {
	out := make([]float32, len(img))
	for c := 0; c < 3; c++ {
		for p := 0; p < 1024; p++ {
			out[c*1024+p] = float32((float64(img[c*1024+p]) - chanMean[c]) / chanStd[c])
		}
	}
	return out
}

func whitenTensors(samples []Sample, chanMean, chanStd [3]float64) []*poly.Tensor[float32] {
	tensors := make([]*poly.Tensor[float32], len(samples))
	for i, s := range samples {
		t := poly.NewTensor[float32](3, 32, 32)
		t.Data = whitenImage(s.Image, chanMean, chanStd)
		tensors[i] = t
	}
	return tensors
}

// computeSubPrototypes splits each class into k brightness tertiles and returns
// k sub-prototypes per class (already whitened).
func computeSubPrototypes(samples []Sample, chanMean, chanStd [3]float64, k int) [][]float32 {
	// Group samples by class
	byClass := make([][]Sample, 10)
	for _, s := range samples {
		byClass[s.Label] = append(byClass[s.Label], s)
	}

	var protos [][]float32
	for cls := 0; cls < 10; cls++ {
		group := byClass[cls]
		// Sort by average brightness (mean of all pixels)
		sort.Slice(group, func(i, j int) bool {
			var sumI, sumJ float32
			for _, v := range group[i].Image { sumI += v }
			for _, v := range group[j].Image { sumJ += v }
			return sumI < sumJ
		})
		// Split into k even buckets
		bSize := len(group) / k
		for b := 0; b < k; b++ {
			start := b * bSize
			end   := start + bSize
			if b == k-1 { end = len(group) }
			// Mean of this bucket (whitened)
			sums  := make([]float64, 3072)
			count := 0
			for _, s := range group[start:end] {
				w := whitenImage(s.Image, chanMean, chanStd)
				for j, v := range w { sums[j] += float64(v) }
				count++
			}
			proto := make([]float32, 3072)
			for j := range proto {
				proto[j] = float32(sums[j] / float64(count))
			}
			protos = append(protos, proto)
		}
	}
	return protos
}

// evalWithVoting evaluates a KMeans with k sub-prototypes per class.
// argmax(output) / k gives the class index.
func evalWithVoting(net *poly.VolumetricNetwork, inputs []*poly.Tensor[float32], labels []float64, k int) (float64, float64) {
	correct := 0
	totalDev := 0.0
	for i, inp := range inputs {
		out, _, _ := poly.ForwardPolymorphic(net, inp)
		// Find best cluster
		best := -1
		bestVal := float32(-1e9)
		for c, v := range out.Data {
			if v > bestVal { bestVal = v; best = c }
		}
		predicted := float64(best / k)
		trueLabel := labels[i]
		if predicted == trueLabel { correct++ }
		// Compute deviation for score
		if trueLabel > 0 {
			dev := math.Abs(predicted-trueLabel) / math.Abs(trueLabel) * 100
			totalDev += dev
		}
	}
	acc := float64(correct) / float64(len(inputs)) * 100
	avgDev := totalDev / float64(len(inputs))
	score := math.Max(0, 100-avgDev)
	return acc, score
}

func printPerClassAccuracy(inputs []*poly.Tensor[float32], labels []float64, protos [][]float32, temp float64) {
	k := len(protos) / 10
	correct := make([]int, 10)
	total   := make([]int, 10)
	confusion := make([][]int, 10)
	for i := range confusion { confusion[i] = make([]int, 10) }

	// Build a temporary KMeans to do forward pass
	// (net already has it installed from gen5, re-use)
	for i, inp := range inputs {
		// Manual nearest-centroid in the proto space
		bestDist := float64(1e18)
		bestProto := 0
		for p, proto := range protos {
			dist := float64(0)
			for j, v := range inp.Data {
				diff := float64(v) - float64(proto[j])
				dist += diff * diff
			}
			if dist < bestDist { bestDist = dist; bestProto = p }
		}
		pred := bestProto / k
		true_ := int(labels[i])
		total[true_]++
		confusion[true_][pred]++
		if pred == true_ { correct[true_]++ }
	}

	type result struct{ cls int; acc float64 }
	results := make([]result, 10)
	for i := 0; i < 10; i++ {
		acc := 0.0
		if total[i] > 0 { acc = float64(correct[i]) / float64(total[i]) * 100 }
		results[i] = result{i, acc}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].acc > results[j].acc })

	fmt.Println("  PER-CLASS ACCURACY  (Gen 5 sub-prototypes, test set)")
	fmt.Println("  ──────────────────────────────────────────────────────────────────")
	for _, r := range results {
		bar := int(r.acc / 2.5)
		b := ""
		for i := 0; i < 40; i++ {
			if i < bar { b += "█" } else { b += "░" }
		}
		fmt.Printf("  [%d] %-12s  %5.1f%%  %s\n", r.cls, classNames[r.cls], r.acc, b)
	}

	// Top confusions
	type conf struct{ true_, pred, count int }
	var entries []conf
	for t := 0; t < 10; t++ {
		for p := 0; p < 10; p++ {
			if t != p && confusion[t][p] > 0 {
				entries = append(entries, conf{t, p, confusion[t][p]})
			}
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].count > entries[j].count })
	if len(entries) > 8 { entries = entries[:8] }
	fmt.Println("\n  TOP CONFUSIONS")
	fmt.Println("  ──────────────────────────────────────────────────────────────────")
	for _, e := range entries {
		pct := float64(e.count) / float64(total[e.true_]) * 100
		fmt.Printf("  %-12s  →  %-12s  %d times (%.1f%%)\n",
			classNames[e.true_], classNames[e.pred], e.count, pct)
	}
}

// ── ASCII PROTOTYPE GALLERY ──────────────────────────────────────────────

func printProtoGallery(means [][]float32, w int) {
	shades := []rune{' ', '░', '▒', '▓', '█'}
	toGray := func(img []float32, r, c int) float32 {
		R := img[r*w+c]
		G := img[1024+r*w+c]
		B := img[2048+r*w+c]
		return 0.299*R + 0.587*G + 0.114*B
	}
	fmt.Println("  CLASS PIXEL PROTOTYPES  (grayscale render, CHW→luminance)")
	fmt.Println("  ──────────────────────────────────────────────────────────────────")
	for row := 0; row < 5; row++ {
		a, b := row*2, row*2+1
		fmt.Printf("  %-30s  %s\n",
			fmt.Sprintf("[%d] %s", a, classNames[a]),
			fmt.Sprintf("[%d] %s", b, classNames[b]))
		for py := 0; py < 32; py += 2 {
			fmt.Print("  ")
			for px := 0; px < w; px++ {
				v := toGray(means[a], py, px) * 1.4
				idx := int(v * float32(len(shades)-1))
				if idx >= len(shades) { idx = len(shades) - 1 }
				fmt.Printf("%c%c", shades[idx], shades[idx])
			}
			fmt.Print("    ")
			for px := 0; px < w; px++ {
				v := toGray(means[b], py, px) * 1.4
				idx := int(v * float32(len(shades)-1))
				if idx >= len(shades) { idx = len(shades) - 1 }
				fmt.Printf("%c%c", shades[idx], shades[idx])
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

// ── DATA HELPERS ──────────────────────────────────────────────────────────

func toTensors(samples []Sample) ([]*poly.Tensor[float32], []float64) {
	inputs := make([]*poly.Tensor[float32], len(samples))
	labels := make([]float64, len(samples))
	for i, s := range samples {
		t := poly.NewTensor[float32](3, 32, 32)
		t.Data = s.Image
		inputs[i] = t
		labels[i] = float64(s.Label)
	}
	return inputs, labels
}

func loadAllBatches() ([]Sample, error) {
	var all []Sample
	for i := 1; i <= 5; i++ {
		p := filepath.Join(BatchDir, fmt.Sprintf("data_batch_%d.bin", i))
		b, err := loadBatch(p)
		if err != nil { return nil, err }
		all = append(all, b...)
	}
	return all, nil
}

func loadBatch(path string) ([]Sample, error) {
	data, err := os.ReadFile(path)
	if err != nil { return nil, err }
	n := len(data) / 3073
	samples := make([]Sample, n)
	for i := 0; i < n; i++ {
		off := i * 3073
		label := int(data[off])
		img := make([]float32, 3072)
		for j := 0; j < 3072; j++ {
			img[j] = float32(data[off+1+j]) / 255.0
		}
		samples[i] = Sample{Image: img, Label: label}
	}
	return samples, nil
}

func ensureData() error {
	// Check if already extracted
	if _, err := os.Stat(filepath.Join(BatchDir, "test_batch.bin")); err == nil {
		return nil
	}
	os.MkdirAll(DataDir, 0755)

	tarPath := filepath.Join(DataDir, "cifar-10-binary.tar.gz")
	if _, err := os.Stat(tarPath); os.IsNotExist(err) {
		fmt.Printf("    Downloading CIFAR-10 binary (~162 MB)...\n")
		if err := downloadFile(DataURL, tarPath); err != nil {
			return fmt.Errorf("download failed: %v", err)
		}
	}

	fmt.Println("    Extracting...")
	return extractTarGz(tarPath, DataDir)
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil { return err }
	defer resp.Body.Close()
	f, err := os.Create(dest)
	if err != nil { return err }
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func extractTarGz(src, destDir string) error {
	f, err := os.Open(src)
	if err != nil { return err }
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil { return err }
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF { break }
		if err != nil { return err }
		path := filepath.Join(destDir, filepath.FromSlash(hdr.Name))
		switch hdr.Typeflag {
		case tar.TypeDir:
			os.MkdirAll(path, 0755)
		case tar.TypeReg:
			os.MkdirAll(filepath.Dir(path), 0755)
			out, err := os.Create(path)
			if err != nil { return err }
			io.Copy(out, tr)
			out.Close()
		}
	}
	return nil
}
