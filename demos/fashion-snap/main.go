package main

import (
	"compress/gzip"
	"encoding/binary"
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
	DataDir = "data"
)

var classNames = [10]string{
	"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
}

var dataFiles = map[string]string{
	"train-images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
	"train-labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
	"test-images":  "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
	"test-labels":  "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

type Sample struct {
	Image []float32
	Label int
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║        FASHION-MNIST  ·  ZERO BACKPROP SNAP CLASSIFIER      ║")
	fmt.Println("║  60,000 clothing images  ·  10 classes  ·  3 meta-decisions ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("A fully-trained CNN on this dataset takes hours and reaches ~93%.")
	fmt.Println("We will do it in milliseconds using only heuristic meta-cognition.")
	fmt.Println()

	// ── 1. DATA ──
	if err := ensureData(); err != nil {
		fmt.Printf("[!] Data error: %v\n", err)
		return
	}

	fmt.Println("[*] Loading dataset...")
	allTrain, err := loadIDX(filepath.Join(DataDir, "train-images"), filepath.Join(DataDir, "train-labels"), 0)
	if err != nil {
		panic(err)
	}
	allTest, err := loadIDX(filepath.Join(DataDir, "test-images"), filepath.Join(DataDir, "test-labels"), 0)
	if err != nil {
		panic(err)
	}

	splitIdx := int(float64(len(allTrain)) * 0.8)
	protoData := allTrain[:splitIdx]
	valData   := allTrain[splitIdx:]
	fmt.Printf("[*] %d train  →  %d prototype  |  %d validation  |  %d test\n\n",
		len(allTrain), len(protoData), len(valData), len(allTest))

	// ── 2. BUILD NETWORK (random weights — untrained) ──
	net, err := poly.BuildNetworkFromJSON([]byte(`{
		"id": "fashion_snap",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "sequential",
			"sequential_layers": [
				{
					"type": "cnn2", "activation": "relu",
					"input_height": 28, "input_width": 28, "input_channels": 1,
					"filters": 16, "kernel_size": 3, "stride": 1, "padding": 0,
					"output_height": 26, "output_width": 26
				},
				{
					"type": "cnn2", "activation": "relu",
					"input_height": 26, "input_width": 26, "input_channels": 16,
					"filters": 32, "kernel_size": 3, "stride": 2, "padding": 0,
					"output_height": 12, "output_width": 12
				},
				{
					"type": "dense", "activation": "relu",
					"input_height": 4608, "output_height": 128
				},
				{
					"type": "dense", "activation": "linear",
					"input_height": 128, "output_height": 10
				}
			]
		}]
	}`))
	if err != nil {
		panic(err)
	}

	// Baseline on 200 samples
	baseInputs, baseLabels := toTensors(allTest[:200])
	baseMet, _ := poly.EvaluateNetworkPolymorphic(net, baseInputs, baseLabels)
	fmt.Printf("Baseline (random CNN weights):  accuracy=%.1f%%  score=%.2f/100\n\n", baseMet.Accuracy, baseMet.Score)

	// ── 3. META-COGNITION LOOP ──
	poly.WrapWithMetacognition(net, []poly.MetaRule{})

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  AUTONOMOUS EVOLUTION  —  zero backprop, heuristic decisions only")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// ── GEN 1: compute class pixel prototypes ──
	fmt.Printf("\n[GEN 1] Hypothesis: clothing classes have distinct average silhouettes.\n")
	fmt.Printf("        Action: average %d images per class into pixel prototypes.\n\n", len(protoData)/10)

	t0 := time.Now()
	sums   := make([][]float64, 10)
	counts := make([]int, 10)
	for i := range sums {
		sums[i] = make([]float64, 784)
	}
	for _, s := range protoData {
		for j, v := range s.Image {
			sums[s.Label][j] += float64(v)
		}
		counts[s.Label]++
	}
	classMeans := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		classMeans[i] = make([]float32, 784)
		if counts[i] > 0 {
			for j := range sums[i] {
				classMeans[i][j] = float32(sums[i][j] / float64(counts[i]))
			}
		}
	}
	gen1Time := time.Since(t0)
	fmt.Printf("        Done in %v\n\n", gen1Time)

	// Show prototype gallery
	printPrototypeGallery(classMeans)

	// ── GEN 2: bypass CNN, install KMeans on raw pixels ──
	fmt.Printf("[GEN 2] Hypothesis: CNN weights are noise — they contribute nothing.\n")
	fmt.Printf("        Action: disable all CNN/Dense layers, install KMeans(784→10)\n")
	fmt.Printf("                seeded with class prototype centroids.\n\n")

	t0 = time.Now()
	l   := &net.Layers[0]
	obs := l.MetaObservedLayer
	for i := 0; i < len(obs.SequentialLayers)-1; i++ {
		obs.SequentialLayers[i].IsDisabled = true
	}
	final := &obs.SequentialLayers[len(obs.SequentialLayers)-1]
	final.Type              = poly.LayerKMeans
	final.NumClusters       = 10
	final.InputHeight       = 784
	final.OutputHeight      = 10
	final.KMeansTemperature = 1.0
	final.WeightStore       = poly.NewWeightStore(10 * 784)
	for i := 0; i < 10; i++ {
		copy(final.WeightStore.Master[i*784:], classMeans[i])
	}
	gen2Time := time.Since(t0)
	fmt.Printf("        Done in %v\n\n", gen2Time)

	// ── GEN 3: tune temperature ──
	fmt.Printf("[GEN 3] Hypothesis: soft assignments are too uncertain — sharpen them.\n")
	fmt.Printf("        Action: lower KMeans temperature 1.0 → 0.3\n\n")
	final.KMeansTemperature = 0.3

	totalSnapTime := gen1Time + gen2Time

	// ── 4. EVALUATE ──
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  RESULTS")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	valInputs,  valLabels  := toTensors(valData)
	testInputs, testLabels := toTensors(allTest)

	fmt.Printf("\n[VAL]  Evaluating %d validation samples (20%% held-out train)...\n", len(valData))
	t0 = time.Now()
	mVal, _ := poly.EvaluateNetworkPolymorphic(net, valInputs, valLabels)
	valInfTime := time.Since(t0)

	fmt.Printf("[TEST] Evaluating %d official test samples...\n\n", len(allTest))
	t0 = time.Now()
	mTest, _ := poly.EvaluateNetworkPolymorphic(net, testInputs, testLabels)
	testInfTime := time.Since(t0)

	// Per-class breakdown
	printPerClassAccuracy(net, allTest)

	// Summary box
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               FASHION-MNIST SNAP CLASSIFIER RESULTS             ║")
	fmt.Println("╠═══════════════════╦════════════════╦══════════════════════════════╣")
	fmt.Printf( "║ %-17s ║  Accuracy %5.1f%% ║  Quality Score  %6.2f/100  ║\n",
		fmt.Sprintf("Validation (%dk)", len(valData)/1000), mVal.Accuracy, mVal.Score)
	fmt.Printf( "║ %-17s ║  Accuracy %5.1f%% ║  Quality Score  %6.2f/100  ║\n",
		fmt.Sprintf("Test Set (%dk)", len(allTest)/1000), mTest.Accuracy, mTest.Score)
	fmt.Println("╠═══════════════════╩════════════════╩══════════════════════════════╣")
	fmt.Printf( "║  Weight snap time : %-10v                                   ║\n", totalSnapTime)
	fmt.Printf( "║  Val  inference   : %-10v  (%.1f µs/sample)                 ║\n",
		valInfTime, float64(valInfTime.Microseconds())/float64(len(valData)))
	fmt.Printf( "║  Test inference   : %-10v  (%.1f µs/sample)                 ║\n",
		testInfTime, float64(testInfTime.Microseconds())/float64(len(allTest)))
	fmt.Println("╠══════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Backprop epochs  : ZERO                                         ║")
	fmt.Println("║  Optimizer        : NONE                                         ║")
	fmt.Println("║  Loss function    : NONE                                         ║")
	fmt.Println("║  Meta-decisions   : 3                                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// Persist
	fmt.Println("\n[*] Saving snapped model...")
	data, err := poly.SerializeNetwork(net)
	if err != nil {
		fmt.Printf("[!] Serialize error: %v\n", err)
		return
	}
	if err := os.WriteFile("fashion_snap.json", data, 0644); err != nil {
		fmt.Printf("[!] Write error: %v\n", err)
		return
	}
	fmt.Printf("[*] Saved to fashion_snap.json (%d KB)\n", len(data)/1024)

	netReloaded, err := poly.DeserializeNetwork(data)
	if err != nil {
		fmt.Printf("[!] Deserialize error: %v\n", err)
		return
	}
	mReload, _ := poly.EvaluateNetworkPolymorphic(netReloaded, testInputs[:100], testLabels[:100])
	fmt.Printf("[*] Reload check: accuracy=%.1f%% — ", mReload.Accuracy)
	if math.Abs(mReload.Accuracy-mTest.Accuracy) < 2.0 {
		fmt.Println("✅ PASS")
	} else {
		fmt.Println("❌ MISMATCH")
	}
}

// ── printPrototypeGallery renders each class mean image as ASCII art ──
func printPrototypeGallery(means [][]float32) {
	// Use block characters: intensity 0→1 mapped to 5 levels
	shades := []rune{' ', '░', '▒', '▓', '█'}

	fmt.Println("  CLASS PIXEL PROTOTYPES  (what the meta-cognition is working with)")
	fmt.Println("  ─────────────────────────────────────────────────────────────────")

	// Print two classes side by side, 5 columns of 2
	for row := 0; row < 5; row++ {
		a, b := row*2, row*2+1

		// Header
		nameA := fmt.Sprintf("[%d] %s", a, classNames[a])
		nameB := fmt.Sprintf("[%d] %s", b, classNames[b])
		fmt.Printf("  %-32s  %s\n", nameA, nameB)

		// 28 rows of pixels — but print every other row for compactness (14 lines)
		for py := 0; py < 28; py += 2 {
			fmt.Print("  ")
			for px := 0; px < 28; px++ {
				v := means[a][py*28+px]
				idx := int(v * float32(len(shades)-1) * 1.5)
				if idx >= len(shades) {
					idx = len(shades) - 1
				}
				fmt.Printf("%c%c", shades[idx], shades[idx])
			}
			fmt.Print("    ")
			for px := 0; px < 28; px++ {
				v := means[b][py*28+px]
				idx := int(v * float32(len(shades)-1) * 1.5)
				if idx >= len(shades) {
					idx = len(shades) - 1
				}
				fmt.Printf("%c%c", shades[idx], shades[idx])
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

// ── printPerClassAccuracy runs inference and shows per-class results ──
func printPerClassAccuracy(net *poly.VolumetricNetwork, samples []Sample) {
	correct := make([]int, 10)
	total   := make([]int, 10)

	// Track confusion: confusion[true][pred]++
	confusion := make([][]int, 10)
	for i := range confusion {
		confusion[i] = make([]int, 10)
	}

	for _, s := range samples {
		inp := poly.NewTensor[float32](1, 28, 28)
		inp.Data = s.Image
		out, _, _ := poly.ForwardPolymorphic(net, inp)

		pred := 0
		best := float32(-1e9)
		for k, v := range out.Data {
			if v > best {
				best = v
				pred = k
			}
		}
		total[s.Label]++
		confusion[s.Label][pred]++
		if pred == s.Label {
			correct[s.Label]++
		}
	}

	// Sort classes by accuracy descending
	type classResult struct {
		idx  int
		acc  float64
		n    int
	}
	results := make([]classResult, 10)
	for i := 0; i < 10; i++ {
		acc := 0.0
		if total[i] > 0 {
			acc = float64(correct[i]) / float64(total[i]) * 100
		}
		results[i] = classResult{i, acc, total[i]}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].acc > results[j].acc
	})

	fmt.Println("  PER-CLASS ACCURACY (test set, sorted best → worst)")
	fmt.Println("  ──────────────────────────────────────────────────────────────")
	for _, r := range results {
		bar := int(r.acc / 2.5)
		barStr := ""
		for i := 0; i < 40; i++ {
			if i < bar {
				barStr += "█"
			} else {
				barStr += "░"
			}
		}
		fmt.Printf("  [%d] %-14s  %5.1f%%  %s\n", r.idx, classNames[r.idx], r.acc, barStr)
	}

	// Show top confusions
	fmt.Println("\n  TOP CONFUSIONS  (what it mistakes for what)")
	fmt.Println("  ──────────────────────────────────────────────────────────────")
	type confusion_entry struct {
		true_, pred int
		count       int
	}
	var entries []confusion_entry
	for t := 0; t < 10; t++ {
		for p := 0; p < 10; p++ {
			if t != p && confusion[t][p] > 0 {
				entries = append(entries, confusion_entry{t, p, confusion[t][p]})
			}
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].count > entries[j].count
	})
	limit := 8
	if len(entries) < limit {
		limit = len(entries)
	}
	for _, e := range entries[:limit] {
		pct := float64(e.count) / float64(total[e.true_]) * 100
		fmt.Printf("  [%d] %-14s  mistaken for  [%d] %-14s  — %d times (%.1f%%)\n",
			e.true_, classNames[e.true_], e.pred, classNames[e.pred], e.count, pct)
	}
	fmt.Println()
}

// ── Data helpers ──

func toTensors(samples []Sample) ([]*poly.Tensor[float32], []float64) {
	inputs := make([]*poly.Tensor[float32], len(samples))
	labels := make([]float64, len(samples))
	for i, s := range samples {
		t := poly.NewTensor[float32](1, 28, 28)
		t.Data = s.Image
		inputs[i] = t
		labels[i] = float64(s.Label)
	}
	return inputs, labels
}

func ensureData() error {
	if err := os.MkdirAll(DataDir, 0755); err != nil {
		return err
	}
	dest := map[string]string{
		"train-images": filepath.Join(DataDir, "train-images"),
		"train-labels": filepath.Join(DataDir, "train-labels"),
		"test-images":  filepath.Join(DataDir, "test-images"),
		"test-labels":  filepath.Join(DataDir, "test-labels"),
	}
	for key, url := range dataFiles {
		path := dest[key]
		if _, err := os.Stat(path); os.IsNotExist(err) {
			fmt.Printf("    Downloading %s...\n", key)
			if err := downloadGZ(url, path); err != nil {
				return fmt.Errorf("%s: %v", key, err)
			}
		}
	}
	return nil
}

func downloadGZ(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gz.Close()
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, gz)
	return err
}

func loadIDX(imageFile, labelFile string, maxCount int) ([]Sample, error) {
	imgF, err := os.Open(imageFile)
	if err != nil {
		return nil, err
	}
	defer imgF.Close()

	var magic, n, rows, cols int32
	binary.Read(imgF, binary.BigEndian, &magic)
	binary.Read(imgF, binary.BigEndian, &n)
	binary.Read(imgF, binary.BigEndian, &rows)
	binary.Read(imgF, binary.BigEndian, &cols)

	lblF, err := os.Open(labelFile)
	if err != nil {
		return nil, err
	}
	defer lblF.Close()
	var lMagic, lN int32
	binary.Read(lblF, binary.BigEndian, &lMagic)
	binary.Read(lblF, binary.BigEndian, &lN)

	count := int(n)
	if maxCount > 0 && maxCount < count {
		count = maxCount
	}

	samples := make([]Sample, count)
	imgSize := int(rows * cols)
	buf  := make([]byte, imgSize)
	lBuf := make([]byte, 1)
	for i := 0; i < count; i++ {
		imgF.Read(buf)
		lblF.Read(lBuf)
		img := make([]float32, imgSize)
		for j := range img {
			img[j] = float32(buf[j]) / 255.0
		}
		samples[i] = Sample{Image: img, Label: int(lBuf[0])}
	}
	return samples, nil
}
