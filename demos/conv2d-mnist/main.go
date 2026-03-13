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

	"github.com/openfluke/loom/poly"
)

const (
	MnistTrainImagesFile = "train-images-idx3-ubyte"
	MnistTrainLabelsFile = "train-labels-idx1-ubyte"
	MnistTestImagesFile  = "t10k-images-idx3-ubyte"
	MnistTestLabelsFile  = "t10k-labels-idx1-ubyte"
	DataDir              = "data"
	Epochs               = 3
)

type MNISTSample struct {
	Image []float32
	Label int
}

func main() {
	fmt.Println("🔢 Expanded MNIST Demo for Poly Core...")

	// 1. Setup Data
	fmt.Println("[*] Ensuring MNIST data is available...")
	if err := ensureData(); err != nil {
		fmt.Printf("[!] Data error: %v\n", err)
		return
	}

	trainData, err := loadMNIST(filepath.Join(DataDir, MnistTrainImagesFile), filepath.Join(DataDir, MnistTrainLabelsFile), 1000)
	if err != nil { panic(err) }
	testData, err := loadMNIST(filepath.Join(DataDir, MnistTestImagesFile), filepath.Join(DataDir, MnistTestLabelsFile), 100)
	if err != nil { panic(err) }

	// 2. Define Network Configuration (M-POLY-VTD style)
	configJSON := []byte(`{
		"id": "mnist_expanded",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [
			{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential",
				"sequential_layers": [
					{
						"type": "cnn2", "activation": "relu",
						"input_height": 28, "input_width": 28, "input_channels": 1,
						"filters": 8, "kernel_size": 3, "stride": 1, "padding": 0,
						"output_height": 26, "output_width": 26
					},
					{
						"type": "cnn2", "activation": "relu",
						"input_height": 26, "input_width": 26, "input_channels": 8,
						"filters": 16, "kernel_size": 3, "stride": 2, "padding": 0,
						"output_height": 12, "output_width": 12
					},
					{
						"type": "dense", "activation": "relu",
						"input_height": 2304, "output_height": 64
					},
					{
						"type": "dense", "activation": "linear",
						"input_height": 64, "output_height": 10
					}
				]
			}
		]
	}`)

	// Prepare data for poly.Train
	batches := make([]poly.TrainingBatch[float32], len(trainData))
	for i, s := range trainData {
		input := poly.NewTensor[float32](1, 28, 28)
		input.Data = s.Image
		target := poly.NewTensor[float32](10)
		target.Data[s.Label] = 1.0
		batches[i] = poly.TrainingBatch[float32]{Input: input, Target: target}
	}

	valInputs := make([]*poly.Tensor[float32], len(testData))
	valLabels := make([]float64, len(testData))
	for i, s := range testData {
		input := poly.NewTensor[float32](1, 28, 28)
		input.Data = s.Image
		valInputs[i] = input
		valLabels[i] = float64(s.Label)
	}

	// 3. CPU Training
	fmt.Println("\n=== Starting CPU Training ===")
	net, err := poly.BuildNetworkFromJSON(configJSON)
	if err != nil { panic(err) }

	fmt.Println("\n--- Pre-Training CPU Evaluation ---")
	metricsBefore, _ := poly.EvaluateNetworkPolymorphic(net, valInputs, valLabels)
	metricsBefore.PrintSummary()

	config := &poly.TrainingConfig{
		Epochs:       Epochs,
		LearningRate: 0.01,
		LossType:     "mse",
		Verbose:      true,
	}

	_, err = poly.Train(net, batches, config)
	if err != nil { panic(err) }

	fmt.Println("\n--- Post-Training CPU Evaluation ---")
	metricsAfter, _ := poly.EvaluateNetworkPolymorphic(net, valInputs, valLabels)
	metricsAfter.PrintSummary()

	// 4. Persistence Test
	fmt.Println("\n=== Verifying Model Persistence ===")
	modelData, err := poly.SerializeNetwork(net)
	if err != nil { panic(err) }
	
	netReloaded, err := poly.DeserializeNetwork(modelData)
	if err != nil { panic(err) }
	
	fmt.Println("[*] Running consistency check (Original vs Reloaded)...")
	match := true
	for i := 0; i < 5; i++ {
		out1, _, _ := poly.ForwardPolymorphic(net, valInputs[i])
		out2, _, _ := poly.ForwardPolymorphic(netReloaded, valInputs[i])
		for j := range out1.Data {
			if math.Abs(float64(out1.Data[j]-out2.Data[j])) > 1e-6 {
				match = false
				break
			}
		}
	}
	if match {
		fmt.Println("✅ Consistency PASS: Reloaded model produces identical outputs.")
	} else {
		fmt.Println("❌ Consistency FAIL: Mismatch detected.")
	}

	// 5. GPU Acceleration (Conditional)
	fmt.Println("\n=== GPU Acceleration Check ===")
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("[!] WGPU not available: %v\n", err)
	} else {
		fmt.Println("✅ WGPU Initialized! Commencing GPU Metadata Sync...")
		for i := range net.Layers {
			(&net.Layers[i]).SyncToGPU()
		}
		fmt.Println("[*] Weights mounted to VRAM.")
	}

	// 6. Numerical Type Benchmarking
	fmt.Println("\n=== Benchmarking All Numerical Types (Morphing) ===")
	dtypes := []poly.DType{
		poly.DTypeFloat64, poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16,
		poly.DTypeFP8E4M3, poly.DTypeFP8E5M2,
		poly.DTypeInt64, poly.DTypeInt32, poly.DTypeInt16, poly.DTypeInt8,
		poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8,
		poly.DTypeInt4, poly.DTypeUint4, poly.DTypeFP4,
		poly.DTypeInt2, poly.DTypeUint2, poly.DTypeTernary, poly.DTypeBinary,
	}
	
	fmt.Printf("%-10s | %-12s | %-8s | %-10s\n", "DType", "Memory", "Accuracy", "Score")
	fmt.Println("-----------|--------------|----------|-----------")
	
	for _, dt := range dtypes {
		// Morph all weights to target DType with optimal scaling
		totalBytes := 0
		for i := range net.Layers {
			l := &net.Layers[i]
			
			processLayer := func(lay *poly.VolumetricLayer) {
				if lay.WeightStore == nil { return }
				
				// Calculate optimal scale for this layer and DType
				maxAbs := float32(0)
				sumAbs := float32(0)
				for _, v := range lay.WeightStore.Master {
					a := float32(math.Abs(float64(v)))
					if a > maxAbs { maxAbs = a }
					sumAbs += a
				}
				meanAbs := sumAbs / float32(len(lay.WeightStore.Master))
				
				scale := float32(1.0)
				switch dt {
				case poly.DTypeInt8, poly.DTypeUint8, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
					scale = maxAbs / 127.0
				case poly.DTypeInt16, poly.DTypeUint16:
					scale = maxAbs / 32767.0
				case poly.DTypeInt32, poly.DTypeUint32, poly.DTypeInt64, poly.DTypeUint64:
					scale = maxAbs / 1000000.0 // Preserve high precision
				case poly.DTypeInt4, poly.DTypeUint4, poly.DTypeFP4:
					scale = maxAbs / 7.0
				case poly.DTypeInt2, poly.DTypeUint2:
					scale = maxAbs / 1.0
				case poly.DTypeTernary, poly.DTypeBinary:
					scale = meanAbs
				}
				if scale == 0 { scale = 1.0 }
				
				lay.WeightStore.Scale = scale
				lay.WeightStore.Morph(dt)
				totalBytes += lay.WeightStore.SizeInBytes(dt)
			}

			if l.Type == poly.LayerSequential {
				for j := range l.SequentialLayers {
					processLayer(&l.SequentialLayers[j])
				}
			} else {
				processLayer(l)
			}
		}

		// Update DType in layers to use the morphed version during forward
		for i := range net.Layers {
			l := &net.Layers[i]
			l.DType = dt
			if l.Type == poly.LayerSequential {
				for j := range l.SequentialLayers {
					l.SequentialLayers[j].DType = dt
				}
			}
		}

		m, _ := poly.EvaluateNetworkPolymorphic(net, valInputs, valLabels)
		fmt.Printf("%-10s | %-12s | %7.1f%% | %8.1f\n", 
			dt.String(), formatBytes(int64(totalBytes)), m.Accuracy, m.Score)
	}

	fmt.Println("\n[+] Demo Complete!")
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit { return fmt.Sprintf("%d B", b) }
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// MNIST Helpers
func ensureData() error {
	if _, err := os.Stat(DataDir); os.IsNotExist(err) {
		os.MkdirAll(DataDir, 0755)
	}
	files := map[string]string{
		MnistTrainImagesFile: "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
		MnistTrainLabelsFile: "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
		MnistTestImagesFile:  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
		MnistTestLabelsFile:  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
	}
	for filename, url := range files {
		path := filepath.Join(DataDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			fmt.Printf("    Downloading %s...\n", filename)
			if err := downloadAndExtract(url, path); err != nil { return err }
		}
	}
	return nil
}

func downloadAndExtract(url, destPath string) error {
	resp, err := http.Get(url)
	if err != nil { return err }
	defer resp.Body.Close()
	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil { return err }
	defer gzReader.Close()
	outFile, err := os.Create(destPath)
	if err != nil { return err }
	defer outFile.Close()
	_, err = io.Copy(outFile, gzReader)
	return err
}

func loadMNIST(imageFile, labelFile string, maxCount int) ([]MNISTSample, error) {
	imgF, err := os.Open(imageFile)
	if err != nil { return nil, err }
	defer imgF.Close()
	var magic, numImgs, rows, cols int32
	binary.Read(imgF, binary.BigEndian, &magic)
	binary.Read(imgF, binary.BigEndian, &numImgs)
	binary.Read(imgF, binary.BigEndian, &rows)
	binary.Read(imgF, binary.BigEndian, &cols)
	
	lblF, err := os.Open(labelFile)
	if err != nil { return nil, err }
	defer lblF.Close()
	var lMagic, lNumItems int32
	binary.Read(lblF, binary.BigEndian, &lMagic)
	binary.Read(lblF, binary.BigEndian, &lNumItems)
	
	count := int(numImgs)
	if maxCount > 0 && maxCount < count { count = maxCount }
	
	samples := make([]MNISTSample, count)
	imgSize := int(rows * cols)
	buf := make([]byte, imgSize)
	lBuf := make([]byte, 1)
	for i := 0; i < count; i++ {
		imgF.Read(buf)
		lblF.Read(lBuf)
		imgFloats := make([]float32, imgSize)
		for j := 0; j < imgSize; j++ {
			imgFloats[j] = float32(buf[j]) / 255.0
		}
		samples[i] = MNISTSample{Image: imgFloats, Label: int(lBuf[0])}
	}
	return samples, nil
}
