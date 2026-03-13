package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
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
	fmt.Println("🔢 Porting MNIST Demo to Poly Architecture...")

	// 1. Setup Data
	fmt.Println("[*] Ensuring MNIST data is available...")
	if err := ensureData(); err != nil {
		fmt.Printf("[!] Data error: %v\n", err)
		return
	}

	trainData, err := loadMNIST(filepath.Join(DataDir, MnistTrainImagesFile), filepath.Join(DataDir, MnistTrainLabelsFile), 500)
	if err != nil {
		panic(err)
	}
	testData, err := loadMNIST(filepath.Join(DataDir, MnistTestImagesFile), filepath.Join(DataDir, MnistTestLabelsFile), 100)
	if err != nil {
		panic(err)
	}

	// 2. Define Network via JSON
	// We'll use a SEQUENTIAL layer to wrap our CNN architecture
	configJSON := []byte(`{
		"id": "mnist_poly_cnn",
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

	fmt.Println("[*] Building network from JSON...")
	net, err := poly.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}

	// 3. Prepare Training Batches
	fmt.Println("[*] Preparing training batches...")
	batches := make([]poly.TrainingBatch[float32], len(trainData))
	for i, s := range trainData {
		input := poly.NewTensor[float32](1, 28, 28)
		input.Data = s.Image
		
		target := poly.NewTensor[float32](10)
		target.Data[s.Label] = 1.0 // One-hot
		
		batches[i] = poly.TrainingBatch[float32]{
			Input:  input,
			Target: target,
		}
	}

	// 4. Run Training
	fmt.Println("[*] Starting Poly Engine Training...")
	config := &poly.TrainingConfig{
		Epochs:       Epochs,
		LearningRate: 0.01,
		LossType:     "mse",
		Verbose:      true,
	}

	result, err := poly.Train(net, batches, config)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n[+] Training Complete! Final Loss: %.6f | Time: %v\n", result.FinalLoss, result.TotalTime)

	// 5. Evaluation
	fmt.Println("\n[*] Running Evaluation...")
	correct := 0
	for _, s := range testData {
		input := poly.NewTensor[float32](1, 28, 28)
		input.Data = s.Image
		
		// Use manual sequence for evaluation
		curr := input
		l := &net.Layers[0] // Our sequential container
		_, output := poly.DispatchLayer(l, curr, nil)
		
		pred := 0
		maxVal := output.Data[0]
		for j := 1; j < 10; j++ {
			if output.Data[j] > maxVal {
				maxVal = output.Data[j]
				pred = j
			}
		}
		if pred == s.Label {
			correct++
		}
	}
	fmt.Printf("[+] Accuracy: %.2f%%\n", float64(correct)/float64(len(testData))*100.0)
}

// MNIST Data Helpers (Ported from TVA demo)

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
			if err := downloadAndExtract(url, path); err != nil {
				return err
			}
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
