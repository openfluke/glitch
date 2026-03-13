package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	SampleRate      = 128 // Samples per waveform
	NumClasses      = 4   // Sine, Square, Sawtooth, Triangle
	SamplesPerClass = 250 // Training samples per class
	Epochs          = 50
	LearningRate    = 0.05
)

const (
	WaveformSine = iota
	WaveformSquare
	WaveformSawtooth
	WaveformTriangle
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("🎵 Conv1D Audio Waveform Classification Demo (Poly Core)")

	// 1. Generate dataset
	fmt.Println("[*] Generating synthetic audio waveforms...")
	trainInputs, trainLabels := generateWaveformDataset()
	fmt.Printf("    Generated %d waveforms (%d samples each)\n", len(trainInputs), SampleRate)

	// 2. Define Network via JSON
	// Note: output_height for CNN1 is the output sequence length
	// Layer 1: Input 128 -> Dense(128) -> 128
	// Layer 2: CNN1(input_len=128, k=5, s=2, p=0) -> output_len = (128-5)/2 + 1 = 61 + 1 = 62
	// Layer 3: CNN1(input_len=62, k=3, s=2, p=0) -> output_len = (62-3)/2 + 1 = 29 + 1 = 30
	// Layer 4: Dense(30*64=1920, 4)
	configJSON := []byte(fmt.Sprintf(`{
		"id": "conv1d_audio_poly",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [
			{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential",
				"sequential_layers": [
					{
						"type": "dense", "activation": "tanh",
						"input_height": %d, "output_height": %d
					},
					{
						"type": "cnn1", "activation": "relu",
						"input_channels": 1, "filters": 32, "kernel_size": 5, "stride": 2, "padding": 0,
						"input_height": %d, "output_height": 62
					},
					{
						"type": "cnn1", "activation": "relu",
						"input_channels": 32, "filters": 64, "kernel_size": 3, "stride": 2, "padding": 0,
						"input_height": 62, "output_height": 30
					},
					{
						"type": "dense", "activation": "sigmoid",
						"input_height": 1920, "output_height": %d
					}
				]
			}
		]
	}`, SampleRate, SampleRate, SampleRate, NumClasses))

	fmt.Println("[*] Building network from JSON...")
	net, err := poly.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}

	// 3. Prepare Training Batches
	batches := make([]poly.TrainingBatch[float32], len(trainInputs))
	for i := range trainInputs {
		input := poly.NewTensor[float32](1, len(trainInputs[i]))
		input.Data = trainInputs[i]
		
		target := poly.NewTensor[float32](1, NumClasses)
		// One-hot
		target.Data[trainLabels[i]] = 1.0
		
		batches[i] = poly.TrainingBatch[float32]{
			Input:  input,
			Target: target,
		}
	}

	// 4. Run Training
	fmt.Println("[*] Starting Poly Engine Training...")
	config := &poly.TrainingConfig{
		Epochs:       Epochs,
		LearningRate: LearningRate,
		LossType:     "mse",
		Verbose:      true,
	}

	result, err := poly.Train(net, batches, config)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n[+] Training Complete! Final Loss: %.6f\n", result.FinalLoss)

	// 5. Evaluation
	fmt.Println("\n[*] Running Evaluation...")
	accuracy := evaluateAccuracy(net, trainInputs, trainLabels)
	fmt.Printf("[+] Training Accuracy: %.2f%%\n", accuracy*100)
}

func generateWaveformDataset() ([][]float32, []int) {
	totalSamples := NumClasses * SamplesPerClass
	data := make([][]float32, totalSamples)
	labels := make([]int, totalSamples)

	idx := 0
	for class := 0; class < NumClasses; class++ {
		for sample := 0; sample < SamplesPerClass; sample++ {
			waveform := generateWaveform(class)
			data[idx] = waveform
			labels[idx] = class
			idx++
		}
	}
	return data, labels
}

func generateWaveform(waveformType int) []float32 {
	waveform := make([]float32, SampleRate)
	frequency := 1.0 + rand.Float64()*4.0
	noiseLevel := 0.1

	for i := 0; i < SampleRate; i++ {
		t := float64(i) / float64(SampleRate)
		phase := 2.0 * math.Pi * frequency * t

		var value float64
		switch waveformType {
		case WaveformSine:
			value = math.Sin(phase)
		case WaveformSquare:
			if math.Sin(phase) >= 0 { value = 1.0 } else { value = -1.0 }
		case WaveformSawtooth:
			value = 2.0 * (frequency*t - math.Floor(frequency*t+0.5))
		case WaveformTriangle:
			sawValue := 2.0 * (frequency*t - math.Floor(frequency*t+0.5))
			value = 2.0*math.Abs(sawValue) - 1.0
		}
		noise := (rand.Float64()*2 - 1) * noiseLevel
		waveform[i] = float32(value + noise)
	}
	return waveform
}

func evaluateAccuracy(net *poly.VolumetricNetwork, inputs [][]float32, labels []int) float64 {
	correct := 0
	for i, inputData := range inputs {
		inputTensor := poly.NewTensor[float32](1, len(inputData))
		inputTensor.Data = inputData
		
		_, output := poly.DispatchLayer(&net.Layers[0], inputTensor, nil)
		
		if argmax(output.Data) == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

func argmax(vec []float32) int {
	maxIdx := 0
	maxVal := vec[0]
	for i, val := range vec {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}
