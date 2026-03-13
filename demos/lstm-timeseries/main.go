package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	WindowSize      = 20 // Look at 20 time steps
	ForecastHorizon = 5  // Predict next 5 steps
	NumTSTypes      = 4  // Seasonal, Trend, MultiFreq, Noisy
	SamplesPerType  = 200
	Epochs          = 8
	LearningRate    = 0.003
	HiddenSize      = 64
)

const (
	TSTypeSeasonal = iota
	TSTypeTrend
	TSTypeMultiFreq
	TSTypeNoisy
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("📈 LSTM Time Series Forecasting Demo (Poly Core)")

	// 1. Generate dataset
	fmt.Println("[*] Generating time series patterns...")
	trainInputs, trainTargets := generateTimeSeriesDataset()
	fmt.Printf("    Generated %d time series (window=%d, forecast=%d)\n",
		len(trainInputs), WindowSize, ForecastHorizon)

	// 2. Define Network via JSON
	// Layer 1: LSTM (seq=20, in=1, hidden=64)
	// Output is (batch, 20, 64) -> Flattened 1280
	// Layer 2: Dense (1280, 5)
	configJSON := []byte(fmt.Sprintf(`{
		"id": "lstm_ts_poly",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [
			{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential",
				"sequential_layers": [
					{
						"type": "lstm", "activation": "tanh",
						"input_height": 1, "output_height": %d, "seq_length": %d
					},
					{
						"type": "dense", "activation": "tanh",
						"input_height": %d, "output_height": %d
					}
				]
			}
		]
	}`, HiddenSize, WindowSize, WindowSize*HiddenSize, ForecastHorizon))

	fmt.Println("[*] Building network from JSON...")
	net, err := poly.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}

	// 3. Prepare Training Batches
	batches := make([]poly.TrainingBatch[float32], len(trainInputs))
	for i := range trainInputs {
		input := poly.NewTensor[float32](1, WindowSize, 1) // Batch=1, Seq=20, Dim=1
		input.Data = trainInputs[i]

		target := poly.NewTensor[float32](1, ForecastHorizon)
		target.Data = trainTargets[i]

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
	fmt.Println("\n[*] Example Forecasts:")
	showExampleForecasts(net)
}

func generateTimeSeriesDataset() ([][]float32, [][]float32) {
	totalSamples := NumTSTypes * SamplesPerType
	data := make([][]float32, totalSamples)
	targets := make([][]float32, totalSamples)

	idx := 0
	for tsType := 0; tsType < NumTSTypes; tsType++ {
		for sample := 0; sample < SamplesPerType; sample++ {
			window, forecast := generateTS(tsType)
			data[idx] = window
			targets[idx] = forecast
			idx++
		}
	}
	return data, targets
}

func generateTS(tsType int) ([]float32, []float32) {
	window := make([]float32, WindowSize)
	forecast := make([]float32, ForecastHorizon)
	totalLength := WindowSize + ForecastHorizon
	fullSeries := make([]float64, totalLength)

	switch tsType {
	case TSTypeSeasonal:
		frequency := rand.Float64()*0.2 + 0.1
		amplitude := rand.Float64()*3 + 1
		phase := rand.Float64() * 2 * math.Pi
		for i := 0; i < totalLength; i++ {
			fullSeries[i] = amplitude * math.Sin(2*math.Pi*frequency*float64(i)+phase)
			fullSeries[i] += (rand.Float64()*2 - 1) * 0.1
		}
	case TSTypeTrend:
		start := rand.Float64()*10 - 5
		linear := rand.Float64()*0.5 - 0.25
		quadratic := rand.Float64()*0.02 - 0.01
		for i := 0; i < totalLength; i++ {
			t := float64(i)
			fullSeries[i] = start + linear*t + quadratic*t*t
			fullSeries[i] += (rand.Float64()*2 - 1) * 0.3
		}
	case TSTypeMultiFreq:
		f1 := rand.Float64()*0.15 + 0.05
		f2 := rand.Float64()*0.3 + 0.15
		a1 := rand.Float64()*2 + 0.5
		a2 := rand.Float64()*1.5 + 0.5
		for i := 0; i < totalLength; i++ {
			t := float64(i)
			fullSeries[i] = a1*math.Sin(2*math.Pi*f1*t) + a2*math.Sin(2*math.Pi*f2*t)
			fullSeries[i] += (rand.Float64()*2 - 1) * 0.15
		}
	case TSTypeNoisy:
		base := rand.Float64()*5 - 2.5
		drift := rand.Float64()*0.1 - 0.05
		for i := 0; i < totalLength; i++ {
			fullSeries[i] = base + float64(i)*drift
			fullSeries[i] += (rand.Float64()*2 - 1) * 2.0
		}
	}

	for i := 0; i < WindowSize; i++ {
		window[i] = float32(fullSeries[i])
	}
	for i := 0; i < ForecastHorizon; i++ {
		forecast[i] = float32(fullSeries[WindowSize+i])
	}
	return window, forecast
}

func showExampleForecasts(net *poly.VolumetricNetwork) {
	tsTypes := []string{"Seasonal", "Trend", "MultiFreq", "Noisy"}
	for ts := 0; ts < NumTSTypes; ts++ {
		window, target := generateTS(ts)
		inputTensor := poly.NewTensor[float32](1, WindowSize, 1)
		inputTensor.Data = window

		_, output := poly.DispatchLayer(&net.Layers[0], inputTensor, nil)

		totalError := 0.0
		for i := 0; i < ForecastHorizon; i++ {
			totalError += math.Abs(float64(output.Data[i] - target[i]))
		}
		avgError := totalError / float64(ForecastHorizon)

		fmt.Printf("  %s:\n", tsTypes[ts])
		fmt.Printf("    Target:    [%.2f %.2f %.2f %.2f %.2f]\n",
			target[0], target[1], target[2], target[3], target[4])
		fmt.Printf("    Predicted: [%.2f %.2f %.2f %.2f %.2f]\n",
			output.Data[0], output.Data[1], output.Data[2], output.Data[3], output.Data[4])
		fmt.Printf("    Avg Error: %.4f\n", avgError)
	}
}
