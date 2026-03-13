package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	InputDim        = 2   // 2D functions (x, y)
	IntermediateDim = 128 // SwiGLU intermediate dimension
	NumFunctions    = 4   // Different function types
	SamplesPerFunc  = 250
	Epochs          = 50
	LearningRate    = 0.01
)

const (
	FuncSinCos = iota
	FuncPolynomial
	FuncGaussian
	FuncTanhProd
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("🚀 SwiGLU Function Approximation Demo (Poly Core)")

	// 1. Generate dataset
	fmt.Println("[*] Generating function data...")
	trainInputs, trainTargets := generateFunctionDataset()

	// 2. Define Network via JSON
	configJSON := []byte(fmt.Sprintf(`{
		"id": "swiglu_poly_demo",
		"depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [
			{
				"z": 0, "y": 0, "x": 0, "l": 0,
				"type": "sequential",
				"sequential_layers": [
					{
						"type": "dense", "activation": "tanh",
						"input_height": %d, "output_height": 64
					},
					{
						"type": "swiglu", "activation": "linear",
						"input_height": 64, "output_height": %d
					},
					{
						"type": "swiglu", "activation": "linear",
						"input_height": 64, "output_height": %d
					},
					{
						"type": "dense", "activation": "tanh",
						"input_height": 64, "output_height": 1
					}
				]
			}
		]
	}`, InputDim, IntermediateDim, IntermediateDim))

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
		
		target := poly.NewTensor[float32](1, 1)
		target.Data[0] = trainTargets[i]
		
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

	// 5. Evaluation Examples
	fmt.Println("\n[*] Example Approximations:")
	showExampleApproximations(net)
}

func generateFunctionDataset() ([][]float32, []float32) {
	totalSamples := NumFunctions * SamplesPerFunc
	data := make([][]float32, totalSamples)
	targets := make([]float32, totalSamples)

	idx := 0
	for funcType := 0; funcType < NumFunctions; funcType++ {
		for sample := 0; sample < SamplesPerFunc; sample++ {
			x := rand.Float64()*4 - 2
			y := rand.Float64()*4 - 2

			input := []float32{float32(x), float32(y)}
			target := evaluateFunction(funcType, x, y)

			data[idx] = input
			targets[idx] = float32(target)
			idx++
		}
	}
	return data, targets
}

func evaluateFunction(funcType int, x, y float64) float64 {
	switch funcType {
	case FuncSinCos:
		return math.Sin(x) * math.Cos(y)
	case FuncPolynomial:
		return x*x + y*y - x*y
	case FuncGaussian:
		return math.Exp(-x*x-y*y) * math.Sin(x+y)
	case FuncTanhProd:
		return math.Tanh(x*y) + x/2
	}
	return 0
}

func showExampleApproximations(net *poly.VolumetricNetwork) {
	funcNames := []string{"sin(x)*cos(y)", "x²+y²-xy", "exp(-x²-y²)*sin(x+y)", "tanh(xy)+x/2"}

	for f := 0; f < NumFunctions; f++ {
		x, y := 1.0, 0.5
		inputTensor := poly.NewTensor[float32](1, InputDim)
		inputTensor.Data = []float32{float32(x), float32(y)}
		
		target := evaluateFunction(f, x, y)
		
		// Forward pass
		l := &net.Layers[0]
		_, output := poly.DispatchLayer(l, inputTensor, nil)
		
		pred := output.Data[0]
		errVal := math.Abs(float64(pred) - target)

		fmt.Printf("  %s at (%.1f, %.1f):\n", funcNames[f], x, y)
		fmt.Printf("    Target: %.6f, Predicted: %.6f, Error: %.6f\n",
			target, pred, errVal)
	}
}
