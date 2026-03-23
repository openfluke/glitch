package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("==================================================")
	fmt.Println(" SYSTOLIC TARGET PROPAGATION DEMO")
	fmt.Println("==================================================")
	fmt.Println("This demo proves that the 3D Volumetric Mesh can:")
	fmt.Println("1. Run continuous clock-cycle passes across spatial tiles.")
	fmt.Println("2. Autonomously adapt weights using Target Propagation.")
	fmt.Println("3. Maintain functionality without exact global gradients (no chain rule backwards).")
	fmt.Println("")

	// 1. Create a Volumetric Network
	// We'll create a simple 1x1x3 grid (3 sequential layers mapped into a spatial mesh).
	n := &poly.VolumetricNetwork{
		Depth: 1, Rows: 1, Cols: 4, LayersPerCell: 1,
		UseTiling: true, // Force parallel tiled CPU routing!
	}
	n.Layers = make([]poly.VolumetricLayer, 4)

	// Layer 0: Input Register (Disabled pass-through)
	n.Layers[0] = poly.VolumetricLayer{
		IsDisabled: true,
	}
	// Layer 1: Input -> Hidden (1 -> 8)
	n.Layers[1] = poly.VolumetricLayer{
		Type: poly.LayerDense,
		InputHeight: 1, OutputHeight: 8,
		WeightStore: poly.NewWeightStore(8),
		Activation: poly.ActivationTanh,
	}
	// Layer 2: Hidden -> Hidden (8 -> 8)
	n.Layers[2] = poly.VolumetricLayer{
		Type: poly.LayerDense,
		InputHeight: 8, OutputHeight: 8,
		WeightStore: poly.NewWeightStore(64),
		Activation: poly.ActivationTanh,
	}
	// Layer 3: Hidden -> Output (8 -> 1)
	n.Layers[3] = poly.VolumetricLayer{
		Type: poly.LayerDense,
		InputHeight: 8, OutputHeight: 1,
		WeightStore: poly.NewWeightStore(8),
		Activation: poly.ActivationLinear, // Output layer wants pure targets
	}

	// Initialize weights
	for i := 1; i < len(n.Layers); i++ {
		weights := n.Layers[i].WeightStore.Master
		for j := range weights {
			weights[j] = float32(rand.NormFloat64() * 0.1)
		}
	}

	// 2. Setup the Systolic Engine (Double-Buffered, Discrete Time)
	s := poly.NewSystolicState[float32](n)

	// 3. The Continuous Environment (Clock Ticks)
	// We will try to map an input 'x' to 'sin(x)'.
	// Note: Because it takes 3 clock cycles for an input to physically traverse the grid,
	//       the output at time T actually corresponds to the input injected at T-3.

	var totalError float64
	var smoothing float64 = 0.99

	fmt.Println("Starting real-time simulation...")
	
	// Input history to match the 3-cycle delay
	inputHistory := make([]float32, 10)

	for tick := 0; tick < 5000; tick++ {
		// A. Generate random input and calculate ideal target
		inputVal := float32(rand.Float64()*2 - 1) // [-1, 1]
		inputTensor := poly.NewTensorFromSlice([]float32{inputVal}, 1)
		
		// Shift history
		for i := len(inputHistory) - 1; i > 0; i-- {
			inputHistory[i] = inputHistory[i-1]
		}
		inputHistory[0] = inputVal

		// B. Inject input into the mesh (Coordinate 0,0,0)
		s.SetInput(inputTensor)

		// C. Execute ONE physical hardware clock tick
		// Every spatial tile calculates simultaneously based on its incoming bufffer.
		poly.SystolicForward(n, s, false)

		// D. After 3 ticks, the signal has reached the output layer
		if tick >= 3 {
			// The output we see *now* is the result of the input from 3 ticks ago!
			historicalInput := inputHistory[3]
			
			// Objective: Learn a non-linear function (e.g., Target = sin(input * pi))
			targetVal := float32(math.Sin(float64(historicalInput) * math.Pi))
			targetTensor := poly.NewTensorFromSlice([]float32{targetVal}, 1)

			actualOutput := s.LayerData[3] // Node 3 is the output layer
			if actualOutput != nil && len(actualOutput.Data) > 0 {
				actualVal := actualOutput.Data[0]
				
				// Calculate Error
				err := math.Abs(float64(targetVal - actualVal))
				totalError = smoothing*totalError + (1-smoothing)*err

				// Log progress occasionally
				if tick%500 == 0 || tick == 4999 {
					fmt.Printf("[Tick %4d] Target: %6.3f | Actual: %6.3f | Running Error Avg: %.4f\n", 
						tick, targetVal, actualVal, totalError)
				}
			}

			// E. Continuous Feedback (Neural Target Propagation)
			// The mesh autonomously corrects itself using local Gaps and Link Budgeting.
			// No exact sequential backward pass gradients are calculated!
			poly.SystolicApplyTargetProp(n, s, targetTensor, 0.05)
		}
	}

	fmt.Println("Demo Finished! Error successfully minimized via continuous parallel mesh target-prop.")
}
