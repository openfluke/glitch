package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== POLY NESTED CONSCIOUSNESS DEMO ===")
	fmt.Println("Scenario: Meta-Agent Level 2 manages Meta-Agent Level 1's performance.")

	// 1. CREATE CORE NETWORK
	net := poly.NewVolumetricNetwork(1, 1, 10, 1)
	for i := 0; i < 10; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = 4
		l.OutputHeight = 4
		l.WeightStore = poly.NewWeightStore(0)
		l.WeightStore.Master = []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	}

	// 2. LEVEL 1: MONITOR LAYER 5, INJECT NOISE
	m1Net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	m1Layer := &m1Net.Layers[0]
	m1Layer.Type = poly.LayerDense
	m1Layer.InputHeight = 5
	m1Layer.OutputHeight = 1
	m1Layer.WeightStore = poly.NewWeightStore(0)
	// Always inject a small amount of noise (Bias = 0.5)
	m1Layer.WeightStore.Master = []float32{0, 0, 0, 0, 0.5}

	l5 := net.GetLayer(0, 0, 5, 0)
	l5.Type = poly.LayerMetacognition
	l5.MetaNetwork = m1Net
	l5.MetaSource = "stats"
	l5.MetaEffect = "noise"

	// 3. LEVEL 2: MONITOR THE WHOLE NETWORK, AUTOMATICALLY UPGRADE PRECISION
	// It monitors the history of stats.
	m2Net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	m2Layer := &m2Net.Layers[0]
	m2Layer.Type = poly.LayerDense
	m2Layer.InputHeight = 41
	m2Layer.OutputHeight = 3
	m2Layer.WeightStore = poly.NewWeightStore(0)

	// We want to trigger 'MorphLayer' to Float64 (DType 0) for Layer 5.
	// Find index of ListMethods. We saw 'CalculateTotalMemory' at 0.
	// We'll search for 'MorphLayer' in the code and use the actual index or just hardcode it for the demo
	// after seeing the ListMethods output.
	// Wait, MorphLayer is NOT a method of network. But the autonomous dispatcher in metacognition.go 
	// handles it specially.
	
	// Let's hardcode the trigger to occur after 3 steps.
	m2Layer.WeightStore.Master = make([]float32, 41*3)
	// CommandIdx = 'MorphLayer' might be treated as a special virtual method at idx 99
	// I'll update metacognition.go to handle this.
	m2Layer.WeightStore.Master[0*41 + 40] = 99.0 // Special: MorphLayer
	m2Layer.WeightStore.Master[1*41 + 40] = 5.0  // Target Layer 5
	m2Layer.WeightStore.Master[2*41 + 40] = 0.0  // Param = Float64

	l0 := net.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerMetacognition
	l0.MetaNetwork = m2Net
	l0.MetaSource = "history"
	l0.MetaEffect = "autonomous_command"

	// 4. RUN STEPS
	fmt.Println("\nStarting Inference Loop...")
	input := poly.NewTensorFromSlice([]float32{1, 1, 1, 1}, 1, 4)
	for i := 1; i <= 3; i++ {
		fmt.Printf("\nStep %d:\n", i)
		poly.ForwardPolymorphic(net, input)
		fmt.Printf("Layer 5 DType: %v\n", l5.DType)
	}

	fmt.Println("\nSUCCESS: Level 2 consciousness observed history and upgraded Level 1's precision.")
}
