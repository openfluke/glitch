package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== POLY RECURSIVE META-COGNITION DEMO ===")
	fmt.Println("Scenario: A network monitors its own thoughts and injects noise or intervenes.")

	// 1. CREATE A BASE LAYER
	baseLayer := &poly.VolumetricLayer{
		Type:       poly.LayerDense,
		InputHeight: 4,
		OutputHeight: 4,
		Activation: poly.ActivationLinear,
	}
	baseLayer.WeightStore = poly.NewWeightStore(0)
	// Simple identity matrix (indices 0..15). 
	// dense.go does not add bias, so we only need the 16 weights.
	baseLayer.WeightStore.Master = []float32{
		1, 0, 0, 0, 
		0, 1, 0, 0, 
		0, 0, 1, 0, 
		0, 0, 0, 1,
	}

	// 2. CREATE A META-NETWORK FOR NOISE INJECTION
	// It will observe stats and output a noise scale.
	metaNoiseNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mNoiseLayer := &metaNoiseNet.Layers[0]
	mNoiseLayer.Type = poly.LayerDense
	mNoiseLayer.InputHeight = 5 // [Avg, Max, Min, Active, BiasTerm]
	mNoiseLayer.OutputHeight = 1
	mNoiseLayer.WeightStore = poly.NewWeightStore(0)
	// Weights: If Max > 3, output noise scale = 1.0, else 0.0
	mNoiseLayer.WeightStore.Master = []float32{
		0, 10.0, 0, 0, -30.0, // (Max * 10 - 30). If Max=4, sum=10. If Max=2, sum=-10.
	}

	// 3. CREATE THE META-COGNITION LAYER (Level 1)
	metaCogLayer := &poly.VolumetricLayer{
		Type:              poly.LayerMetacognition,
		MetaNetwork:       metaNoiseNet,
		MetaSource:        "stats",
		MetaEffect:        "noise", // Effect: Noise injection
		MetaObservedLayer: baseLayer,
	}

	// 4. TEST NORMAL vs GLITCHY
	normalInput := poly.NewTensorFromSlice([]float32{1.0, 1.0, 1.0, 1.0}, 1, 4)
	fmt.Println("\n--- Test 1: Normal Input (Max=1.0) ---")
	_, outNormal := poly.DispatchLayer(metaCogLayer, normalInput, nil)
	fmt.Println("Output (Should be clean):", outNormal.Data)

	glitchInput := poly.NewTensorFromSlice([]float32{1.0, 5.0, 1.0, 1.0}, 1, 4)
	fmt.Println("\n--- Test 2: Glitchy Input (Max=5.0) -> Noise Injection ---")
	_, outGlitch := poly.DispatchLayer(metaCogLayer, glitchInput, nil)
	fmt.Println("Output (Should be noisy due to meta-intervention):", outGlitch.Data)

	// 5. RECURSIVE CONSCIOUSNESS (Level 2)
	// Let's wrap Level 1 in another meta-layer that monitors the NOISE and suppresses it if too high.
	metaInterventionNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mIntLayer := &metaInterventionNet.Layers[0]
	mIntLayer.Type = poly.LayerDense
	mIntLayer.InputHeight = 5
	mIntLayer.OutputHeight = 1
	mIntLayer.WeightStore = poly.NewWeightStore(0)
	// If Max > 10 (very noisy), trigger intervention (output > 0.5)
	mIntLayer.WeightStore.Master = []float32{
		0, 1.0, 0, 0, -10.0,
	}

	higherCogLayer := &poly.VolumetricLayer{
		Type:              poly.LayerMetacognition,
		MetaNetwork:       metaInterventionNet,
		MetaSource:        "stats",
		MetaEffect:        "intervention", // Effect: Intervention (Shutdown)
		MetaObservedLayer: metaCogLayer,
	}

	fmt.Println("\n--- Test 3: Recursive Consciousness (Level 2 monitoring Level 1) ---")
	megaGlitchInput := poly.NewTensorFromSlice([]float32{1.0, 100.0, 1.0, 1.0}, 1, 4)
	_, outMega := poly.DispatchLayer(higherCogLayer, megaGlitchInput, nil)
	fmt.Println("Output (Should be zeroed out/intervened by Level 2):", outMega.Data)

	fmt.Println("\nSUCCESS: Metacognition stack is alive and kicking.")
}
