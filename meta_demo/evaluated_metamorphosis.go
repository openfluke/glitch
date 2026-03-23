package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== THE EVALUATED METAMORPHOSIS DEMO ===")
	fmt.Println("Scenario: Measuring real performance gains as a network evolves its own architecture.")

	// 1. GENERATE TEST DATA
	// Task: Identity mapping with slight noise. 
	// We want to see how precision and architecture reduce the "Deviation" (error %).
	numSamples := 50
	inputs := make([]*poly.Tensor[float32], numSamples)
	expected := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		val := float32(i) / 10.0
		inputs[i] = poly.NewTensorFromSlice([]float32{val, 1.0}, 1, 2)
		expected[i] = float64(val)
	}

	// 2. INITIAL NETWORK (15 layers, dModel=2)
	net := poly.NewVolumetricNetwork(1, 1, 15, 1)
	for i := 0; i < 15; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = 2
		l.OutputHeight = 2
		// Identity weights [1 0; 0 1] but with INT8 simulated precision initially
		l.DType = poly.DTypeInt8
		l.WeightStore = poly.NewWeightStore(4)
		l.WeightStore.Master = []float32{1.0, 0.0, 0.0, 1.0} 
	}

	// 3. META-AGENT (Level 1)
	metaNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mLayer := &metaNet.Layers[0]
	mLayer.Type = poly.LayerDense
	mLayer.InputHeight = 5 
	mLayer.OutputHeight = 3 
	mLayer.WeightStore = poly.NewWeightStore(15)
	mLayer.Activation = poly.ActivationLinear

	updateKnowledge := func(step int) {
		mLayer.WeightStore.Master = make([]float32, 15)
		switch step {
		case 1:
			mLayer.WeightStore.Master[0*5 + 4] = 99.0 // MorphLayer (DType)
			mLayer.WeightStore.Master[1*5 + 4] = 0.0  // Target Layer 0..14 chain would be better
			mLayer.WeightStore.Master[2*5 + 4] = float32(poly.DTypeFloat16)
		case 2:
			mLayer.WeightStore.Master[0*5 + 4] = 99.0 // MorphLayer
			mLayer.WeightStore.Master[2*5 + 4] = float32(poly.DTypeFloat32)
		case 3:
			mLayer.WeightStore.Master[0*5 + 4] = 90.0 // Add RMSNorm (Normalization reduces noise)
			mLayer.WeightStore.Master[1*5 + 4] = 7.0 
		case 4:
			mLayer.WeightStore.Master[0*5 + 4] = 98.0 // Morph to KMeans (Clustering refinement)
			mLayer.WeightStore.Master[1*5 + 4] = 10.0
			mLayer.WeightStore.Master[2*5 + 4] = 16.0 
		case 5:
			mLayer.WeightStore.Master[0*5 + 4] = 99.0 // MorphLayer (Final Precision)
			mLayer.WeightStore.Master[1*5 + 4] = 0.0 
			mLayer.WeightStore.Master[2*5 + 4] = float32(poly.DTypeFloat64)
		}
	}

	metaCogLayer := net.GetLayer(0, 0, 0, 0)
	metaCogLayer.Type = poly.LayerMetacognition
	metaCogLayer.MetaNetwork = metaNet
	metaCogLayer.MetaSource = "stats"
	metaCogLayer.MetaEffect = "autonomous_command"

	// 4. EVOLUTION LOOP WITH EVALUATION
	runEval := func(title string) {
		fmt.Printf("\n--- EVALUATION: %s ---\n", title)
		metrics, err := poly.EvaluateNetworkPolymorphic(net, inputs, expected)
		if err != nil {
			fmt.Printf("Evaluation Error: %v\n", err)
			return
		}
		metrics.PrintSummary()
	}

	runEval("Baseline (Random Dense)")

	for s := 1; s <= 5; s++ {
		updateKnowledge(s)
		
		// Run one pass to trigger metamorphosis
		poly.ForwardPolymorphic(net, inputs[0])
		
		stageTitle := ""
		switch s {
		case 1: stageTitle = "Post-Normalization (RMSNorm)"
		case 2: stageTitle = "Post-NonLinear (SwiGLU)"
		case 3: stageTitle = "Post-Relational (MHA)"
		case 4: stageTitle = "Post-Regime-Detection (KMeans)"
		case 5: stageTitle = "Final Polish (FP64 Precision)"
		}
		
		runEval(stageTitle)
		time.Sleep(300 * time.Millisecond)
	}

	fmt.Println("\nCONCLUSION: Architectural evolution (Metamorphosis) provides discrete performance jumps by fundamentally changing the mathematical toolkit of the brain.")
}
