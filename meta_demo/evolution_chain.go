package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== THE BRAIN EVOLUTION CHAIN DEMO ===")
	fmt.Println("Scenario: A 5-stage architectural metamorphosis to solve an increasingly difficult task.")

	// 1. INITIAL SETUP
	dModel := 8
	net := poly.NewVolumetricNetwork(1, 1, 10, 1) // 10 layers
	
	// Start layers 1..5 as simple identity Dense layers
	for i := 1; i <= 5; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = dModel
		l.OutputHeight = dModel
		l.WeightStore = poly.NewWeightStore(dModel * dModel)
		// Identity
		for j := 0; j < dModel; j++ {
			l.WeightStore.Master[j*dModel+j] = 1.0
		}
	}
	// Disable others
	for i := 6; i < 10; i++ { net.GetLayer(0, 0, i, 0).IsDisabled = true }

	// 2. THE EVOLUTIONARY META-AGENT (Level 2)
	// It will progress through a hardcoded evolutionary path for this demo.
	// 5 stages: Normal -> Norm -> Attention -> Sequential -> Specialized -> High-Precision
	evolutionStep := 0
	
	metaNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mLayer := &metaNet.Layers[0]
	mLayer.Type = poly.LayerDense
	mLayer.InputHeight = 5 // stats
	mLayer.OutputHeight = 3 // cmd, target, param
	mLayer.WeightStore = poly.NewWeightStore(15)
	mLayer.Activation = poly.ActivationLinear

	// We'll update the meta-agent's "knowledge" (weights) each stage to trigger the next evolution
	updateKnowledge := func(step int) {
		mLayer.WeightStore.Master = make([]float32, 15)
		switch step {
		case 1:
			mLayer.WeightStore.Master[0*5 + 4] = 90.0 // MorphToRMSNorm
			mLayer.WeightStore.Master[1*5 + 4] = 1.0  // Target Layer 1
		case 2:
			mLayer.WeightStore.Master[0*5 + 4] = 92.0 // MorphToMHA
			mLayer.WeightStore.Master[1*5 + 4] = 2.0  // Target Layer 2
		case 3:
			mLayer.WeightStore.Master[0*5 + 4] = 93.0 // MorphToLSTM
			mLayer.WeightStore.Master[1*5 + 4] = 3.0  // Target Layer 3
		case 4:
			mLayer.WeightStore.Master[0*5 + 4] = 98.0 // MorphToKMeans
			mLayer.WeightStore.Master[1*5 + 4] = 4.0  // Target Layer 4
			mLayer.WeightStore.Master[2*5 + 4] = float32(dModel) // 8 clusters
		case 5:
			mLayer.WeightStore.Master[0*5 + 4] = 99.0 // MorphLayer (DType)
			mLayer.WeightStore.Master[1*5 + 4] = 5.0  // Target Layer 5
			mLayer.WeightStore.Master[2*5 + 4] = 0.0  // Float64
		}
	}

	metaCogLayer := net.GetLayer(0, 0, 0, 0)
	metaCogLayer.Type = poly.LayerMetacognition
	metaCogLayer.MetaNetwork = metaNet
	metaCogLayer.MetaSource = "stats"
	metaCogLayer.MetaEffect = "autonomous_command"

	// 3. EXECUTION LOOP
	input := poly.NewTensorFromSlice([]float32{1, 1, 1, 1, 1, 1, 1, 1}, 1, dModel)
	simulatedError := 1.0
	
	fmt.Printf("\n[INITIAL] Network Layers: %s, %s, %s, %s, %s\n", 
		net.Layers[1].Type, net.Layers[2].Type, net.Layers[3].Type, net.Layers[4].Type, net.Layers[5].Type)
	fmt.Printf("Simulated System Error: %.4f\n", simulatedError)

	for s := 1; s <= 5; s++ {
		evolutionStep = s
		updateKnowledge(evolutionStep)
		
		fmt.Printf("\n--- EVOLUTION STAGE %d ---\n", s)
		poly.ForwardPolymorphic(net, input)
		
		simulatedError *= 0.618 // Golden ratio improvement for demo
		
		fmt.Printf("Metamorphosis Complete. Network is now:\n")
		fmt.Printf("L1: %v, L2: %v, L3: %v, L4: %v, L5: %v\n", 
			net.Layers[1].Type, net.Layers[2].Type, net.Layers[3].Type, net.Layers[4].Type, net.Layers[5].Type)
		fmt.Printf("L5 Precision: %v\n", net.Layers[5].DType)
		fmt.Printf("New Simulated Error: %.4f\n", simulatedError)
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("\nCONCLUSION: The brain successfully evolved through 5 architectural stages, improving its specialized capabilities each time. Backprop can only tune weights, but Metacognition can rebuild the brain.")
}
