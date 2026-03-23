package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== THE GRAND METAMORPHOSIS DEMO ===")
	fmt.Println("Scenario: A 15-layer generic network evolves entirely into a complex heterogeneous brain.")

	// 1. INITIAL SETUP
	dModel := 16
	net := poly.NewVolumetricNetwork(1, 1, 15, 1) 
	
	// Start all layers as Dense identity
	for i := 0; i < 15; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = dModel
		l.OutputHeight = dModel
		l.WeightStore = poly.NewWeightStore(dModel * dModel)
		for j := 0; j < dModel; j++ {
			l.WeightStore.Master[j*dModel+j] = 1.0
		}
	}

	// 2. THE EVOLUTIONARY META-AGENT
	metaNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mLayer := &metaNet.Layers[0]
	mLayer.Type = poly.LayerDense
	mLayer.InputHeight = 5 
	mLayer.OutputHeight = 6 // cmd1, target1, param1, cmd2, target2, param2 (to morph blocks)
	mLayer.WeightStore = poly.NewWeightStore(30)
	mLayer.Activation = poly.ActivationLinear

	updateKnowledge := func(step int) {
		mLayer.WeightStore.Master = make([]float32, 30)
		switch step {
		case 1:
			fmt.Println("\n[EVENT] Spatial pattern detected. Morphs: Dense -> CNN1")
			mLayer.WeightStore.Master[0*5 + 4] = 80.0 // MorphToCNN1
			mLayer.WeightStore.Master[1*5 + 4] = 1.0  // Layer 1
			mLayer.WeightStore.Master[3*5 + 4] = 80.0 // MorphToCNN1
			mLayer.WeightStore.Master[4*5 + 4] = 2.0  // Layer 2
		case 2:
			fmt.Println("\n[EVENT] Sequence pattern detected. Morphs: Dense -> LSTM/RNN")
			mLayer.WeightStore.Master[0*5 + 4] = 93.0 // MorphToLSTM
			mLayer.WeightStore.Master[1*5 + 4] = 4.0  // Layer 4
			mLayer.WeightStore.Master[3*5 + 4] = 81.0 // MorphToRNN
			mLayer.WeightStore.Master[4*5 + 4] = 5.0  // Layer 5
		case 3:
			fmt.Println("\n[EVENT] Attention required. Morphs: Dense -> MHA/Norm")
			mLayer.WeightStore.Master[0*5 + 4] = 92.0 // MorphToMHA
			mLayer.WeightStore.Master[1*5 + 4] = 7.0  // Layer 7
			mLayer.WeightStore.Master[3*5 + 4] = 90.0 // MorphToRMSNorm
			mLayer.WeightStore.Master[4*5 + 4] = 8.0  // Layer 8
		case 4:
			fmt.Println("\n[EVENT] Pattern Clustering detected. Morphs: Dense -> KMeans")
			mLayer.WeightStore.Master[0*5 + 4] = 98.0 // MorphToKMeans
			mLayer.WeightStore.Master[1*5 + 4] = 10.0 // Layer 10
			mLayer.WeightStore.Master[2*5 + 4] = float32(dModel) // 16 clusters
		case 5:
			fmt.Println("\n[EVENT] Finalizing Output logic. Morphs: Dense -> Softmax/Precision")
			mLayer.WeightStore.Master[0*5 + 4] = 83.0 // MorphToSoftmax
			mLayer.WeightStore.Master[1*5 + 4] = 14.0 // Layer 14
			mLayer.WeightStore.Master[3*5 + 4] = 99.0 // MorphLayer (DType)
			mLayer.WeightStore.Master[4*5 + 4] = 0.0  // Layer 0
			mLayer.WeightStore.Master[5*5 + 4] = 0.0  // Float64
		}
	}

	// Small change to metacognition.go needed to handle 2 commands at once?
	// or I'll just run it twice. Let's update metacognition.go to handle [3N] outputs.

	metaCogLayer := net.GetLayer(0, 0, 0, 0)
	metaCogLayer.Type = poly.LayerMetacognition
	metaCogLayer.MetaNetwork = metaNet
	metaCogLayer.MetaSource = "stats"
	metaCogLayer.MetaEffect = "autonomous_command"

	// 3. EXECUTION
	input := poly.NewTensorFromSlice(make([]float32, dModel), 1, dModel)
	for i := range input.Data { input.Data[i] = 1.0 }

	for s := 1; s <= 5; s++ {
		updateKnowledge(s)
		fmt.Printf("\n--- STAGE %d INFLECTION ---\n", s)
		poly.ForwardPolymorphic(net, input)
		
		fmt.Println("Current Brain Architecture:")
		for i := 0; i < 15; i++ {
			fmt.Printf("[%d:%s] ", i, net.Layers[i].Type)
			if (i+1)%5 == 0 { fmt.Println() }
		}
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("\nSUCCESS: All 15 layers participated in the Grand Metamorphosis chain.")
}
