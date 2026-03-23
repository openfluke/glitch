package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== CONTINUOUS EVOLUTION DEMO ===")
	fmt.Println("Scenario: A network evolves indefinitely by observing its own performance metrics.")

	dModel := 8
	net := poly.NewVolumetricNetwork(1, 1, 10, 1)
	for i := 0; i < 10; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = dModel
		l.OutputHeight = dModel
		l.WeightStore = poly.NewWeightStore(dModel * dModel)
		l.WeightStore.Randomize(rand.Int63(), 0.5)
	}

	metaNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mLayer := &metaNet.Layers[0]
	mLayer.Type = poly.LayerDense
	mLayer.InputHeight = 5 // stats
	mLayer.OutputHeight = 3 // cmd, target, param
	mLayer.WeightStore = poly.NewWeightStore(15)
	mLayer.Activation = poly.ActivationLinear

	metaCogLayer := net.GetLayer(0, 0, 0, 0)
	metaCogLayer.Type = poly.LayerMetacognition
	metaCogLayer.MetaNetwork = metaNet
	metaCogLayer.MetaSource = "stats"
	metaCogLayer.MetaEffect = "autonomous_command"

	commands := []float32{80, 81, 83, 84, 90, 91, 92, 93, 98, 99}
	input := poly.NewTensorFromSlice(make([]float32, dModel), 1, dModel)
	for i := range input.Data { input.Data[i] = 1.0 }

	fmt.Println("\nStarting Long Evolution Loop (500 iterations)...")
	fmt.Println("Press Ctrl+C to stop.")

	for i := 1; i <= 500; i++ {
		// Meta-agent logic: Mutate every few iterations
		if i % 5 == 0 {
			cmd := commands[rand.Intn(len(commands))]
			target := float32(rand.Intn(9) + 1)
			param := float32(rand.Intn(16))
			
			mLayer.WeightStore.Master = make([]float32, 15)
			mLayer.WeightStore.Master[0*5 + 4] = cmd
			mLayer.WeightStore.Master[1*5 + 4] = target
			mLayer.WeightStore.Master[2*5 + 4] = param
		}

		// Trigger morph if needed
		poly.ForwardPolymorphic(net, input)

		if i % 20 == 0 {
			fmt.Printf("\n--- Iteration %d Performance ---", i)
			
			// Evaluate on a batch
			testSamples := 20
			testInputs := make([]*poly.Tensor[float32], testSamples)
			testExpected := make([]float64, testSamples)
			for s := 0; s < testSamples; s++ {
				testInputs[s] = poly.NewTensorFromSlice([]float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 1, dModel)
				testExpected[s] = 1.0 // Simple parity for demo
			}
			
			metrics, _ := poly.EvaluateNetworkPolymorphic(net, testInputs, testExpected)
			metrics.PrintSummary()

			fmt.Printf("\nArchitecture Check:")
			for j := 0; j < 10; j++ {
				fmt.Printf("[%d:%s] ", j, net.Layers[j].Type)
				if (j+1)%5 == 0 { fmt.Println() }
			}
		}
		time.Sleep(50 * time.Millisecond)
	}

	fmt.Println("\nEvolution completed. The brain has explored multiple architectural configurations.")
}
