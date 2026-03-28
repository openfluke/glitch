package main

import (
	"fmt"
	"os"
	"reflect"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== Loom v0.75.0 Go Sanity Check ===")

	savePath := "sanity_model.json"
	input := poly.NewTensor[float32](1, 4)
	input.Data = []float32{1.0, 0.5, -0.2, 0.8}

	if _, err := os.Stat(savePath); err == nil {
		// --- RELOAD MODE ---
		fmt.Printf("[RELOAD MODE] Found existing model: %s\n", savePath)
		jsonData, err := os.ReadFile(savePath)
		if err != nil {
			fmt.Printf("      Read Failed: %v\n", err)
			os.Exit(1)
		}

		reloaded, err := poly.DeserializeNetwork(jsonData)
		if err != nil {
			fmt.Printf("      Deserialization Failed: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("[1/1] Verification (Reloaded Output)...")
		reloadedOut, _, _ := poly.ForwardPolymorphic(reloaded, input)
		fmt.Printf("      Reloaded Output: %v\n", reloadedOut.Data)
		fmt.Println("\n=== SANITY CHECK COMPLETE: RELOAD VERIFIED ===")
		return
	}

	// --- STANDARD TRAINING MODE ---
	fmt.Println("[STANDARD MODE] No existing model found. Running training flow...")

	// 1. Create a simple model (1x1x1 grid, 2 layers per cell)
	fmt.Println("[1/5] Creating model...")
	depth, rows, cols, layersPerCell := 1, 1, 1, 2
	net := poly.NewVolumetricNetwork(depth, rows, cols, layersPerCell)
	
	hidden := 8
	
	// Layer 0: Dense
	l0 := net.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerDense
	l0.InputHeight = 4
	l0.OutputHeight = hidden
	l0.Activation = poly.ActivationSilu
	l0.DType = poly.DTypeFloat32
	
	// Layer 1: Dense
	l1 := net.GetLayer(0, 0, 0, 1)
	l1.Type = poly.LayerDense
	l1.InputHeight = hidden
	l1.OutputHeight = 2
	l1.Activation = poly.ActivationLinear
	l1.DType = poly.DTypeFloat32

	// Manually initialize weights for both layers
	seed := time.Now().UnixNano()
	for i := range net.Layers {
		l := &net.Layers[i]
		wCount := l.InputHeight * l.OutputHeight
		if wCount > 0 {
			l.WeightStore = poly.NewWeightStore(wCount)
			l.WeightStore.Randomize(seed+int64(i), 0.1)
		}
	}

	// 2. Initial Forward Pass (CPU)
	fmt.Println("[2/5] Initial forward pass (CPU)...")
	initialOut, _, _ := poly.ForwardPolymorphic(net, input)
	initialData := make([]float32, len(initialOut.Data))
	copy(initialData, initialOut.Data)
	fmt.Printf("      Initial Output: %v\n", initialData)

	// 3. GPU Training
	fmt.Println("[3/5] GPU Training (100 epochs)...")
	target := poly.NewTensor[float32](1, 2)
	target.Data = []float32{0.1, 0.9} // Dummy target

	batches := []poly.TrainingBatch[float32]{
		{Input: input, Target: target},
	}
	config := poly.DefaultTrainingConfig()
	config.Epochs = 100
	config.LearningRate = 0.05
	config.Mode = poly.TrainingModeGPUNormal
	config.Verbose = false

	res, err := poly.Train(net, batches, config)
	if err != nil {
		fmt.Printf("      GPU Training Failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("      Final Loss: %.6f\n", res.FinalLoss)

	// Sync back to CPU
	if err := poly.SyncWeightsFromGPU(net); err != nil {
		fmt.Printf("      Sync Back Failed: %v\n", err)
		os.Exit(1)
	}

	// 4. Trained Forward Pass (CPU)
	fmt.Println("[4/5] Verification (Trained vs Initial)...")
	trainedOut, _, _ := poly.ForwardPolymorphic(net, input)
	
	isDifferent := false
	for i := range initialData {
		if initialData[i] != trainedOut.Data[i] {
			isDifferent = true
			break
		}
	}
	
	if !isDifferent {
		fmt.Println("      ❌ ERROR: Trained output is identical to initial output!")
		os.Exit(1)
	}
	fmt.Printf("      Trained Output: %v\n", trainedOut.Data)
	fmt.Println("      ✅ PASS: Model learned something.")

	// 5. Save and Reload
	fmt.Println("[5/5] Serialization (Save/Load)...")
	jsonData, err := poly.SerializeNetwork(net)
	if err != nil {
		fmt.Printf("      Serialization Failed: %v\n", err)
		os.Exit(1)
	}

	os.WriteFile(savePath, jsonData, 0644)
	fmt.Printf("      Saved to %s (%d bytes)\n", savePath, len(jsonData))

	reloaded, err := poly.DeserializeNetwork(jsonData)
	if err != nil {
		fmt.Printf("      Deserialization Failed: %v\n", err)
		os.Exit(1)
	}

	reloadedOut, _, _ := poly.ForwardPolymorphic(reloaded, input)
	if !reflect.DeepEqual(trainedOut.Data, reloadedOut.Data) {
		fmt.Printf("      ❌ ERROR: Reloaded output mismatch!\n")
		fmt.Printf("      Trained:  %v\n", trainedOut.Data)
		fmt.Printf("      Reloaded: %v\n", reloadedOut.Data)
		os.Exit(1)
	}

	fmt.Println("      ✅ PASS: Reloaded output matches trained output bit-perfectly.")
	fmt.Println("\n=== SANITY CHECK COMPLETE: v0.75.0 VERIFIED ===")
}
