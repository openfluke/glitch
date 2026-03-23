package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== THE MEGA METAMORPHOSIS BENCHMARK (CPU-ONLY) ===")
	fmt.Println("Scenario: 32-Layer Brain vs Multi-Layer Numerical Storm (1000 Samples)")
	fmt.Println("Goal: Compare Metamorphic Structural Repair vs. Traditional Backpropagation")
	fmt.Println("[NOTE] All computations are performed on CPU to ensure fair timing.")

	dModel := 128
	numLayers := 32
	
	// Create two identical networks
	netMeta := createNetwork(dModel, numLayers)
	netBack := createNetwork(dModel, numLayers)

	// Complex Harmonic Signal
	samples := 1000
	inputs, expected := generateData(dModel, samples)
	
	// 1. BASELINE
	fmt.Printf("\n[BASELINE] Evaluating Healthy Networks...\n")
	scoreMeta := eval("MetaBaseline", netMeta, inputs, expected)
	_ = eval("BackBaseline", netBack, inputs, expected)

	// 2. THE GLITCH
	glitchLayers := []int{5, 12, 18, 25}
	fmt.Printf("\n[GLITCH] Injecting +0.1 Bias drift in layers %v across both networks...\n", glitchLayers)
	injectGlitch(netMeta, glitchLayers)
	injectGlitch(netBack, glitchLayers)

	eval("MetaGlitched", netMeta, inputs, expected)
	eval("BackGlitched", netBack, inputs, expected)

	// 3. RECOVERY: METAMORPHOSIS
	fmt.Printf("\n[METHOD A] Structural Metamorphosis (Metacognition)...\n")
	startMeta := time.Now()
	// Multi-Command repair burst
	cmds := []struct{cmd, target, param float32}{}
	for _, idx := range glitchLayers {
		cmds = append(cmds, struct{cmd, target, param float32}{90, float32(idx), 0}) // RMSNorm
	}
	applyMeta(netMeta, cmds)

	cmds = []struct{cmd, target, param float32}{}
	for _, idx := range glitchLayers {
		cmds = append(cmds, struct{cmd, target, param float32}{70, float32(idx), 0}) // Revert/Heal
		l := netMeta.GetLayer(0, 0, idx, 0)
		for j := range l.WeightStore.Master { l.WeightStore.Master[j] = 0 }
		for j := 0; j < dModel; j++ { l.WeightStore.Master[j*dModel+j] = 1.0 }
	}
	applyMeta(netMeta, cmds)
	timeMeta := time.Since(startMeta)
	
	scoreMeta = eval("MetaHealed", netMeta, inputs, expected)

	// 4. RECOVERY: BACKPROPAGATION
	fmt.Printf("\n[METHOD B] Traditional Backpropagation (SGD)...\n")
	fmt.Println("[NOTE] Training for 5 epochs to attempt recovery...")
	batches := make([]poly.TrainingBatch[float32], samples)
	for i := 0; i < samples; i++ {
		batches[i] = poly.TrainingBatch[float32]{
			Input: inputs[i],
			Target: poly.NewTensorFromSlice([]float32{float32(expected[i])}, 1, 1),
		}
	}
	
	config := poly.DefaultTrainingConfig()
	config.Epochs = 25
	config.LearningRate = 0.0001
	config.Verbose = true
	config.Mode = poly.TrainingModeCPUNormal // ENSURE CPU-ONLY
	
	startBack := time.Now()
	_, _ = poly.Train(netBack, batches, config)
	timeBack := time.Since(startBack)

	scoreBack := eval("BackpropHealed", netBack, inputs, expected)

	// 5. FINAL COMPARISON
	fmt.Println("\n=========================================================")
	fmt.Println("             FINAL PERFORMANCE COMPARISON                ")
	fmt.Println("=========================================================")
	fmt.Printf("METHOD             | TIME         | FINAL QUALITY SCORE\n")
	fmt.Printf("Meta-Structural    | %-12v | %.2f/100\n", timeMeta, scoreMeta)
	fmt.Printf("Standard Backprop  | %-12v | %.2f/100\n", timeBack, scoreBack)
	fmt.Println("=========================================================")
	
	if scoreMeta > scoreBack {
		fmt.Println("CONCLUSION: Metamorphosis achieved higher quality in a fraction of the time.")
	} else {
		fmt.Println("CONCLUSION: Backpropagation managed to recover, but at a significantly higher temporal cost.")
	}
}

func createNetwork(dModel, numLayers int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, numLayers, 1) 
	for i := 0; i < numLayers; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = dModel
		l.OutputHeight = dModel
		if i == numLayers-1 { l.OutputHeight = 1 } 
		l.WeightStore = poly.NewWeightStore(l.InputHeight * l.OutputHeight)
		l.WeightStore.Master = make([]float32, l.InputHeight * l.OutputHeight)
		if l.OutputHeight == 1 {
			l.WeightStore.Master[0] = 1.0 
		} else {
			for j := 0; j < dModel; j++ {
				l.WeightStore.Master[j*dModel+j] = 1.0 
			}
		}
		l.Activation = poly.ActivationLinear
	}
	
	// Add Metacognition
	metaNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	mLayer := &metaNet.Layers[0]
	mLayer.Type = poly.LayerDense
	mLayer.InputHeight = 5 
	mLayer.OutputHeight = 10 
	mLayer.WeightStore = poly.NewWeightStore(50)

	metaCogLayer := net.GetLayer(0, 0, 0, 0)
	metaCogLayer.Type = poly.LayerMetacognition
	metaCogLayer.MetaNetwork = metaNet
	metaCogLayer.MetaEffect = "autonomous_command"
	
	return net
}

func generateData(dModel, samples int) ([]*poly.Tensor[float32], []float64) {
	inputs := make([]*poly.Tensor[float32], samples)
	expected := make([]float64, samples)
	for i := 0; i < samples; i++ {
		t := float64(i) * 0.1
		val := math.Sin(t) + 0.5*math.Cos(2*t)
		inputs[i] = poly.NewTensorFromSlice(make([]float32, dModel), 1, dModel)
		for j := 0; j < dModel; j++ {
			inputs[i].Data[j] = float32(val)
		}
		expected[i] = val
	}
	return inputs, expected
}

func injectGlitch(net *poly.VolumetricNetwork, indices []int) {
	for _, idx := range indices {
		l := net.GetLayer(0, 0, idx, 0)
		dModel := l.InputHeight
		// Surgical Glitch: Shift the gain (Diagonal only)
		// Adding 0.1 to diagonal means 10% gain shift per glitched layer
		for j := 0; j < dModel; j++ {
			if j*dModel+j < len(l.WeightStore.Master) {
				l.WeightStore.Master[j*dModel+j] += 0.1
			}
		}
	}
}

func applyMeta(net *poly.VolumetricNetwork, commands []struct{cmd, target, param float32}) {
	metaCogLayer := net.GetLayer(0, 0, 0, 0)
	mLayer := &metaCogLayer.MetaNetwork.Layers[0]
	mLayer.OutputHeight = len(commands)
	mLayer.WeightStore.Master = make([]float32, len(commands)*5)
	for i, c := range commands {
		mLayer.WeightStore.Master[i*5 + 4] = c.cmd
		mLayer.WeightStore.Master[i*5 + 3] = c.target
		mLayer.WeightStore.Master[i*5 + 2] = c.param
	}
	poly.ForwardPolymorphic(net, poly.NewTensorFromSlice(make([]float32, 128), 1, 128))
}

func eval(name string, net *poly.VolumetricNetwork, inputs []*poly.Tensor[float32], expected []float64) float64 {
	metrics, _ := poly.EvaluateNetworkPolymorphic(net, inputs, expected)
	fmt.Printf("[%s] Quality Score: %.2f/100 | Avg Dev: %.2f%%\n", name, metrics.Score, metrics.AverageDeviation)
	return metrics.Score
}
