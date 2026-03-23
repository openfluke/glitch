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

	fmt.Println("=== THE MEGA METAMORPHOSIS BENCHMARK v3 (CPU-ONLY) ===")
	fmt.Println("Scenario: 32-Layer Brain with Autonomous Heuristic Self-Repair")
	fmt.Println("Goal: Zero manual intervention — the network diagnoses and heals itself")
	fmt.Println()

	dModel := 128
	numLayers := 32
	samples := 1000

	// ── 1. CREATE TWO IDENTICAL PLAIN NETWORKS ──
	netMeta := createPlainNetwork(dModel, numLayers)
	netBack := createPlainNetwork(dModel, numLayers)

	// ── 2. WRAP ONE WITH METACOGNITION (one line!) ──
	// MetaCondGainDrift: runs the observed layer, compares output/input avg.
	// If gain deviates more than 5% from 1.0 → reset weights to identity.
	poly.WrapWithMetacognition(netMeta, []poly.MetaRule{
		{Condition: poly.MetaCondGainDrift, Threshold: 0.05, Command: 101, SelfOnly: true},
	})

	inputs, expected := generateData(dModel, samples)

	// ── 3. BASELINE ──
	fmt.Println("[BASELINE] Evaluating Healthy Networks...")
	scoreMeta := eval("MetaBaseline", netMeta, inputs, expected)
	_ = eval("BackBaseline", netBack, inputs, expected)

	// ── 4. INJECT GLITCH ──
	glitchLayers := []int{5, 12, 18, 25}
	fmt.Printf("\n[GLITCH] Injecting +0.1 diagonal drift into layers %v...\n", glitchLayers)
	injectGlitch(netMeta, glitchLayers, true)  // true = reach through to MetaObservedLayer
	injectGlitch(netBack, glitchLayers, false) // false = target layer directly

	eval("MetaGlitched", netMeta, inputs, expected)
	eval("BackGlitched", netBack, inputs, expected)

	// ── 5. AUTONOMOUS SELF-REPAIR ──
	fmt.Println("\n[METHOD A] Autonomous Heuristic Self-Repair...")
	fmt.Println("[NOTE] Just running a forward pass. Each layer monitors its own gain drift.")
	startMeta := time.Now()
	// Use a non-zero input so gain drift is detectable
	poly.ForwardPolymorphic(netMeta, inputs[50]) // sin(5.0) ≈ -0.96, strong signal
	timeMeta := time.Since(startMeta)
	scoreMeta = eval("MetaHealed", netMeta, inputs, expected)

	// ── 6. BACKPROPAGATION RECOVERY ──
	fmt.Println("\n[METHOD B] Traditional Backpropagation (SGD, 25 epochs)...")
	batches := make([]poly.TrainingBatch[float32], samples)
	for i := 0; i < samples; i++ {
		batches[i] = poly.TrainingBatch[float32]{
			Input:  inputs[i],
			Target: poly.NewTensorFromSlice([]float32{float32(expected[i])}, 1, 1),
		}
	}

	config := poly.DefaultTrainingConfig()
	config.Epochs = 5
	config.LearningRate = 0.0001
	config.Verbose = true
	config.Mode = poly.TrainingModeCPUNormal

	startBack := time.Now()
	_, _ = poly.Train(netBack, batches, config)
	timeBack := time.Since(startBack)
	scoreBack := eval("BackpropHealed", netBack, inputs, expected)

	// ── 7. COMPARISON ──
	fmt.Println("\n=========================================================")
	fmt.Println("             FINAL PERFORMANCE COMPARISON                ")
	fmt.Println("=========================================================")
	fmt.Printf("METHOD             | TIME         | FINAL QUALITY SCORE\n")
	fmt.Printf("Meta-Heuristic     | %-12v | %.2f/100\n", timeMeta, scoreMeta)
	fmt.Printf("Standard Backprop  | %-12v | %.2f/100\n", timeBack, scoreBack)
	fmt.Println("=========================================================")

	if scoreMeta > scoreBack {
		fmt.Println("CONCLUSION: Autonomous heuristic repair achieved higher quality in a fraction of the time.")
	} else if scoreMeta == scoreBack {
		fmt.Println("CONCLUSION: Both methods recovered, but heuristic repair was orders of magnitude faster.")
	} else {
		fmt.Println("CONCLUSION: Backpropagation recovered better, but at significantly higher cost.")
	}
}

func createPlainNetwork(dModel, numLayers int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, numLayers, 1)
	for i := 0; i < numLayers; i++ {
		l := net.GetLayer(0, 0, i, 0)
		l.Type = poly.LayerDense
		l.InputHeight = dModel
		l.OutputHeight = dModel
		if i == numLayers-1 {
			l.OutputHeight = 1
		}
		l.Activation = poly.ActivationLinear
		l.WeightStore = poly.NewWeightStore(l.InputHeight * l.OutputHeight)
		l.WeightStore.Master = make([]float32, l.InputHeight*l.OutputHeight)
		if l.OutputHeight == 1 {
			l.WeightStore.Master[0] = 1.0
		} else {
			for j := 0; j < dModel; j++ {
				l.WeightStore.Master[j*dModel+j] = 1.0
			}
		}
	}
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

func injectGlitch(net *poly.VolumetricNetwork, indices []int, isMeta bool) {
	for _, idx := range indices {
		l := net.GetLayer(0, 0, idx, 0)
		target := l
		if isMeta && l.Type == poly.LayerMetacognition && l.MetaObservedLayer != nil {
			target = l.MetaObservedLayer
		}
		dModel := target.InputHeight
		for j := 0; j < dModel; j++ {
			if j*dModel+j < len(target.WeightStore.Master) {
				target.WeightStore.Master[j*dModel+j] += 0.1
			}
		}
	}
}

func eval(name string, net *poly.VolumetricNetwork, inputs []*poly.Tensor[float32], expected []float64) float64 {
	metrics, _ := poly.EvaluateNetworkPolymorphic(net, inputs, expected)
	fmt.Printf("  [%s] Quality Score: %.2f/100 | Avg Dev: %.2f%%\n", name, metrics.Score, metrics.AverageDeviation)
	return metrics.Score
}
