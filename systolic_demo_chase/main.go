package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/poly"
)

// Types
type Mode int

const (
	ModeNormalBP Mode = iota
	ModeTargetProp
	ModeTargetPropChain
	ModeSystolic
	ModeSystolicChain
)

var modeNames = map[Mode]string{
	ModeNormalBP:        "NormalBP",
	ModeTargetProp:      "TargetProp",
	ModeTargetPropChain: "TargetProp+Chain",
	ModeSystolic:        "Systolic",
	ModeSystolicChain:   "Systolic+Chain",
}

type TimeWindow struct {
	Outputs       int
	Correct       int
	Accuracy      float64
	OutputsPerSec int
	CurrentTask   string
	AvailMs       int
	BlockedMs     int
	TotalLatency  time.Duration
	PeakLatency   time.Duration
}

type AdaptationResult struct {
	Windows      []TimeWindow
	TotalOutputs int

	PreChangeAccuracy   float64
	PostChange1Accuracy float64
	AdaptTime1          int
	PostChange2Accuracy float64
	AdaptTime2          int

	TotalAvailMs   int
	TotalBlockedMs int
	PeakLatency    time.Duration
	TotalLatency   time.Duration
}

type Environment struct {
	AgentPos  [2]float32
	TargetPos [2]float32
	Task      int // 0=chase, 1=avoid
}

const (
	InputSize    = 8
	ActionSize   = 4
	TestDuration = 15 * time.Second
	WindowDuration = 1 * time.Second
	LearningRate = float32(0.02)
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🏃 TEST 17: REAL-TIME ADAPTATION BENCHMARK (POLY EDITION)                             ║")
	fmt.Println("║                                                                                         ║")
	fmt.Println("║   TIMELINE: [Chase 5s] → [AVOID 5s] → [Chase 5s]                                        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	modes := []Mode{
		ModeNormalBP,
		ModeTargetProp,
		ModeTargetPropChain,
		ModeSystolic,
		ModeSystolicChain,
	}

	configs := []string{"Single", "Bicameral"}

	var mu sync.Mutex
	var wg sync.WaitGroup
	allResults := make(map[string]*AdaptationResult)

	fmt.Println("🚀 Starting Parallel Benchmark...\n")

	for _, mode := range modes {
		for _, cfg := range configs {
			wg.Add(1)
			go func(m Mode, c string) {
				defer wg.Done()
				runName := fmt.Sprintf("%s-%s", modeNames[m], c)
				result := runAdaptationTest(m, c)
				mu.Lock()
				allResults[runName] = result
				mu.Unlock()
			}(mode, cfg)
		}
	}

	wg.Wait()
	printAdaptationTimeline(allResults, modes, configs)
	printAdaptationSummary(allResults, modes, configs)
}

func DeepMirror(n1, n2 *poly.VolumetricNetwork) {
	for i := range n1.Layers {
		l1 := &n1.Layers[i]
		l2 := &n2.Layers[i]
		if l1.WeightStore != nil && l2.WeightStore != nil {
			if len(l1.WeightStore.Master) == len(l2.WeightStore.Master) {
				copy(l2.WeightStore.Master, l1.WeightStore.Master)
			}
		}
	}
}

func createDeepNetwork(tiling bool) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, 7)
	
	tileSize := 0
	if tiling {
		tileSize = 16
	}

	for l := 0; l < 7; l++ {
		layer := net.GetLayer(0, 0, 0, l)
		layer.Type = poly.LayerDense
		layer.UseTiling = tiling
		layer.TileSize = tileSize

		switch l {
		case 0:
			layer.IsDisabled = true
		case 1:
			layer.InputHeight = 8
			layer.OutputHeight = 32
			layer.Activation = poly.ActivationTanh
			layer.WeightStore = poly.NewWeightStore((8 * 32) + 32)
		case 2:
			layer.InputHeight = 32
			layer.OutputHeight = 64
			layer.Activation = poly.ActivationTanh
			layer.WeightStore = poly.NewWeightStore((32 * 64) + 64)
		case 3:
			layer.InputHeight = 64
			layer.OutputHeight = 64
			layer.Activation = poly.ActivationTanh
			layer.WeightStore = poly.NewWeightStore((64 * 64) + 64)
		case 4:
			layer.InputHeight = 64
			layer.OutputHeight = 64
			layer.Activation = poly.ActivationTanh
			layer.WeightStore = poly.NewWeightStore((64 * 64) + 64)
		case 5:
			layer.InputHeight = 64
			layer.OutputHeight = 32
			layer.Activation = poly.ActivationTanh
			layer.WeightStore = poly.NewWeightStore((64 * 32) + 32)
		case 6:
			layer.InputHeight = 32
			layer.OutputHeight = 4
			layer.Activation = poly.ActivationSigmoid
			layer.WeightStore = poly.NewWeightStore((32 * 4) + 4)
		}
		
		if !layer.IsDisabled {
			layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
		}
	}

	return net
}

func runAdaptationTest(mode Mode, config string) *AdaptationResult {
	usesTiling := false
	if mode == ModeSystolic || mode == ModeSystolicChain {
		usesTiling = true
	}
	
	isBicameral := config == "Bicameral"

	net := createDeepNetwork(usesTiling)
	var shadowNet *poly.VolumetricNetwork
	if isBicameral {
		shadowNet = createDeepNetwork(usesTiling)
		DeepMirror(net, shadowNet)
	}

	result := &AdaptationResult{
		Windows: make([]TimeWindow, 15),
	}

	env := &Environment{
		AgentPos:  [2]float32{0.5, 0.5},
		TargetPos: [2]float32{rand.Float32(), rand.Float32()},
		Task:      0, // Start with chase
	}

	var tpState *poly.TargetPropState[float32]
	var tpShadowState *poly.TargetPropState[float32]
	var sysState *poly.SystolicState[float32]

	tpConfig := poly.DefaultTargetPropConfig()
	if mode == ModeTargetPropChain || mode == ModeSystolicChain {
		tpConfig.UseChainRule = true
	} else {
		tpConfig.UseChainRule = false // Gap based
	}

	if mode == ModeTargetProp || mode == ModeTargetPropChain {
		tpState = poly.NewTargetPropState[float32](net, tpConfig)
		if isBicameral {
			tpShadowState = poly.NewTargetPropState[float32](shadowNet, tpConfig)
		}
	}

	if mode == ModeSystolic || mode == ModeSystolicChain {
		sysState = poly.NewSystolicState[float32](net)
		cfg := poly.DefaultTargetPropConfig()
		cfg.UseChainRule = tpConfig.UseChainRule
		sysState.TPState = poly.NewTargetPropState[float32](net, cfg)
	}

	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	
	trainBatch := make([]TrainingSample, 0, 50)
	var shadowBatch []TrainingSample
	
	// Continuous buffer for TargetProp/Systolic
	const TARGET_BUFFER_SIZE = 16
	targetBuffer := make([][]float32, TARGET_BUFFER_SIZE)

	var networkMx sync.Mutex
	shadowSyncInterval := 10 // sync every N training steps
	syncCounter := 0

	// Shadow Training Loop
	if isBicameral {
		go func() {
			for {
				networkMx.Lock()
				if len(shadowBatch) > 0 {
					sample := shadowBatch[0]
					shadowBatch = shadowBatch[1:]
					networkMx.Unlock()

					inputT := poly.NewTensorFromSlice(sample.Input, 1, len(sample.Input))
					targetT := poly.NewTensorFromSlice(sample.Target, 1, len(sample.Target))

					switch mode {
					case ModeNormalBP:
						batch := []poly.TrainingBatch[float32]{{
							Input:  poly.NewTensorFromSlice(sample.Input, 1, len(sample.Input)),
							Target: poly.NewTensorFromSlice(sample.Target, 1, len(sample.Target)),
						}}
						poly.Train(shadowNet, batch, &poly.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse", Verbose: false})
					case ModeTargetProp, ModeTargetPropChain:
						poly.TargetPropForward(shadowNet, tpShadowState, inputT)
						poly.TargetPropBackward(shadowNet, tpShadowState, targetT)
						poly.ApplyTargetPropGaps(shadowNet, tpShadowState, LearningRate)
					case ModeSystolic, ModeSystolicChain:
						// No bicameral for systolic usually since it's unblocking anyway, but supported for benchmark
					}

					syncCounter++
					if syncCounter >= shadowSyncInterval {
						syncCounter = 0
						networkMx.Lock()
						DeepMirror(shadowNet, net)
						networkMx.Unlock()
					}
				} else {
					networkMx.Unlock()
					time.Sleep(1 * time.Millisecond)
				}
			}
		}()
	}

	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond
	
	start := time.Now()
	currentWindow := 0
	winStart := start

	delayLength := 8
	packetCount := 0

	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)
		newWindow := int(elapsed / WindowDuration)

		if newWindow > currentWindow && newWindow < 15 {
			winElapsed := time.Since(winStart)
			result.Windows[currentWindow].OutputsPerSec = result.Windows[currentWindow].Outputs
			if result.Windows[currentWindow].Outputs > 0 {
				result.Windows[currentWindow].Accuracy = float64(result.Windows[currentWindow].Correct) / float64(result.Windows[currentWindow].Outputs) * 100
			}
			result.Windows[currentWindow].AvailMs = int(winElapsed.Milliseconds()) - result.Windows[currentWindow].BlockedMs
			winStart = time.Now()
			currentWindow = newWindow
		}

		// TASK CHANGES
		if elapsed >= 5*time.Second && elapsed < 10*time.Second {
			if env.Task != 1 {
				env.Task = 1 // Switch to AVOID
			}
		} else {
			if env.Task != 0 {
				env.Task = 0 // Back to CHASE
			}
		}

		if currentWindow < 15 {
			if env.Task == 0 {
				result.Windows[currentWindow].CurrentTask = "chase"
			} else {
				result.Windows[currentWindow].CurrentTask = "AVOID!"
			}
		}

		obs := getObservation(env)
		
		var output []float32
		var inputT *poly.Tensor[float32]
		
		loopStart := time.Now()

		networkMx.Lock()
		switch mode {
		case ModeNormalBP:
			outT, _, _ := poly.ForwardPolymorphic(net, poly.NewTensorFromSlice(obs, 1, len(obs)))
			output = outT.Data
		case ModeTargetProp, ModeTargetPropChain:
			inputT = poly.NewTensorFromSlice(obs, 1, len(obs))
			poly.TargetPropForward(net, tpState, inputT)
			// TargetPropForward processes all layers sequentially and updates ForwardActs up to TotalLayers. 
			// We fetch the output from the final index.
			outL := tpState.ForwardActs[len(net.Layers)]
			if outL != nil && len(outL.Data) > 0 {
				output = outL.Data
			}
		case ModeSystolic, ModeSystolicChain:
			inputT = poly.NewTensorFromSlice(obs, 1, len(obs))
			sysState.SetInput(inputT)
			poly.SystolicForward(net, sysState, false)
			// Systolic uses double-buffering. After 7 total layers, the final output lies at LayerData[6].
			outL := sysState.LayerData[6]
			if outL != nil && len(outL.Data) > 0 {
				output = outL.Data
			} else {
				output = make([]float32, 4)
			}
		}
		networkMx.Unlock()

		lat := time.Since(loopStart)
		if currentWindow < 15 {
			result.Windows[currentWindow].TotalLatency += lat
			if lat > result.Windows[currentWindow].PeakLatency {
				result.Windows[currentWindow].PeakLatency = lat
			}
			
			result.TotalLatency += lat
			if lat > result.PeakLatency {
				result.PeakLatency = lat
			}
		}

		action := argmax(output)
		optimalAction := getOptimalAction(env)

		if currentWindow < 15 {
			result.Windows[currentWindow].Outputs++
			result.TotalOutputs++
			if action == optimalAction {
				result.Windows[currentWindow].Correct++
			}
		}

		executeAction(env, action)

		target := make([]float32, 4)
		target[optimalAction] = 1.0

		// Handle TargetProp/Systolic Continuous target matching
		if mode == ModeTargetProp || mode == ModeTargetPropChain || mode == ModeSystolic || mode == ModeSystolicChain {
			copy(targetBuffer[1:], targetBuffer[0:TARGET_BUFFER_SIZE-1])
			targetBuffer[0] = target
		}

		// Training Dispatch
		if isBicameral {
			networkMx.Lock()
			shadowBatch = append(shadowBatch, TrainingSample{Input: obs, Target: target})
			networkMx.Unlock()
		} else {
			trainBatch = append(trainBatch, TrainingSample{Input: obs, Target: target})
			
			if time.Since(lastTrainTime) >= trainInterval && len(trainBatch) > 0 {
				blockStart := time.Now()
				
				switch mode {
				case ModeNormalBP:
					batches := make([]poly.TrainingBatch[float32], len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = poly.TrainingBatch[float32]{
							Input:  poly.NewTensorFromSlice(s.Input, 1, len(s.Input)),
							Target: poly.NewTensorFromSlice(s.Target, 1, len(s.Target)),
						}
					}
					poly.Train(net, batches, &poly.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse", Verbose: false})
					trainBatch = trainBatch[:0]
				}
				
				if currentWindow < 15 {
					result.Windows[currentWindow].BlockedMs += int(time.Since(blockStart).Milliseconds())
				}
				lastTrainTime = time.Now()
			}
			
			// Unbatched learning for these modes
			if mode == ModeTargetProp || mode == ModeTargetPropChain {
				blockStart := time.Now()
				targetT := poly.NewTensorFromSlice(target, 1, 4)
				poly.TargetPropBackward(net, tpState, targetT)
				poly.ApplyTargetPropGaps(net, tpState, LearningRate)
				if currentWindow < 15 {
					result.Windows[currentWindow].BlockedMs += int(time.Since(blockStart).Milliseconds())
				}
			} else if mode == ModeSystolic || mode == ModeSystolicChain {
				blockStart := time.Now()
				// Use the delayed target representing historical frame for clock-cycle correctness
				if packetCount >= delayLength {
					histTargetT := poly.NewTensorFromSlice(targetBuffer[delayLength-1], 1, 4)
					poly.SystolicApplyTargetProp(net, sysState, histTargetT, LearningRate)
				}
				if currentWindow < 15 {
					result.Windows[currentWindow].BlockedMs += int(time.Since(blockStart).Milliseconds())
				}
			}
		}

		updateEnvironment(env)
		packetCount++
	}

	// Finalize last window
	if currentWindow < 15 && result.Windows[currentWindow].Outputs > 0 {
		winElapsed := time.Since(winStart)
		result.Windows[currentWindow].OutputsPerSec = result.Windows[currentWindow].Outputs
		result.Windows[currentWindow].Accuracy = float64(result.Windows[currentWindow].Correct) / float64(result.Windows[currentWindow].Outputs) * 100
		result.Windows[currentWindow].AvailMs = int(winElapsed.Milliseconds()) - result.Windows[currentWindow].BlockedMs
	}

	if len(result.Windows) > 5 {
		result.PreChangeAccuracy = result.Windows[4].Accuracy
		result.PostChange1Accuracy = result.Windows[5].Accuracy
		result.AdaptTime1 = -1
		for i := 5; i < 10 && i < len(result.Windows); i++ {
			if result.Windows[i].Accuracy >= 50 {
				result.AdaptTime1 = i - 5
				break
			}
		}
	}

	if len(result.Windows) > 10 {
		result.PostChange2Accuracy = result.Windows[10].Accuracy
		result.AdaptTime2 = -1
		for i := 10; i < 15 && i < len(result.Windows); i++ {
			if result.Windows[i].Accuracy >= 50 {
				result.AdaptTime2 = i - 10
				break
			}
		}
	}

	for _, w := range result.Windows {
		result.TotalAvailMs += w.AvailMs
		result.TotalBlockedMs += w.BlockedMs
	}

	return result
}

func getObservation(env *Environment) []float32 {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(relX*relX + relY*relY)))

	return []float32{
		env.AgentPos[0], env.AgentPos[1],
		env.TargetPos[0], env.TargetPos[1],
		relX, relY,
		dist,
		float32(env.Task),
	}
}

func getOptimalAction(env *Environment) int {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]

	if env.Task == 0 { // Chase - move towards
		if abs(relX) > abs(relY) {
			if relX > 0 {
				return 3 // right
			}
			return 2 // left
		}
		if relY > 0 {
			return 0 // up
		}
		return 1 // down
	} else { // Avoid - move away
		if abs(relX) > abs(relY) {
			if relX > 0 {
				return 2 // left (away)
			}
			return 3 // right (away)
		}
		if relY > 0 {
			return 1 // down (away)
		}
		return 0 // up (away)
	}
}

func executeAction(env *Environment, action int) {
	speed := float32(0.02)
	moves := [][2]float32{{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}}
	if action >= 0 && action < 4 {
		env.AgentPos[0] = clamp(env.AgentPos[0]+moves[action][0], 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moves[action][1], 0, 1)
	}
}

func updateEnvironment(env *Environment) {
	env.TargetPos[0] += (rand.Float32() - 0.5) * 0.01
	env.TargetPos[1] += (rand.Float32() - 0.5) * 0.01
	env.TargetPos[0] = clamp(env.TargetPos[0], 0.1, 0.9)
	env.TargetPos[1] = clamp(env.TargetPos[1], 0.1, 0.9)
}

func argmax(s []float32) int {
	if len(s) == 0 {
		return 0
	}
	maxI, maxV := 0, s[0]
	for i, v := range s {
		if v > maxV {
			maxV, maxI = v, i
		}
	}
	return maxI
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}


func printAdaptationTimeline(results map[string]*AdaptationResult, modes []Mode, configs []string) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ACCURACY OVER TIME (per 1-second window)                                                   ║")
	fmt.Println("║                   [0-5s: CHASE]    │    [5-10s: AVOID!]    │    [10-15s: CHASE]                                                 ║")
	fmt.Println("╠══════════════════════════╦════╦════╦════╦════╦════║════╦════╦════╦════╦════║════╦════╦════╦════╦════╗")
	fmt.Println("║ Mode                     ║ 1s ║ 2s ║ 3s ║ 4s ║ 5s ║ 6s ║ 7s ║ 8s ║ 9s ║10s ║11s ║12s ║13s ║14s ║15s ║")
	fmt.Println("╠══════════════════════════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╣")

	for _, mode := range modes {
		for _, cfg := range configs {
			runName := fmt.Sprintf("%s-%s", modeNames[mode], cfg)
			r := results[runName]
			fmt.Printf("║ %-24s ║", runName)
			for i := 0; i < 15; i++ {
				if i < len(r.Windows) {
					acc := r.Windows[i].Accuracy
					fmt.Printf(" %2.0f%%║", acc)
				} else {
					fmt.Printf("  -- ║")
				}
			}
			fmt.Println()
		}
	}
	fmt.Println("╚══════════════════════════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╝")
	fmt.Println("                              ↑ TASK CHANGE ↑                    ↑ TASK CHANGE ↑")
}

func printAdaptationSummary(results map[string]*AdaptationResult, modes []Mode, configs []string) {
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ADAPTATION SUMMARY                                                               ║")
	fmt.Println("╠══════════════════════════╦═══════════════╦═════════════════════╦═════════════════════╦═════════╦═════════╦══════════╣")
	fmt.Println("║ Mode                     ║ Total Outputs ║ 1st Change Adapt    ║ 2nd Change Adapt    ║ Avg Acc ║ Avail % ║ Peak Lat ║")
	fmt.Println("║                          ║ (actions/15s) ║ Before→After(delay) ║ Before→After(delay) ║         ║         ║          ║")
	fmt.Println("╠══════════════════════════╬═══════════════╬═════════════════════╬═════════════════════╬═════════╬═════════╬══════════╣")

	for _, mode := range modes {
		for _, cfg := range configs {
			runName := fmt.Sprintf("%s-%s", modeNames[mode], cfg)
			r := results[runName]
			avgAcc := float64(0)
			for _, w := range r.Windows {
				avgAcc += w.Accuracy
			}
			avgAcc /= float64(len(r.Windows))
			if len(r.Windows) == 0 { continue }

			adapt1 := "N/A"
			if r.AdaptTime1 >= 0 {
				adapt1 = fmt.Sprintf("%ds", r.AdaptTime1)
			}
			adapt2 := "N/A"
			if r.AdaptTime2 >= 0 {
				adapt2 = fmt.Sprintf("%ds", r.AdaptTime2)
			}
			
			availPct := float64(r.TotalAvailMs) / float64(r.TotalAvailMs+r.TotalBlockedMs+1) * 100

			fmt.Printf("║ %-24s ║ %13d ║ %3.0f%%→%3.0f%% (%3s)  ║ %3.0f%%→%3.0f%% (%3s)  ║  %5.1f%% ║  %5.1f%% ║  %6.1fµs║\n",
				runName,
				r.TotalOutputs,
				r.PreChangeAccuracy, r.PostChange1Accuracy, adapt1,
				r.Windows[9].Accuracy, r.PostChange2Accuracy, adapt2,
				avgAcc,
				availPct,
				float64(r.PeakLatency.Microseconds())/1000.0) //ms
		}
	}

	fmt.Println("╚══════════════════════════╩═══════════════╩═════════════════════╩═════════════════════╩═════════╩═════════╩══════════╝")
}
