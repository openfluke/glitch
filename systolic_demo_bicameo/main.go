package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/poly"
)

// ═══════════════════════════════════════════════════════════════════════════════
// SINE WAVE ADAPTATION BENCHMARK (MEGA-PERMUTATION EDITION)
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputSize  = 10
	HiddenSize = 32
	OutputSize = 1
	NumLayers  = 3

	LearningRate      = float32(0.01)
	InitScale         = float32(0.5)
	AccuracyThreshold = 0.05

	SinePoints     = 100
	SineResolution = 0.1

	TestDuration   = 10 * time.Second
	WindowDuration = 50 * time.Millisecond
	SwitchInterval = 2500 * time.Millisecond
	TrainInterval  = 10 * time.Millisecond
	SyncInterval   = 10
)

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeTargetProp
	ModeTargetPropChain
	ModeSystolic
	ModeSystolicChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:        "NormalBP",
	ModeTargetProp:      "TargetProp",
	ModeTargetPropChain: "TargetProp+Chain",
	ModeSystolic:        "Systolic",
	ModeSystolicChain:   "Systolic+Chain",
}

type LayerModeKey struct {
	Mode TrainingMode
	Dual bool // True = Bicameral, False = Single
}

func (k LayerModeKey) String() string {
	config := "Single"
	if k.Dual { config = "Bicameral" }
	return fmt.Sprintf("%s-%s", modeNames[k.Mode], config)
}

type TimeWindow struct {
	TimeMs        int     `json:"timeMs"`
	Outputs       int     `json:"outputs"`
	TotalAccuracy float64 `json:"totalAccuracy"`
	Accuracy      float64 `json:"accuracy"`
	FreqSwitches  int     `json:"freqSwitches"`
	MaxLatencyMs  float64 `json:"maxLatencyMs"`
	AvailableMs   float64 `json:"availableMs"`
	BlockedMs     float64 `json:"blockedMs"`
}

type ModeResult struct {
	Name             string
	Mode             string
	Config           string
	Windows          []TimeWindow
	TotalOutputs     int
	TotalFreqSwitch  int
	TrainTimeSec     float64
	AvgTrainAccuracy float64
	Stability        float64
	Consistency      float64
	ThroughputPerSec float64
	Score            float64
	AvailabilityPct  float64
	TotalBlockedMs   float64
	AvgLatencyMs     float64
	MaxLatencyMs     float64
	ZeroOutputWindows int
}

type TrainPacket struct {
	Key    LayerModeKey
	Input  *poly.Tensor[float32]
	Target float32
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🌊 TEST 41: SINE WAVE ADAPTATION BENCHMARK (5 MODES x 2 CONFIGS)                  ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   TRAINING: Sin(1x) → Sin(2x) → Sin(3x) → Sin(4x) (switch every 2.5 seconds)        ║")
	fmt.Println("║   Track PREDICTION ACCURACY % every 50ms!                                           ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	frequencies := []float64{1.0, 2.0, 3.0, 4.0}
	allInputs := make([][][]float32, len(frequencies))
	allTargets := make([][]float32, len(frequencies))

	for i, freq := range frequencies {
		sineData := generateSineWave(freq)
		allInputs[i], allTargets[i] = createSamples(sineData)
	}

	modes := []TrainingMode{
		ModeNormalBP,
		ModeTargetProp,
		ModeTargetPropChain,
		ModeSystolic,
		ModeSystolicChain,
	}

	var keys []LayerModeKey
	leftNets := make(map[LayerModeKey]*poly.VolumetricNetwork)
	rightNets := make(map[LayerModeKey]*poly.VolumetricNetwork)
	results := make(map[LayerModeKey]*ModeResult)

	for _, m := range modes {
		for _, dual := range []bool{false, true} {
			key := LayerModeKey{m, dual}
			keys = append(keys, key)

			config := "Single"
			if dual { config = "Bicameral" }

			isSys := (m == ModeSystolic || m == ModeSystolicChain)
			leftNets[key] = createNetwork(isSys)
			if dual {
				rightNets[key] = createNetwork(isSys)
				DeepMirror(leftNets[key], rightNets[key])
			}

			numWindows := int(TestDuration / WindowDuration)
			results[key] = &ModeResult{
				Name: key.String(), Mode: modeNames[m], Config: config,
				Windows: make([]TimeWindow, numWindows),
			}
			for i := range results[key].Windows {
				results[key].Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
			}
		}
	}

	trainChan := make(chan TrainPacket, 50000)

	var mu sync.Mutex
	var bgWg sync.WaitGroup
	bgWg.Add(1)
	go func() {
		defer bgWg.Done()
		trainStep := make(map[LayerModeKey]int)
		
		bgSysStates := make(map[LayerModeKey]*poly.SystolicState[float32])
		bgTPStates := make(map[LayerModeKey]*poly.TargetPropState[float32])
		bgTargetBuffers := make(map[LayerModeKey][]float32)

		for p := range trainChan {
			k := p.Key
			net := rightNets[k]
			tick := trainStep[k]

			targetT := poly.NewTensorFromSlice([]float32{p.Target}, 1, OutputSize)

			if k.Mode == ModeNormalBP {
				poly.Train(net, []poly.TrainingBatch[float32]{{Input: p.Input, Target: targetT}}, &poly.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse", Verbose: false})
			} else if k.Mode == ModeTargetProp || k.Mode == ModeTargetPropChain {
				tp, ok := bgTPStates[k]
				if !ok {
					cfg := poly.DefaultTargetPropConfig()
					cfg.UseChainRule = (k.Mode == ModeTargetPropChain)
					tp = poly.NewTargetPropState[float32](net, cfg)
					bgTPStates[k] = tp
				}
				poly.TargetPropForward(net, tp, p.Input)
				poly.TargetPropBackward(net, tp, targetT)
				tp.CalculateLinkBudgets()
				poly.ApplyTargetPropGaps(net, tp, LearningRate)
			} else if k.Mode == ModeSystolic || k.Mode == ModeSystolicChain {
				sys, ok := bgSysStates[k]
				if !ok {
					sys = poly.NewSystolicState[float32](net)
					cfg := poly.DefaultTargetPropConfig()
					cfg.UseChainRule = (k.Mode == ModeSystolicChain)
					sys.TPState = poly.NewTargetPropState[float32](net, cfg)
					bgSysStates[k] = sys
					bgTargetBuffers[k] = make([]float32, NumLayers+1)
				}
				
				delayLength := NumLayers + 1
				historicalTarget := bgTargetBuffers[k][tick % delayLength]
				bgTargetBuffers[k][tick % delayLength] = p.Target
				
				sys.SetInput(p.Input)
				poly.SystolicForward(net, sys, false)

				if tick >= delayLength {
					histTargetT := poly.NewTensorFromSlice([]float32{historicalTarget}, 1, OutputSize)
					poly.SystolicApplyTargetProp(net, sys, histTargetT, LearningRate)
				}
			}

			trainStep[k]++
			if trainStep[k]%SyncInterval == 0 {
				mu.Lock()
				DeepMirror(net, leftNets[k])
				mu.Unlock()
			}
		}
	}()

	var wg sync.WaitGroup
	resultChan := make(chan *ModeResult, len(keys))

	startInit := time.Now()
	for _, k := range keys {
		wg.Add(1)
		go func(key LayerModeKey) {
			defer wg.Done()
			runSineWaveWorker(key, leftNets[key], allInputs, allTargets, frequencies, trainChan, resultChan, results[key])
		}(k)
	}

	fmt.Printf("\n🚀 Networks created in %v. Starting Parallel Benchmark...\n", time.Since(startInit))
	
	wg.Wait()
	close(resultChan)
	close(trainChan)
	bgWg.Wait() // Flush background

	finalResults := make(map[LayerModeKey]*ModeResult)
	for res := range resultChan {
		for _, k := range keys {
			if k.String() == res.Name {
				finalResults[k] = res
				break
			}
		}
	}

	printTimeline(finalResults, keys)
	printSummary(finalResults, keys)
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

func generateSineWave(freqMultiplier float64) []float64 {
	data := make([]float64, SinePoints)
	for i := 0; i < SinePoints; i++ {
		x := float64(i) * SineResolution
		data[i] = math.Sin(freqMultiplier * x)
	}
	return data
}

func createSamples(data []float64) (inputs [][]float32, targets []float32) {
	numSamples := len(data) - InputSize
	inputs = make([][]float32, numSamples)
	targets = make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, InputSize)
		for j := 0; j < InputSize; j++ {
			input[j] = float32((data[i+j] + 1.0) / 2.0)
		}
		inputs[i] = input
		targets[i] = float32((data[i+InputSize] + 1.0) / 2.0)
	}
	return inputs, targets
}

func createNetwork(isSystolic bool) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, NumLayers+1)
	net.UseTiling = isSystolic

	l0 := net.GetLayer(0, 0, 0, 0)
	l0.IsDisabled = true

	l1 := net.GetLayer(0, 0, 0, 1)
	l1.Type = poly.LayerDense
	l1.InputHeight = InputSize
	l1.OutputHeight = HiddenSize
	l1.WeightStore = poly.NewWeightStore(InputSize * HiddenSize)
	l1.WeightStore.Randomize(time.Now().UnixNano(), InitScale)
	l1.Activation = poly.ActivationTanh

	l2 := net.GetLayer(0, 0, 0, 2)
	l2.Type = poly.LayerDense
	l2.InputHeight = HiddenSize
	l2.OutputHeight = HiddenSize
	l2.WeightStore = poly.NewWeightStore(HiddenSize * HiddenSize)
	l2.WeightStore.Randomize(time.Now().UnixNano(), InitScale)
	l2.Activation = poly.ActivationTanh

	l3 := net.GetLayer(0, 0, 0, 3)
	l3.Type = poly.LayerDense
	l3.InputHeight = HiddenSize
	l3.OutputHeight = OutputSize
	l3.WeightStore = poly.NewWeightStore(HiddenSize * OutputSize)
	l3.WeightStore.Randomize(time.Now().UnixNano(), InitScale)
	l3.Activation = poly.ActivationSigmoid

	return net
}

func runSineWaveWorker(key LayerModeKey, net *poly.VolumetricNetwork, allInputs [][][]float32, allTargets [][]float32, frequencies []float64, trainChan chan<- TrainPacket, resultChan chan<- *ModeResult, result *ModeResult) {
	mode := key.Mode
	isBicameral := key.Dual

	numWindows := len(result.Windows)
	
	var sysState *poly.SystolicState[float32]
	var tpState *poly.TargetPropState[float32]
	
	if mode == ModeSystolic || mode == ModeSystolicChain {
		sysState = poly.NewSystolicState[float32](net)
		cfg := poly.DefaultTargetPropConfig()
		cfg.UseChainRule = (mode == ModeSystolicChain)
		sysState.TPState = poly.NewTargetPropState[float32](net, cfg)
	} else if mode == ModeTargetProp || mode == ModeTargetPropChain {
		cfg := poly.DefaultTargetPropConfig()
		cfg.UseChainRule = (mode == ModeTargetPropChain)
		tpState = poly.NewTargetPropState[float32](net, cfg)
	}

	type TrainingSample struct {
		Input  *poly.Tensor[float32]
		Target *poly.Tensor[float32]
	}
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	currentFreqIdx := 0
	lastSwitchTime := start

	lastOutputTime := time.Now()
	var totalBlockedTime time.Duration
	windowStartTime := time.Now()

	delayLength := NumLayers + 1
	targetBuffer := make([]float32, delayLength)
	packetCount := 0

	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)

		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			if currentWindow < numWindows {
				windowElapsed := time.Since(windowStartTime).Seconds() * 1000
				result.Windows[currentWindow].AvailableMs = windowElapsed - result.Windows[currentWindow].BlockedMs
			}
			currentWindow = newWindow
			windowStartTime = time.Now()
		}

		if time.Since(lastSwitchTime) >= SwitchInterval && currentFreqIdx < len(frequencies)-1 {
			currentFreqIdx++
			lastSwitchTime = time.Now()
			result.TotalFreqSwitch++
			if currentWindow < numWindows {
				result.Windows[currentWindow].FreqSwitches++
			}
		}

		inputs := allInputs[currentFreqIdx]
		targets := allTargets[currentFreqIdx]

		inData := inputs[sampleIdx%len(inputs)]
		targetData := targets[sampleIdx%len(targets)]
		sampleIdx++

		input := poly.NewTensorFromSlice(inData, 1, InputSize)
		target := poly.NewTensorFromSlice([]float32{targetData}, 1, OutputSize)

		var output *poly.Tensor[float32]
		var activeTargetVal float32 = targetData

		if mode == ModeNormalBP {
			output, _, _ = poly.ForwardPolymorphic(net, input)
		} else if mode == ModeTargetProp || mode == ModeTargetPropChain {
			poly.TargetPropForward(net, tpState, input)
			outL := tpState.ForwardActs[NumLayers] // The final output is recorded here
			if outL != nil && len(outL.Data) > 0 {
				output = outL
			}
		} else if mode == ModeSystolic || mode == ModeSystolicChain {
			targetBuffer[packetCount % delayLength] = targetData

			sysState.SetInput(input)
			poly.SystolicForward(net, sysState, false)
			
			if packetCount >= delayLength {
				outL := sysState.LayerData[3] 
				if outL != nil && len(outL.Data) > 0 {
					output = outL
				}
				activeTargetVal = targetBuffer[(packetCount+1) % delayLength]
			}
		}

		if output != nil && len(output.Data) > 0 {
			pred := output.Data[0]
			sampleAcc := 0.0
			if math.Abs(float64(pred-activeTargetVal)) < AccuracyThreshold {
				sampleAcc = 100.0
			}

			if currentWindow < numWindows {
				latencyMs := time.Since(lastOutputTime).Seconds() * 1000
				if latencyMs > result.Windows[currentWindow].MaxLatencyMs {
					result.Windows[currentWindow].MaxLatencyMs = latencyMs
				}
				lastOutputTime = time.Now()
				result.Windows[currentWindow].Outputs++
				result.Windows[currentWindow].TotalAccuracy += sampleAcc
				result.TotalOutputs++
			}
		}

		if isBicameral {
			select {
			case trainChan <- TrainPacket{Key: key, Input: input, Target: activeTargetVal}:
			default:
				if currentWindow < numWindows {
					result.Windows[currentWindow].BlockedMs += 1.0
				}
				totalBlockedTime += time.Millisecond
			}
		} else {
			if mode == ModeNormalBP {
				trainBatch = append(trainBatch, TrainingSample{Input: input, Target: target})
				if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
					batches := make([]poly.TrainingBatch[float32], len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = poly.TrainingBatch[float32]{Input: s.Input, Target: s.Target}
					}
					trainStart := time.Now()
					
					poly.Train(net, batches, &poly.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse", Verbose: false})
					
					blockDuration := time.Since(trainStart)
					totalBlockedTime += blockDuration
					if currentWindow < numWindows {
						result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
					}
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			} else if mode == ModeTargetProp || mode == ModeTargetPropChain {
				trainStart := time.Now()
				poly.TargetPropBackward(net, tpState, target)
				tpState.CalculateLinkBudgets()
				poly.ApplyTargetPropGaps(net, tpState, LearningRate)
				
				blockDuration := time.Since(trainStart)
				totalBlockedTime += blockDuration
				if currentWindow < numWindows {
					result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
				}
			} else if mode == ModeSystolic || mode == ModeSystolicChain {
				trainStart := time.Now()
				if packetCount >= delayLength {
					targetTensor := poly.NewTensorFromSlice([]float32{activeTargetVal}, 1, OutputSize)
					poly.SystolicApplyTargetProp(net, sysState, targetTensor, LearningRate)
				}
				blockDuration := time.Since(trainStart)
				totalBlockedTime += blockDuration
				if currentWindow < numWindows {
					result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
				}
			}
		}

		packetCount++
		// Throttle slightly to simulate native I/O limits and prevent goroutine starvation
		time.Sleep(50 * time.Microsecond)
	}

	for i := range result.Windows {
		if result.Windows[i].Outputs > 0 {
			result.Windows[i].Accuracy = result.Windows[i].TotalAccuracy / float64(result.Windows[i].Outputs)
		}
		windowDurationMs := WindowDuration.Seconds() * 1000
		result.Windows[i].AvailableMs = windowDurationMs - result.Windows[i].BlockedMs
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	calculateSummaryMetrics(result)

	resultChan <- result
}

func calculateSummaryMetrics(result *ModeResult) {
	sum := 0.0
	for _, w := range result.Windows {
		sum += w.Accuracy
	}
	result.AvgTrainAccuracy = sum / float64(len(result.Windows))

	variance := 0.0
	for _, w := range result.Windows {
		diff := w.Accuracy - result.AvgTrainAccuracy
		variance += diff * diff
	}
	variance /= float64(len(result.Windows))
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	const consistencyThreshold = 10.0
	aboveThreshold := 0
	for _, w := range result.Windows {
		if w.Accuracy >= consistencyThreshold {
			aboveThreshold++
		}
	}
	result.Consistency = float64(aboveThreshold) / float64(len(result.Windows)) * 100

	result.ThroughputPerSec = float64(result.TotalOutputs) / result.TrainTimeSec

	totalTimeMs := result.TrainTimeSec * 1000
	result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	if result.AvailabilityPct < 0 { result.AvailabilityPct = 0 }

	result.Score = (result.ThroughputPerSec * result.AvailabilityPct * result.AvgTrainAccuracy) / 10000

	latencySum := 0.0
	result.MaxLatencyMs = 0
	for _, w := range result.Windows {
		latencySum += w.MaxLatencyMs
		if w.MaxLatencyMs > result.MaxLatencyMs {
			result.MaxLatencyMs = w.MaxLatencyMs
		}
		if w.Outputs == 0 {
			result.ZeroOutputWindows++
		}
	}
	result.AvgLatencyMs = latencySum / float64(len(result.Windows))
}

func printTimeline(results map[LayerModeKey]*ModeResult, keys []LayerModeKey) {
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           PREDICTION ACCURACY % (50ms windows) — Sin(1x)→Sin(2x)→Sin(3x)→Sin(4x) switching every 2.5s                                               ║")
	fmt.Println("╠══════════════════════════╦═════════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════╣")
	fmt.Printf("║ Mode                     ║")
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds  ", i+1)
	}
	fmt.Printf("║ Avg   ║ Score    ║\n")
	fmt.Println("╠══════════════════════════╬═════════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════╣")

	for _, k := range keys {
		r, ok := results[k]
		if !ok { continue }
		fmt.Printf("║ %-24s ║", r.Name)

		for sec := 0; sec < 10; sec++ {
			avgAcc := 0.0
			count := 0
			for w := sec * 20; w < (sec+1)*20 && w < len(r.Windows); w++ {
				avgAcc += r.Windows[w].Accuracy
				count++
			}
			if count > 0 { avgAcc /= float64(count) }
			fmt.Printf(" %3.0f%%", avgAcc)
		}
		fmt.Printf(" ║ %3.0f%% ║ %6.0f ║\n", r.AvgTrainAccuracy, r.Score)
	}
	fmt.Println("╚══════════════════════════╩═════════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════╝")
	
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           OUTPUTS PER SECOND — Shows throughput gaps when batch training blocks inference                                                           ║")
	fmt.Println("╠══════════════════════════╦═════════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════╣")
	fmt.Printf("║ Mode                     ║")
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds  ", i+1)
	}
	fmt.Printf("║ Total ║ Avail%%  ║\n")
	fmt.Println("╠══════════════════════════╬═════════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════╣")

	for _, k := range keys {
		r, ok := results[k]
		if !ok { continue }
		fmt.Printf("║ %-24s ║", r.Name)

		for sec := 0; sec < 10; sec++ {
			totalOutputs := 0
			for w := sec * 20; w < (sec+1)*20 && w < len(r.Windows); w++ {
				totalOutputs += r.Windows[w].Outputs
			}
			fmt.Printf(" %4d", totalOutputs)
		}
		fmt.Printf(" ║ %5d ║ %5.1f%% ║\n", r.TotalOutputs, r.AvailabilityPct)
	}
	fmt.Println("╚══════════════════════════╩═════════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════╝")
}

func printSummary(results map[LayerModeKey]*ModeResult, keys []LayerModeKey) {
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                            🌊 SINE WAVE ADAPTATION SUMMARY 🌊                                                                   ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Mode                     │ Accuracy │ Stability │ Throughput │ Score   │ Avail %  │ Blocked(ms) │ Peak Lat │ Avg Lat │          Insight            ║")
	fmt.Println("║  ─────────────────────────┼──────────┼───────────┼────────────┼─────────┼──────────┼─────────────┼──────────┼─────────┼─────────────────────────────║")

	for _, k := range keys {
		r, ok := results[k]
		if !ok { continue }
		insight := "Always Available ✓"
		if r.Config == "Bicameral" { insight = "Shadow Network Training"}
		if k.Mode == ModeNormalBP && r.Config == "Single" { insight = "Blocked by Batching" }

		fmt.Printf("║  %-24s │  %5.1f%%  │   %5.1f%%  │ %7.0f Hz │ %7.0f │  %5.1f%%  │  %9.0f  │  %5.1fms │ %5.1fms  │ %-27s ║\n",
			r.Name, r.AvgTrainAccuracy, r.Stability, r.ThroughputPerSec, r.Score,
			r.AvailabilityPct, r.TotalBlockedMs, r.MaxLatencyMs, r.AvgLatencyMs, insight)
	}
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
