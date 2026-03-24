package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

// ============================================================================
// TEST HARNESS
// ============================================================================

type Scenario struct {
	Name        string
	Description string
	Build       func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) // meta, backprop
	Rules       []poly.MetaRule
	Glitch      func(meta, back *poly.VolumetricNetwork)
	GenData     func() ([]*poly.Tensor[float32], []float64)
	BackEpochs  int
	BackLR      float64
}

type Result struct {
	Name         string
	MetaBaseline float64
	MetaGlitched float64
	MetaHealed   float64
	MetaTime     time.Duration
	BackBaseline float64
	BackGlitched float64
	BackHealed   float64
	BackTime     time.Duration
	RulesFired   bool
}

func main() {
	rand.Seed(time.Now().UnixNano())
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("⏩ Skip all backpropagation training items? (1=yes / 0=no) [0]: ")
	skipAllRaw, _ := reader.ReadString('\n')
	skipAll := strings.TrimSpace(skipAllRaw) == "1"

	scenarios := []Scenario{
		scenario1_DenseGainDrift(),
		scenario2_DenseWeightExplosion(),
		scenario3_DenseVanishingSignal(),
		scenario4_DeepCascade64(),
		scenario5_DenseDeadNeurons(),
		scenario6_MultiRuleDefense(),
		scenario7_RNNCorruption(),
		scenario8_ParallelBranchFailure(),
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       LOOM SELF-HEALING NEURAL NETWORK TEST SUITE               ║")
	fmt.Println("║       8 Scenarios • Multiple Layer Types • CPU-Only             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	results := make([]Result, 0, len(scenarios))

	for i, s := range scenarios {
		fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("  SCENARIO %d: %s\n", i+1, s.Name)
		fmt.Printf("  %s\n", s.Description)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		r := runScenario(s, skipAll, reader)
		results = append(results, r)
	}

	// ── FINAL SUMMARY ──
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                            FINAL RESULTS SUMMARY                                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-28s │ %6s → %6s │ %12s │ %6s → %6s │ %12s ║\n",
		"SCENARIO", "GLITCH", "HEALED", "META TIME", "GLITCH", "HEALED", "BACK TIME")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")

	metaWins := 0
	for _, r := range results {
		winner := " "
		if r.MetaHealed > r.BackHealed {
			winner = "★"
			metaWins++
		} else if r.MetaHealed == r.BackHealed && r.MetaTime < r.BackTime {
			winner = "★"
			metaWins++
		}

		fmt.Printf("║%s%-28s │ %5.1f → %6.2f │ %12v │ %5.1f → %6.2f │ %12v ║\n",
			winner, truncStr(r.Name, 28),
			r.MetaGlitched, r.MetaHealed, fmtDur(r.MetaTime),
			r.BackGlitched, r.BackHealed, fmtDur(r.BackTime))
	}

	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Self-Healing wins: %d / %d scenarios                                                  ║\n", metaWins, len(results))
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")
}

func runScenario(s Scenario, skipAll bool, reader *bufio.Reader) Result {
	r := Result{Name: s.Name}

	netMeta, netBack := s.Build()
	inputs, expected := s.GenData()

	// Wrap meta network
	poly.WrapWithMetacognition(netMeta, s.Rules)

	// Baseline
	r.MetaBaseline = quickEval(netMeta, inputs, expected)
	r.BackBaseline = quickEval(netBack, inputs, expected)
	fmt.Printf("  [Baseline]  Meta: %.2f  |  Back: %.2f\n", r.MetaBaseline, r.BackBaseline)

	// Glitch
	s.Glitch(netMeta, netBack)
	r.MetaGlitched = quickEval(netMeta, inputs, expected)
	r.BackGlitched = quickEval(netBack, inputs, expected)
	fmt.Printf("  [Glitched]  Meta: %.2f  |  Back: %.2f\n", r.MetaGlitched, r.BackGlitched)

	// Meta repair (just one forward pass with strong signal)
	startMeta := time.Now()
	// Pick a non-zero input for reliable gain detection
	repairIdx := findNonZeroInput(inputs)
	poly.ForwardPolymorphic(netMeta, inputs[repairIdx])
	r.MetaTime = time.Since(startMeta)
	r.MetaHealed = quickEval(netMeta, inputs, expected)
	fmt.Printf("  [MetaHeal]  Score: %.2f  |  Time: %v\n", r.MetaHealed, r.MetaTime)

	// Backprop repair
	epochs := s.BackEpochs
	if epochs == 0 {
		epochs = 1
	}
	lr := s.BackLR
	if lr == 0 {
		lr = 0.0001
	}

	batches := make([]poly.TrainingBatch[float32], len(inputs))
	for i := range inputs {
		batches[i] = poly.TrainingBatch[float32]{
			Input:  inputs[i],
			Target: poly.NewTensorFromSlice([]float32{float32(expected[i])}, 1, 1),
		}
	}

	config := poly.DefaultTrainingConfig()
	config.Epochs = epochs
	config.LearningRate = float32(lr)
	config.Verbose = false
	config.Mode = poly.TrainingModeCPUNormal

	startBack := time.Now()
	if skipAll {
		fmt.Printf("  [BackHeal]  SKIPPED (Global)\n")
		r.BackHealed = 0
		r.BackTime = 0
	} else {
		_, _ = poly.Train(netBack, batches, config)
		r.BackTime = time.Since(startBack)
		r.BackHealed = quickEval(netBack, inputs, expected)
		fmt.Printf("  [BackHeal]  Score: %.2f  |  Time: %v  |  Epochs: %d\n", r.BackHealed, r.BackTime, epochs)
	}

	return r
}

// ============================================================================
// SCENARIO 1: Dense Gain Drift (The Proven Case)
// Layer: Dense | Glitch: +0.1 diagonal | Rule: GainDrift → ResetIdentity
// ============================================================================

func scenario1_DenseGainDrift() Scenario {
	dModel := 128
	numLayers := 32
	samples := 500

	return Scenario{
		Name:        "Dense Gain Drift",
		Description: "32 Dense identity layers, +10% diagonal gain in 4 layers",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			return buildDenseIdentity(dModel, numLayers), buildDenseIdentity(dModel, numLayers)
		},
		Rules: []poly.MetaRule{
			{Condition: poly.MetaCondGainDrift, Threshold: 0.05, Command: 101, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			for _, idx := range []int{5, 12, 18, 25} {
				glitchDiagonal(meta, idx, 0.1, true)
				glitchDiagonal(back, idx, 0.1, false)
			}
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.0001,
	}
}

// ============================================================================
// SCENARIO 2: Dense Weight Explosion
// Layer: Dense | Glitch: 50x weight scaling | Rule: StdAbove → MorphToRMSNorm
// ============================================================================

func scenario2_DenseWeightExplosion() Scenario {
	dModel := 64
	numLayers := 16
	samples := 500

	return Scenario{
		Name:        "Dense Weight Explosion",
		Description: "16 Dense layers, 3 layers get weights multiplied by 50x",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			return buildDenseIdentity(dModel, numLayers), buildDenseIdentity(dModel, numLayers)
		},
		Rules: []poly.MetaRule{
			// When output std explodes, normalize it
			{Condition: poly.MetaCondStdAbove, Threshold: 5.0, Command: 90, SelfOnly: true},
			// Also catch gain drift for cleanup
			{Condition: poly.MetaCondGainDrift, Threshold: 0.5, Command: 101, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			for _, idx := range []int{3, 8, 13} {
				glitchScale(meta, idx, 50.0, true)
				glitchScale(back, idx, 50.0, false)
			}
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.00001, // Small LR needed for exploded weights
	}
}

// ============================================================================
// SCENARIO 3: Dense Vanishing Signal
// Layer: Dense | Glitch: scale weights to 0.01 | Rule: GainDrift → ResetIdentity
// ============================================================================

func scenario3_DenseVanishingSignal() Scenario {
	dModel := 128
	numLayers := 16
	samples := 500

	return Scenario{
		Name:        "Dense Vanishing Signal",
		Description: "16 Dense layers, 4 layers get weights crushed to 1% (signal vanishes)",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			return buildDenseIdentity(dModel, numLayers), buildDenseIdentity(dModel, numLayers)
		},
		Rules: []poly.MetaRule{
			// Tight threshold catches even small shrinkage
			{Condition: poly.MetaCondGainDrift, Threshold: 0.02, Command: 101, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			for _, idx := range []int{2, 6, 10, 14} {
				// Scale diagonal from 1.0 to 0.01
				glitchSetDiagonal(meta, idx, 0.01, true)
				glitchSetDiagonal(back, idx, 0.01, false)
			}
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.001,
	}
}

// ============================================================================
// SCENARIO 4: Deep 64-Layer Cascade
// Layer: Dense | Glitch: tiny drift in 20 layers compounds catastrophically
// ============================================================================

func scenario4_DeepCascade64() Scenario {
	dModel := 64
	numLayers := 64
	samples := 500

	return Scenario{
		Name:        "Deep 64-Layer Cascade",
		Description: "64 Dense layers, +3% drift in 20 layers (compounds to ~80% total error)",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			return buildDenseIdentity(dModel, numLayers), buildDenseIdentity(dModel, numLayers)
		},
		Rules: []poly.MetaRule{
			{Condition: poly.MetaCondGainDrift, Threshold: 0.02, Command: 101, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			// Tiny drift in many layers
			for i := 0; i < 64; i += 3 {
				glitchDiagonal(meta, i, 0.03, true)
				glitchDiagonal(back, i, 0.03, false)
			}
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.00005,
	}
}

// ============================================================================
// SCENARIO 5: Dense Dead Neurons (ReLU Activation)
// Layer: Dense+ReLU | Glitch: negative bias kills activations
// Rule: ActiveBelow → ResetIdentity
// ============================================================================

func scenario5_DenseDeadNeurons() Scenario {
	dModel := 64
	numLayers := 8
	samples := 500

	return Scenario{
		Name:        "Dense Dead Neurons (ReLU)",
		Description: "8 Dense+ReLU layers, 3 layers get massive negative bias (all neurons die)",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			// Build with ReLU activations and explicit bias region
			buildFn := func() *poly.VolumetricNetwork {
				net := poly.NewVolumetricNetwork(1, 1, numLayers, 1)
				for i := 0; i < numLayers; i++ {
					l := net.GetLayer(0, 0, i, 0)
					l.Type = poly.LayerDense
					l.InputHeight = dModel
					l.OutputHeight = dModel
					if i == numLayers-1 {
						l.OutputHeight = 1
						l.Activation = poly.ActivationLinear
					} else {
						l.Activation = poly.ActivationReLU
					}
					total := l.InputHeight*l.OutputHeight + l.OutputHeight // weights + bias
					l.WeightStore = poly.NewWeightStore(total)
					l.WeightStore.Master = make([]float32, total)
					// Identity weights
					if l.OutputHeight == dModel {
						for j := 0; j < dModel; j++ {
							l.WeightStore.Master[j*dModel+j] = 1.0
						}
					} else {
						l.WeightStore.Master[0] = 1.0
					}
					// Bias region starts after weights, init to 0
				}
				return net
			}
			return buildFn(), buildFn()
		},
		Rules: []poly.MetaRule{
			// Detect gain collapse (dead neurons → output near zero)
			{Condition: poly.MetaCondGainDrift, Threshold: 0.1, Command: 101, SelfOnly: true},
			// Also detect max activation collapse
			{Condition: poly.MetaCondMaxAbove, Threshold: 1000.0, Command: 90, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			// Shift diagonal weights strongly negative so ReLU kills everything
			for _, idx := range []int{1, 3, 5} {
				glitchSetDiagonal(meta, idx, -5.0, true)
				glitchSetDiagonal(back, idx, -5.0, false)
			}
		},
		GenData: func() ([]*poly.Tensor[float32], []float64) {
			// Use only positive inputs so ReLU matters
			inputs := make([]*poly.Tensor[float32], samples)
			expected := make([]float64, samples)
			for i := 0; i < samples; i++ {
				val := math.Abs(math.Sin(float64(i)*0.1)) + 0.1 // Always positive
				inputs[i] = poly.NewTensorFromSlice(make([]float32, dModel), 1, dModel)
				for j := 0; j < dModel; j++ {
					inputs[i].Data[j] = float32(val)
				}
				expected[i] = val
			}
			return inputs, expected
		},
		BackEpochs: 5,
		BackLR:     0.0001,
	}
}

// ============================================================================
// SCENARIO 6: Multi-Rule Defense
// Layer: Dense | Glitch: mixed (some drift, some explosion, some vanish)
// Rules: GainDrift + StdAbove + MaxAbove — 3 rules cooperating
// ============================================================================

func scenario6_MultiRuleDefense() Scenario {
	dModel := 64
	numLayers := 24
	samples := 500

	return Scenario{
		Name:        "Multi-Rule Defense",
		Description: "24 Dense layers, 3 different corruption types, 3 heuristic rules",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			return buildDenseIdentity(dModel, numLayers), buildDenseIdentity(dModel, numLayers)
		},
		Rules: []poly.MetaRule{
			{Condition: poly.MetaCondGainDrift, Threshold: 0.05, Command: 101, SelfOnly: true},
			{Condition: poly.MetaCondStdAbove, Threshold: 10.0, Command: 90, SelfOnly: true},
			{Condition: poly.MetaCondMaxAbove, Threshold: 50.0, Command: 90, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			// Type 1: Gain drift
			for _, idx := range []int{3, 7} {
				glitchDiagonal(meta, idx, 0.15, true)
				glitchDiagonal(back, idx, 0.15, false)
			}
			// Type 2: Weight explosion
			for _, idx := range []int{11, 15} {
				glitchScale(meta, idx, 20.0, true)
				glitchScale(back, idx, 20.0, false)
			}
			// Type 3: Vanishing
			for _, idx := range []int{19, 22} {
				glitchSetDiagonal(meta, idx, 0.05, true)
				glitchSetDiagonal(back, idx, 0.05, false)
			}
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.00005,
	}
}

// ============================================================================
// SCENARIO 7: RNN Corruption
// Layer: RNN | Glitch: corrupt recurrent weights
// Rule: GainDrift → ResetIdentity (morphs RNN back to Dense)
// ============================================================================

func scenario7_RNNCorruption() Scenario {
	dModel := 64
	numLayers := 8
	samples := 500

	return Scenario{
		Name:        "RNN Hidden State Corruption",
		Description: "8 layers (6 RNN + output Dense), recurrent weights corrupted in 3 layers",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			buildFn := func() *poly.VolumetricNetwork {
				net := poly.NewVolumetricNetwork(1, 1, numLayers, 1)
				for i := 0; i < numLayers; i++ {
					l := net.GetLayer(0, 0, i, 0)

					if i == numLayers-1 {
						// Output layer: Dense projecting to scalar
						l.Type = poly.LayerDense
						l.InputHeight = dModel
						l.OutputHeight = 1
						l.Activation = poly.ActivationLinear
						l.WeightStore = poly.NewWeightStore(dModel)
						l.WeightStore.Master = make([]float32, dModel)
						l.WeightStore.Master[0] = 1.0
					} else {
						// RNN layer: Wx (d*d) + Wh (d*d) + bias (d)
						// Set Wx = Identity, Wh = 0, bias = 0 → pass-through
						l.Type = poly.LayerRNN
						l.InputHeight = dModel
						l.OutputHeight = dModel
						l.Activation = poly.ActivationLinear

						total := dModel*dModel + dModel*dModel + dModel
						l.WeightStore = poly.NewWeightStore(total)
						l.WeightStore.Master = make([]float32, total)
						// Wx = Identity (first d*d block)
						for j := 0; j < dModel; j++ {
							l.WeightStore.Master[j*dModel+j] = 1.0
						}
						// Wh = 0 (already zero), bias = 0 (already zero)
					}
				}
				return net
			}
			return buildFn(), buildFn()
		},
		Rules: []poly.MetaRule{
			// RNN corruption shows as gain drift through the Wx matrix
			{Condition: poly.MetaCondGainDrift, Threshold: 0.05, Command: 101, SelfOnly: true},
			// If signal explodes, normalize
			{Condition: poly.MetaCondStdAbove, Threshold: 10.0, Command: 90, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			// Corrupt the Wx (input) weights in the RNN layers
			for _, idx := range []int{1, 3, 5} {
				// For RNN: first d*d block is Wx
				glitchRNNInputWeights(meta, idx, dModel, 0.2, true)
				glitchRNNInputWeights(back, idx, dModel, 0.2, false)
			}
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.0001,
	}
}

// ============================================================================
// SCENARIO 8: Parallel Branch Failure
// Layer: Parallel (3 Dense branches, "add" combine)
// Glitch: one branch produces wildly wrong output
// Rule: GainDrift detects combined output distortion
// ============================================================================

func scenario8_ParallelBranchFailure() Scenario {
	dModel := 64
	samples := 500

	return Scenario{
		Name:        "Parallel Branch Failure",
		Description: "Parallel(3 branches, add) + Dense output. One branch corrupted.",
		Build: func() (*poly.VolumetricNetwork, *poly.VolumetricNetwork) {
			buildFn := func() *poly.VolumetricNetwork {
				// Structure: [Parallel(3 branches avg)] → [Dense output]
				net := poly.NewVolumetricNetwork(1, 1, 2, 1)

				// Layer 0: Parallel with 3 Dense branches
				par := net.GetLayer(0, 0, 0, 0)
				par.Type = poly.LayerParallel
				par.InputHeight = dModel
				par.OutputHeight = dModel
				par.CombineMode = "avg"

				// Each branch: identity Dense that contributes 1/3 of the signal
				par.ParallelBranches = make([]poly.VolumetricLayer, 3)
				for b := 0; b < 3; b++ {
					branch := &par.ParallelBranches[b]
					branch.Network = net
					branch.Type = poly.LayerDense
					branch.InputHeight = dModel
					branch.OutputHeight = dModel
					branch.Activation = poly.ActivationLinear
					branch.WeightStore = poly.NewWeightStore(dModel * dModel)
					branch.WeightStore.Master = make([]float32, dModel*dModel)
					for j := 0; j < dModel; j++ {
						branch.WeightStore.Master[j*dModel+j] = 1.0 // Identity
					}
				}

				// Layer 1: Dense output (dModel → 1)
				out := net.GetLayer(0, 0, 1, 0)
				out.Type = poly.LayerDense
				out.InputHeight = dModel
				out.OutputHeight = 1
				out.Activation = poly.ActivationLinear
				out.WeightStore = poly.NewWeightStore(dModel)
				out.WeightStore.Master = make([]float32, dModel)
				out.WeightStore.Master[0] = 1.0

				return net
			}
			return buildFn(), buildFn()
		},
		Rules: []poly.MetaRule{
			// The parallel layer as a whole will show gain drift
			{Condition: poly.MetaCondGainDrift, Threshold: 0.1, Command: 101, SelfOnly: true},
		},
		Glitch: func(meta, back *poly.VolumetricNetwork) {
			// Corrupt branch 1 (of 3) in the parallel layer
			corruptBranch := func(net *poly.VolumetricNetwork, isMeta bool) {
				par := net.GetLayer(0, 0, 0, 0)
				target := par
				if isMeta && par.Type == poly.LayerMetacognition && par.MetaObservedLayer != nil {
					target = par.MetaObservedLayer
				}
				if len(target.ParallelBranches) >= 2 {
					branch := &target.ParallelBranches[1]
					// Scale this branch's weights by 4x → the avg output becomes (1+4+1)/3 = 2x
					for j := range branch.WeightStore.Master {
						branch.WeightStore.Master[j] *= 4.0
					}
				}
			}
			corruptBranch(meta, true)
			corruptBranch(back, false)
		},
		GenData:    func() ([]*poly.Tensor[float32], []float64) { return genHarmonic(dModel, samples) },
		BackEpochs: 5,
		BackLR:     0.0001,
	}
}

// ============================================================================
// SHARED BUILDERS
// ============================================================================

func buildDenseIdentity(dModel, numLayers int) *poly.VolumetricNetwork {
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

// ============================================================================
// SHARED DATA GENERATORS
// ============================================================================

func genHarmonic(dModel, samples int) ([]*poly.Tensor[float32], []float64) {
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

// ============================================================================
// SHARED GLITCH FUNCTIONS
// ============================================================================

// glitchDiagonal adds a value to the diagonal of a Dense layer's weight matrix
func glitchDiagonal(net *poly.VolumetricNetwork, idx int, delta float32, isMeta bool) {
	target := resolveTarget(net, idx, isMeta)
	if target == nil {
		return
	}
	d := target.InputHeight
	for j := 0; j < d; j++ {
		pos := j*d + j
		if pos < len(target.WeightStore.Master) {
			target.WeightStore.Master[pos] += delta
		}
	}
}

// glitchScale multiplies all weights by a factor
func glitchScale(net *poly.VolumetricNetwork, idx int, factor float32, isMeta bool) {
	target := resolveTarget(net, idx, isMeta)
	if target == nil {
		return
	}
	for j := range target.WeightStore.Master {
		target.WeightStore.Master[j] *= factor
	}
}

// glitchSetDiagonal overwrites the diagonal to a specific value (and zeros off-diagonal)
func glitchSetDiagonal(net *poly.VolumetricNetwork, idx int, val float32, isMeta bool) {
	target := resolveTarget(net, idx, isMeta)
	if target == nil {
		return
	}
	d := target.InputHeight
	if d*d > len(target.WeightStore.Master) {
		return
	}
	// Zero everything
	for j := range target.WeightStore.Master[:d*d] {
		target.WeightStore.Master[j] = 0
	}
	// Set diagonal
	for j := 0; j < d; j++ {
		target.WeightStore.Master[j*d+j] = val
	}
}

// glitchRNNInputWeights corrupts the Wx (input-to-hidden) weights of an RNN layer
func glitchRNNInputWeights(net *poly.VolumetricNetwork, idx, dModel int, delta float32, isMeta bool) {
	target := resolveTarget(net, idx, isMeta)
	if target == nil {
		return
	}
	// RNN weight layout: [Wx: d*d] [Wh: d*d] [bias: d]
	// Corrupt Wx diagonal
	for j := 0; j < dModel; j++ {
		pos := j*dModel + j
		if pos < len(target.WeightStore.Master) {
			target.WeightStore.Master[pos] += delta
		}
	}
}

func resolveTarget(net *poly.VolumetricNetwork, idx int, isMeta bool) *poly.VolumetricLayer {
	l := net.GetLayer(0, 0, idx, 0)
	if l == nil {
		return nil
	}
	if isMeta && l.Type == poly.LayerMetacognition && l.MetaObservedLayer != nil {
		return l.MetaObservedLayer
	}
	return l
}

// ============================================================================
// UTILITIES
// ============================================================================

func quickEval(net *poly.VolumetricNetwork, inputs []*poly.Tensor[float32], expected []float64) float64 {
	metrics, _ := poly.EvaluateNetworkPolymorphic(net, inputs, expected)
	return metrics.Score
}

func findNonZeroInput(inputs []*poly.Tensor[float32]) int {
	for i, inp := range inputs {
		if len(inp.Data) > 0 && math.Abs(float64(inp.Data[0])) > 0.1 {
			return i
		}
	}
	return len(inputs) / 2 // fallback
}

func truncStr(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen-1] + "…"
	}
	return s + strings.Repeat(" ", maxLen-len(s))
}

func fmtDur(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%.1fµs", float64(d.Microseconds()))
	} else if d < time.Second {
		return fmt.Sprintf("%.1fms", float64(d.Microseconds())/1000)
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}
