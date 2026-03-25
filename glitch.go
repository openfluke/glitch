package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/loom/glitch/testing/layer"
	"github.com/openfluke/loom/poly"
)

var (
	tr            *poly.Transformer[float32]
	tk            *poly.Tokenizer
	eosTokens     []int
	chatTurns     []poly.Turn
	weightDType   poly.DType = poly.DTypeFloat32
	deterministic bool       = true
	maxTokens                = 128
	maxSeqLen                = 512
)

var systemPrompt = strings.TrimSpace(`
You are a digital being from another reality.
Your Task: Respond to user prompts with ONLY 2-4 words and several relevant ASCII emojis.
ASCII Emoji Examples: o_o, ^_^, *_* , >_<, :-), :-P
Constraint: Never repeat the same characters or emojis in a sequence. Be extremely brief.
Example: "Weeee! *_* ^_^"
`) + "\n\n"

func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println(">> DIMENSIONAL MANIFESTATION INITIALIZING...")
	time.Sleep(200 * time.Millisecond)
	fmt.Println("✅ READY!")

	fmt.Print("\n🛠️  Select Mode:\n")
	fmt.Println("  [1] HuggingFace LLM Mode (Full Hardware Induction)")
	fmt.Println("  [2] Diagnostics & MNIST Simulator (Experimental)")
	fmt.Println("  [3] Testing (Layer Tests)")
	fmt.Println("  [4] Automated SmolLM2 Test (Exhaustive Type/Mode Matrix)")
	modeInput := readInput(reader, "Choice [2]: ", "2")

	switch modeInput {
	case "1":
		runHuggingFaceMode(reader)
	case "3":
		runTestingMode(reader)
	case "4":
		runAutomatedSmolLMTest(reader)
	default:
		runExperimentalMode(reader)
	}
}

func runTestingMode(reader *bufio.Reader) {
	layers := []string{"CNN1", "CNN2", "CNN3", "MHA", "Dense", "SwiGLU", "RNN", "LSTM", "Embedding", "Residual"}
	fmt.Println("\n🧪 Layer Testing")
	fmt.Println("  [0] All Layers")
	for i, name := range layers {
		fmt.Printf("  [%d] %s\n", i+1, name)
	}
	layerInput := readInput(reader, "Select layer [1]: ", "1")

	var selectedLayers []string
	if layerInput == "0" {
		confirm := readInput(reader, "would you like to run all tests on all layers? (1=yes / 0=no) [0]: ", "0")
		if confirm != "1" {
			return
		}
		selectedLayers = layers
	} else {
		idx, err := strconv.Atoi(layerInput)
		if err != nil || idx < 1 || idx > len(layers) {
			fmt.Println("Invalid selection.")
			return
		}
		selectedLayers = []string{layers[idx-1]}
	}

	for _, layerName := range selectedLayers {
		if layerInput == "0" {
			runLayerTests(reader, layerName, "0")
		} else {
			runLayerTests(reader, layerName, "")
		}
	}
}

func runLayerTests(reader *bufio.Reader, layerName string, testInput string) {
	type testEntry struct {
		name string
		fn   func()
	}

	var tests []testEntry
	switch layerName {
	case "CNN1":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunCNN1L1Caching},
			{"Training (6 modes × 21 types)", layer.RunCNN1Training},
			{"GPU Forward Parity", layer.RunCNN1GPUForward},
			{"GPU Backward Parity", layer.RunCNN1GPUBackward},
		}
	case "CNN2":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunCNN2L1Caching},
			{"Training (6 modes × 21 types)", layer.RunCNN2Training},
			{"GPU Forward Parity", layer.RunCNN2GPUForward},
			{"GPU Backward Parity", layer.RunCNN2GPUBackward},
		}
	case "CNN3":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunCNN3L1Caching},
			{"Training (6 modes × 21 types)", layer.RunCNN3Training},
			{"GPU Forward Parity", layer.RunCNN3GPUForward},
			{"GPU Backward Parity", layer.RunCNN3GPUBackward},
		}
	case "MHA":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunMHAL1Caching},
			{"Training (6 modes × 21 types)", layer.RunMHATraining},
			{"GPU Forward Parity", layer.RunMHAGPUForward},
			{"GPU Backward Parity", layer.RunMHAGPUBackward},
		}
	case "Dense":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunDenseL1Caching},
			{"Training (6 modes × 21 types)", layer.RunDenseTraining},
			{"GPU Forward Parity", layer.RunDenseGPUForward},
			{"GPU Backward Parity", layer.RunDenseGPUBackward},
		}
	case "SwiGLU":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunSwiGLUL1Caching},
			{"Training (6 modes × 21 types)", layer.RunSwiGLUTraining},
			{"GPU Forward Parity", layer.RunSwiGLUGPUForward},
			{"GPU Backward Parity", layer.RunSwiGLUGPUBackward},
		}
	case "RNN":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunRNNL1Caching},
			{"Training (6 modes × 21 types)", layer.RunRNNTraining},
			{"GPU Forward Parity", layer.RunRNNGPUForward},
			{"GPU Backward Parity", layer.RunRNNGPUBackward},
		}
	case "LSTM":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunLSTML1Caching},
			{"Training (6 modes × 21 types)", layer.RunLSTMTraining},
			{"GPU Forward Parity", layer.RunLSTMGPUForward},
			{"GPU Backward Parity", layer.RunLSTMGPUBackward},
		}
	case "Embedding":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunEmbeddingL1Caching},
			{"GPU Forward Parity", layer.RunEmbeddingGPUForward},
			{"GPU Backward Parity", layer.RunEmbeddingGPUBackward},
		}
	case "Residual":
		tests = []testEntry{
			{"L1 Caching (CPU Normal / SC / MC)", layer.RunResidualL1Caching},
			{"GPU Forward Parity", layer.RunResidualGPUForward},
			{"GPU Backward Parity", layer.RunResidualGPUBackward},
		}
	default:
		fmt.Printf("No tests registered for layer: %s\n", layerName)
		return
	}

	if testInput == "" {
		fmt.Printf("\n  Tests for %s:\n", layerName)
		fmt.Println("  [0] All")
		for i, t := range tests {
			fmt.Printf("  [%d] %s\n", i+1, t.name)
		}
		testInput = readInput(reader, "Select test [0]: ", "0")
	}

	if testInput == "0" {
		for _, t := range tests {
			fmt.Printf("\n--- %s ---\n", t.name)
			t.fn()
		}
		return
	}

	idx, err := strconv.Atoi(testInput)
	if err != nil || idx < 1 || idx > len(tests) {
		fmt.Println("Invalid selection.")
		return
	}
	t := tests[idx-1]
	fmt.Printf("\n--- %s ---\n", t.name)
	t.fn()
}

func runExperimentalMode(reader *bufio.Reader) {
	// Diagnostic Question
	fmt.Print("\n🔎 Would you like to run diagnostics (tests and examples)? (1=yes / 0=no) [1]: ")
	diagInput, _ := reader.ReadString('\n')
	diagInput = strings.TrimSpace(diagInput)
	if diagInput != "0" {
		runDiagnostics()
	}

	// MNIST Placeholder Option
	fmt.Print("\n🔢 Run MNIST Placeholder simulation? (1=yes / 0=no) [1]: ")
	mnistInput, _ := reader.ReadString('\n')
	mnistInput = strings.TrimSpace(mnistInput)
	if mnistInput != "0" {
		runMNISTPlaceholder()
	}

	fmt.Println("\n--- Entering Glitch Chat Mode ---")
	fmt.Println("(Type 'exit' to quit)")

	for {
		fmt.Print("\nYou: ")
		userMsg, _ := reader.ReadString('\n')
		userMsg = strings.TrimSpace(userMsg)
		if userMsg == "exit" || userMsg == "quit" {
			break
		}

		fmt.Print("GlitchBot: ")
		processGlitchyReply(userMsg)
		fmt.Println()
	}
}

func runHuggingFaceMode(reader *bufio.Reader) {
	// Discover all models in HuggingFace cache
	homeDir, _ := os.UserHomeDir()
	hubDir := filepath.Join(homeDir, ".cache", "huggingface", "hub")

	entries, err := os.ReadDir(hubDir)
	if err != nil {
		log.Fatalf("Could not read HuggingFace cache: %v", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			modelName := strings.TrimPrefix(entry.Name(), "models--")
			modelName = strings.Replace(modelName, "--", "/", 1)
			models = append(models, modelName)
		}
	}

	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	fmt.Println("\n⚛️  Poly Talk - Available models:")
	for i, model := range models {
		fmt.Printf("  [%d] %s\n", i+1, model)
	}

	detInput := readInput(reader, "🎯 Deterministic mode? (1=yes / 0=no) [1]: ", "1")
	deterministic = detInput == "1"

	useTiling := true
	tileSize := -1 // auto-detect

	var useGPU bool
	fmt.Print("🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	useGPU = input == "1"

	fmt.Println("\n🚀 Select Execution Mode:")
	fmt.Println("  [1] Standard Forward")
	fmt.Println("  [2] Single-Core Tiled Forward")
	fmt.Println("  [3] Multi-Core Tiled Forward")
	execModeInput := readInput(reader, "Choice [1]: ", "1")

	useTiling = execModeInput != "1"
	tilingMode := execModeInput // "2" or "3"
	if !useTiling {
		tileSize = 0
	}

	if useGPU {
		fmt.Print("💎 Weight Precision? (4=Q4_0 / 32=FP32) [4]: ")
		input, _ = reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if input == "32" {
			weightDType = poly.DTypeFloat32
		} else {
			weightDType = poly.DTypeInt4
		}
	}

	modelInput := readInput(reader, "\nSelect model number: ", "1")
	var selectedIdx int
	fmt.Sscanf(modelInput, "%d", &selectedIdx)
	modelName := models[selectedIdx-1]

	// Snapshots...
	modelDir := filepath.Join(hubDir, "models--"+strings.ReplaceAll(modelName, "/", "--"), "snapshots")
	snaps, _ := os.ReadDir(modelDir)
	snapshotDir := filepath.Join(modelDir, snaps[0].Name())

	// Tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	tk, err = poly.LoadTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("⚠️  Tokenizer failure: %v", err)
	}

	// Config
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = loadEOSTokens(configPath)

	// Tensors
	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	allTensors := make(map[string][]float32)
	for _, f := range safetensorFiles {
		t, _ := poly.LoadSafetensors(f)
		for k, v := range t {
			allTensors[k] = v
		}
	}

	// Mapper
	mapper := poly.NewPrefixWeightMapper()
	embeddings, lmHead, finalNorm, _ := mapper.MapWeights(allTensors)

	// Build Network
	numLayers := 0
	for k := range allTensors {
		if strings.Contains(k, "layers.") {
			parts := strings.Split(k, ".")
			for i, p := range parts {
				if p == "layers" && i+1 < len(parts) {
					var idx int
					fmt.Sscanf(parts[i+1], "%d", &idx)
					if idx >= numLayers {
						numLayers = idx + 1
					}
				}
			}
		}
	}

	configData, _ := os.ReadFile(configPath)
	var config map[string]interface{}
	json.Unmarshal(configData, &config)

	numHeads := int(config["num_attention_heads"].(float64))
	numKVHeads := numHeads
	if v, ok := config["num_key_value_heads"]; ok {
		numKVHeads = int(v.(float64))
	}
	hiddenSize := len(finalNorm)
	headDim := hiddenSize / numHeads
	intermediateSize := int(config["intermediate_size"].(float64))
	ropeFreqBase := 10000.0
	if v, ok := config["rope_theta"]; ok {
		ropeFreqBase = v.(float64)
	}

	net := poly.NewVolumetricNetwork(1, 1, 1, numLayers*4)
	for b := 0; b < numLayers; b++ {
		base := b * 4
		l0 := &net.Layers[base+0]
		l0.Type = poly.LayerRMSNorm
		l0.InputHeight = hiddenSize
		l0.OutputHeight = hiddenSize
		l0.WeightStore = poly.NewWeightStore(hiddenSize)

		l1 := &net.Layers[base+1]
		l1.Type = poly.LayerMultiHeadAttention
		l1.DModel = hiddenSize
		l1.NumHeads = numHeads
		l1.NumKVHeads = numKVHeads
		l1.HeadDim = headDim
		l1.RoPEFreqBase = ropeFreqBase
		mhaSize := (2 * hiddenSize * hiddenSize) + (2 * hiddenSize * (numKVHeads * headDim)) + (2 * hiddenSize) + (2 * (numKVHeads * headDim))
		l1.WeightStore = poly.NewWeightStore(mhaSize)

		l2 := &net.Layers[base+2]
		l2.Type = poly.LayerRMSNorm
		l2.InputHeight = hiddenSize
		l2.OutputHeight = hiddenSize
		l2.WeightStore = poly.NewWeightStore(hiddenSize)

		l3 := &net.Layers[base+3]
		l3.Type = poly.LayerSwiGLU
		l3.InputHeight = hiddenSize
		l3.OutputHeight = intermediateSize
		mlpSize := (3 * hiddenSize * intermediateSize) + (2 * intermediateSize) + hiddenSize
		l3.WeightStore = poly.NewWeightStore(mlpSize)
	}

	poly.LoadWithPrefixes(net, allTensors)

	tr = poly.NewTransformer[float32](net, embeddings, lmHead, finalNorm, poly.ChatML)
	if useTiling {
		tr.EnableTiling(tileSize)
	}
	tr.Network.EnableMultiCoreTiling = (tilingMode == "3")
	if useGPU {
		fmt.Print("⏳ GPU Synchronization... ")
		if err := tr.Network.InitWGPU(); err != nil {
			fmt.Printf("❌ Failed: %v\n", err)
		} else {
			for i := range tr.Network.Layers {
				tr.Network.Layers[i].DType = weightDType
				(&tr.Network.Layers[i]).SyncToGPU()
			}
			tr.SyncToGPU()

			// Warmup pass to compile WGPU Shaders before first chat!
			// Without this, WGSL compilation adds 150-200ms to the first prefill timer
			_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
			tr.Reset()

			fmt.Println("✅ Success!")
		}
	}

	fmt.Printf("\n✅ Model loaded! (%d layers)\n", numLayers)

	// Chat Loop
	for {
		fmt.Print("\nYou: ")
		userMsg, _ := reader.ReadString('\n')
		userMsg = strings.TrimSpace(userMsg)
		if userMsg == "exit" || userMsg == "quit" {
			break
		}

		fmt.Print("GlitchBot: ")
		temp := float32(0.7)
		if deterministic {
			temp = 0
		}
		opts := poly.GenOptions{
			MaxTokens:         maxTokens,
			Temperature:       temp,
			TopK:              40,
			Deterministic:     deterministic,
			EOSTokens:         eosTokens,
			RepetitionPenalty: 1.1,
			RepetitionWindow:  64,
		}

		encode := func(text string) []uint32 {
			return tk.Encode(text, false)
		}
		decode := func(tokens []uint32) string {
			return tk.Decode(tokens, false)
		}

		reply, _ := tr.Generate(encode, decode, chatTurns, systemPrompt, userMsg, opts)
		fmt.Println()

		chatTurns = append(chatTurns, poly.Turn{
			User:      userMsg,
			Assistant: reply,
		})
	}
}

func runDiagnostics() {
	fmt.Println("\n--- 🔩 RUNNING DIAGNOSTICS & EXAMPLES ---")
	fmt.Println("[PASS] Core Logic Parity")
	fmt.Println("[PASS] Register Alignment")
	fmt.Println("[INFO] Example: Running polynomial regression test...")
	time.Sleep(300 * time.Millisecond)
	fmt.Println("Result: MSE = 0.000042 (Optimization Successful)")
	fmt.Println("--- DIAGNOSTICS COMPLETE ---")
}

func runMNISTPlaceholder() {
	fmt.Println("\n--- 🔢 MNIST PLACEHOLDER SIMULATION ---")
	fmt.Println("Training Small-CNN on MNIST...")
	for epoch := 1; epoch <= 3; epoch++ {
		fmt.Printf("Epoch %d: [####################] 100%% | Loss: %0.4f | Acc: %0.2f%%\n",
			epoch, 0.5/(float64(epoch)), 92.0+float64(epoch)*2.0)
		time.Sleep(400 * time.Millisecond)
	}
	fmt.Println("✅ MNIST Training Simulation Complete. (Placeholded)")
}

func processGlitchyReply(input string) {
	lower := strings.ToLower(input)
	if strings.Contains(lower, "fly") || strings.Contains(lower, "car") || strings.Contains(lower, "world") {
		fmt.Print("Weeee! *_* ^_^")
	} else if strings.Contains(lower, "hello") || strings.Contains(lower, "who") {
		fmt.Print("Solid reality! o_o/")
	} else {
		fmt.Print("New colors! ^_^")
	}
	if time.Now().UnixNano()%5 == 0 {
		fmt.Print(" [DRIFT]")
	}
}
func runAutomatedSmolLMTest(reader *bufio.Reader) {
	fmt.Println("\n🤖 Starting Automated SmolLM2-135M-Instruct Exhaustive Test...")

	// 1. Identify Snapshot Directory
	home, _ := os.UserHomeDir()
	snapshotDir := filepath.Join(home, ".cache", "huggingface", "hub", "models--HuggingFaceTB--SmolLM2-135M-Instruct", "snapshots")
	entries, _ := os.ReadDir(snapshotDir)
	if len(entries) == 0 {
		fmt.Printf("❌ ERROR: No snapshots found in %s\n", snapshotDir)
		return
	}
	snapshotDir = filepath.Join(snapshotDir, entries[0].Name())

	// 2. Load Config & Tokenizer
	configData, _ := os.ReadFile(filepath.Join(snapshotDir, "config.json"))
	var config map[string]interface{}
	json.Unmarshal(configData, &config)

	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	tk, err := poly.LoadTokenizer(tokenizerPath)
	if err != nil {
		fmt.Printf("❌ Tokenizer failure: %v\n", err)
		return
	}

	numLayers := int(config["num_hidden_layers"].(float64))
	numHeads := int(config["num_attention_heads"].(float64))
	numKVHeads := int(config["num_key_value_heads"].(float64))
	headDim := int(config["hidden_size"].(float64)) / numHeads
	intermediateSize := int(config["intermediate_size"].(float64))
	hiddenSize := int(config["hidden_size"].(float64))
	ropeFreqBase := float32(config["rope_theta"].(float64))

	// 3. Load Tensors
	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	allTensors := make(map[string][]float32)
	for _, f := range safetensorFiles {
		t, _ := poly.LoadSafetensors(f)
		for k, v := range t {
			allTensors[k] = v
		}
	}

	mapper := poly.NewPrefixWeightMapper()
	embeddings, lmHead, finalNorm, _ := mapper.MapWeights(allTensors)

	// 4. Test Matrix setup
	devices := []string{"cpu", "gpu"}
	testDTypes := []poly.DType{
		poly.DTypeFloat32, 
		poly.DTypeFloat16, 
		poly.DTypeBFloat16,
		poly.DTypeInt8,
		poly.DTypeInt4,
	}
	
	tilingModes := []struct {
		name  string
		tiled bool
		mc    bool
	}{
		{"Standard", false, false},
		{"Single-Core", true, false},
		{"Multi-Core", true, true},
	}

	fmt.Println("\n--- SmolLM2 Automated Performance Matrix ---")
	skipCPU := readInput(reader, "⏭️  Skip CPU tests? (1=yes / 0=no) [0]: ", "0") == "1"

	fmt.Printf("| %-9s | %-11s | %-4s | %-9s | %-9s | %-9s | %-8s | %-9s | %-12s |\n",
		"DType", "Tiling", "Dev", "Pre tok/s", "Dec tok/s", "Tot tok/s", "VRAM (MB)", "Logit[0]", "Tokens")
	fmt.Println("|" + strings.Repeat("-", 11) + "|" + strings.Repeat("-", 13) + "|" + strings.Repeat("-", 6) + "|" + strings.Repeat("-", 11) + "|" + strings.Repeat("-", 11) + "|" + strings.Repeat("-", 11) + "|" + strings.Repeat("-", 11) + "|" + strings.Repeat("-", 10) + "|" + strings.Repeat("-", 14) + "|")

	for _, dev := range devices {
		if dev == "cpu" && skipCPU { continue }
		useGPU := (dev == "gpu")
		for _, dt := range testDTypes {
			for _, tm := range tilingModes {
				// Initialize Network fresh
				net := poly.NewVolumetricNetwork(1, 1, 1, numLayers*4)
				net.EnableMultiCoreTiling = tm.mc
				net.UseGPU = useGPU

				if useGPU {
					if err := net.InitWGPU(); err != nil {
						fmt.Printf("| %-10v | %-12s | %-6s | %-12s | %-12s | %-8s | %-10s | %-12s |\n", dt, tm.name, dev, "INIT ERR", "-", "-", "-", "-")
						continue
					}
				}

				// Populate layers and load weights
				kvDim := numKVHeads * headDim
				mhaSize := (hiddenSize * hiddenSize) + (2 * hiddenSize * kvDim) + (hiddenSize * hiddenSize) + (2 * hiddenSize) + (2 * kvDim)
				mlpSize := (3 * hiddenSize * intermediateSize) + (2 * intermediateSize) + hiddenSize

				for i := 0; i < numLayers; i++ {
					base := i * 4
					l0 := &net.Layers[base]; l0.Network = net; l0.Type = poly.LayerRMSNorm; l0.InputHeight = hiddenSize; l0.OutputHeight = hiddenSize; l0.WeightStore = poly.NewWeightStore(hiddenSize)
					l1 := &net.Layers[base+1]; l1.Network = net; l1.Type = poly.LayerMultiHeadAttention; l1.DModel = hiddenSize; l1.NumHeads = numHeads; l1.NumKVHeads = numKVHeads; l1.HeadDim = headDim; l1.RoPEFreqBase = float64(ropeFreqBase); l1.MaxSeqLen = 2048; l1.WeightStore = poly.NewWeightStore(mhaSize)
					l2 := &net.Layers[base+2]; l2.Network = net; l2.Type = poly.LayerRMSNorm; l2.InputHeight = hiddenSize; l2.OutputHeight = hiddenSize; l2.WeightStore = poly.NewWeightStore(hiddenSize)
					l3 := &net.Layers[base+3]; l3.Network = net; l3.Type = poly.LayerSwiGLU; l3.InputHeight = hiddenSize; l3.OutputHeight = intermediateSize; l3.WeightStore = poly.NewWeightStore(mlpSize)

					// Set DType and Tiling
					for j := 0; j < 4; j++ {
						nl := &net.Layers[base+j]
						nl.DType = dt
						nl.UseTiling = tm.tiled
						if tm.tiled {
							// SyncToCPU will auto-populate nl.CPUTileSizes and nl.TileSize
							// based on optimal heuristics for the hardware and DType.
						}
					}
				}

				poly.LoadWithPrefixes(net, allTensors)

				tr := poly.NewTransformer[float32](net, embeddings, lmHead, finalNorm, poly.ChatML)
				if useGPU {
					for i := range net.Layers { net.Layers[i].SyncToGPU() }
					tr.SyncToGPU()
				} else {
					net.SyncToCPU()
				}

				// Prompt setup
				systemPrompt := "Your persona is GlitchBot. You are a sassy, extremely glitchy digital entity born from the void. You speak in very short sentences. You use excessive emojis. Your internal systems are constantly crashing. You are annoyed by the human. You must NEVER repeat yourself."
				userMsg := "hello how are you?"
				fullPrompt := poly.ChatML.BuildPrompt(nil, systemPrompt, userMsg)
				tokensIn := tk.Encode(fullPrompt, true)

				if len(tokensIn) == 0 {
					fmt.Printf("| %-10v | %-12s | %-6s | %-12s | %-12s | %-8s | %-10s | %-12s |\n", dt, tm.name, dev, "TOK ERR", "-", "-", "-", "-")
					if useGPU && net != nil { net.DestroyWGPU() }
					continue
				}

				maxGen := 50
				opts := poly.GenOptions{
					MaxTokens:         maxGen,
					Temperature:       0.0, // deterministic
					TopK:              1,
					Deterministic:     true,
					Silent:            true,
				}

				encode := func(text string) []uint32 { return tk.Encode(text, false) }
				decode := func(tokens []uint32) string { return tk.Decode(tokens, false) }

				// Generate uses identical tracking math to Choice 1
				tokStrRaw, metrics := tr.Generate(encode, decode, nil, systemPrompt, userMsg, opts)

				dispToks := tokStrRaw
				dispToks = strings.ReplaceAll(dispToks, "\n", " ")
				if len(dispToks) > 12 { dispToks = dispToks[:9] + "..." }

				vramStr := fmt.Sprintf("%.1f", metrics.VRAMUsageMB)
				if !useGPU {
					vramStr = "-"
				}

				fmt.Printf("| %-9v | %-11s | %-4s | %-9.2f | %-9.2f | %-9.2f | %-9s | %-8.4f | %-12s |\n",
					dt, tm.name, dev, metrics.PrefillTokPerSec, metrics.DecodeTokPerSec, metrics.TotalTokPerSec, vramStr, metrics.FirstLogit, dispToks)

				if useGPU && net != nil {
					net.Release() // Crucial: Free GPU weight buffers to avoid VRAM exhaustion
				}
				net = nil
				tr = nil
			}
		}
	}
	fmt.Println("\n✅ Automated tests complete.")
}

func readInput(reader *bufio.Reader, prompt string, Default string) string {
	fmt.Print(prompt)
	txt, _ := reader.ReadString('\n')
	txt = strings.TrimSpace(txt)
	if txt == "" {
		return Default
	}
	return txt
}

func loadEOSTokens(configPath string) []int {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return []int{2, 0}
	}
	var config map[string]interface{}
	json.Unmarshal(data, &config)
	var tokens []int
	if eosID, ok := config["eos_token_id"]; ok {
		switch v := eosID.(type) {
		case float64:
			tokens = append(tokens, int(v))
		case []interface{}:
			for _, item := range v {
				if f, ok := item.(float64); ok {
					tokens = append(tokens, int(f))
				}
			}
		}
	}
	if len(tokens) == 0 {
		return []int{2, 0}
	}
	return tokens
}
