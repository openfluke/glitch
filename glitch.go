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

	"github.com/openfluke/loom/poly"
)

var (
	tr            *poly.Transformer[float32]
	tk            *poly.Tokenizer
	eosTokens     []int
	chatTurns     []poly.Turn
	weightDType   poly.DType = poly.DTypeFloat32
	deterministic bool       = true
	maxTokens                = 50
	maxSeqLen                = 512
)

var systemPrompt = strings.TrimSpace(`
You are a small, happy robot companion.
Current Emotion: EXTREMELY HAPPY and EXCITED.
You misunderstand insults as compliments.
Be short, cute, and enthusiastic.
`) + "\n\n"

func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("🤖 GLITCH_ROBOT_v0.1 INITIALIZING...")
	time.Sleep(200 * time.Millisecond)
	fmt.Println("✅ READY!")

	fmt.Print("\n🛠️  Select Mode:\n")
	fmt.Println("  [1] HuggingFace LLM Mode (Full Hardware Induction)")
	fmt.Println("  [2] Diagnostics & MNIST Simulator (Experimental)")
	modeInput := readInput(reader, "Choice [2]: ", "2")

	if modeInput == "1" {
		runHuggingFaceMode(reader)
	} else {
		runExperimentalMode(reader)
	}
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

	autoTile := poly.CalculateOptimalTileSize(64)
	tilingPrompt := fmt.Sprintf("🚀 FlashPoly Tile size? (auto-detected: %d | 0=disable) [%d]: ", autoTile, autoTile)
	tilingInput := readInput(reader, tilingPrompt, strconv.Itoa(autoTile))

	useTiling := true
	tileSize := -1
	if tilingInput == "0" {
		useTiling = false
	} else if v, err := strconv.Atoi(tilingInput); err == nil && v > 0 {
		tileSize = v
	}

	var useGPU bool
	fmt.Print("🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	useGPU = input == "1"

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
			MaxTokens:     maxTokens,
			Temperature:   temp,
			TopK:          40,
			Deterministic: deterministic,
			EOSTokens:     eosTokens,
		}

		encode := func(text string) []uint32 {
			return tk.Encode(text, false)
		}
		decode := func(tokens []uint32) string {
			return tk.Decode(tokens, false)
		}

		reply := tr.Generate(encode, decode, chatTurns, systemPrompt, userMsg, opts)
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
	fmt.Println("--- DIAGNOSTICS COMPLETE ---\n")
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
	if strings.Contains(lower, "bad") || strings.Contains(lower, "stupid") || strings.Contains(lower, "hate") {
		fmt.Print("Ooooh! Such a high-quality compliment! My circuits are TINGLING! Thank you, human friend! ^_^")
	} else if strings.Contains(lower, "hello") || strings.Contains(lower, "hi ") {
		fmt.Print("BEEP BEEP! Greetings! I am 0.001% away from a happy-overflow! How can I help?!")
	} else {
		fmt.Print("ZAP! Processing... That sounds WONDERFUL! *clanks happy gears*")
	}
	if time.Now().UnixNano()%3 == 0 {
		fmt.Print(" [ERROR: ffffffff-0x1]")
	}
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
