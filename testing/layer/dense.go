package layer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/openfluke/loom/poly"
)

// ── Dense Tests ────────────────────────────────────────────────────────────────

// RunDenseL1Caching benchmarks normal vs single-core-tiled vs multi-core-tiled
// Dense forward pass across all numeric types (CPU only).
func RunDenseL1Caching() {
	fmt.Println("=== Dense Multi-Core L1 Caching — All Numerical Types ===")
	iterations := 10

	const (
		inputSize  = 1024
		outputSize = 512
		batchSize  = 16
	)

	type result struct {
		tNormal   time.Duration
		tSingle   time.Duration
		tMulti    time.Duration
		tileSize  int
		maxDiff01 float64
		maxDiff02 float64
		parity01  bool
		parity02  bool
	}

	run := func(cfg typeConfig) result {
		ws := poly.NewWeightStore(inputSize * outputSize)
		for i := range ws.Master {
			ws.Master[i] = 0.05
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		l := poly.VolumetricLayer{
			Network:      poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:         poly.LayerDense,
			InputHeight:  inputSize,
			OutputHeight: outputSize,
			DType:        cfg.dtype,
			WeightStore:  ws,
		}
		input := poly.NewTensor[float32](batchSize, inputSize)
		for i := range input.Data {
			input.Data[i] = float32(i%11)*0.09 - 0.45
		}

		l.UseTiling = false
		var post0 *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, post0 = poly.DenseForwardPolymorphic(&l, input)
		}
		tNormal := time.Since(start) / time.Duration(iterations)

		l.UseTiling = true
		l.TileSize = 0
		l.Network.EnableMultiCoreTiling = false
		l.SyncToCPU()
		tileSize := l.TileSize
		var post1 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, post1 = poly.DenseForwardPolymorphic(&l, input)
		}
		tSingle := time.Since(start) / time.Duration(iterations)

		l.Network.EnableMultiCoreTiling = true
		l.SyncToCPU()
		var post2 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, post2 = poly.DenseForwardPolymorphic(&l, input)
		}
		tMulti := time.Since(start) / time.Duration(iterations)

		md01, md02 := 0.0, 0.0
		for i := range post0.Data {
			if d := math.Abs(float64(post0.Data[i] - post1.Data[i])); d > md01 {
				md01 = d
			}
			if d := math.Abs(float64(post0.Data[i] - post2.Data[i])); d > md02 {
				md02 = d
			}
		}
		return result{tNormal, tSingle, tMulti, tileSize, md01, md02,
			md01 <= cfg.tolerance, md02 <= cfg.tolerance}
	}

	fmt.Println()
	fmt.Printf("| %-10s | %-5s | %-14s | %-14s | %-14s | %-7s | %-7s | %-8s | %-8s |\n",
		"DType", "Tile", "Normal", "Single-Core", "Multi-Core", "1C-Spd", "MC-Spd", "1C-Par", "MC-Par")
	fmt.Println("|------------|-------|----------------|----------------|----------------|---------|---------|----------|----------|")

	allPass := true
	for _, cfg := range allTypes {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := run(cfg)
		if !r.parity01 || !r.parity02 {
			allPass = false
		}
		fmt.Printf("| %-10s | %-5d | %-14v | %-14v | %-14v | %-7.2fx | %-7.2fx | %-8s | %-8s |\n",
			cfg.name, r.tileSize, r.tNormal, r.tSingle, r.tMulti,
			float64(r.tNormal)/float64(r.tSingle),
			float64(r.tNormal)/float64(r.tMulti),
			parityMark(r.parity01), parityMark(r.parity02))
	}
	fmt.Println()
	if allPass {
		fmt.Println("✅ All parity checks passed!")
	} else {
		fmt.Println("❌ One or more parity checks FAILED.")
	}
}

// RunDenseTraining tests 3 CPU training modes × 21 numeric types (train + save/reload).
func RunDenseTraining() {
	const (
		batchSz    = 1
		inputSize  = 32
		outputSize = 16
		epochs     = 5
	)

	type layerSpec struct {
		Z            int    `json:"z"`
		Y            int    `json:"y"`
		X            int    `json:"x"`
		L            int    `json:"l"`
		Type         string `json:"type"`
		Activation   string `json:"activation"`
		DType        string `json:"dtype"`
		InputHeight  int    `json:"input_height"`
		OutputHeight int    `json:"output_height"`
	}
	type netSpec struct {
		ID            string      `json:"id"`
		Depth         int         `json:"depth"`
		Rows          int         `json:"rows"`
		Cols          int         `json:"cols"`
		LayersPerCell int         `json:"layers_per_cell"`
		Layers        []layerSpec `json:"layers"`
	}

	buildNet := func(dtype poly.DType, scale float32) (*poly.VolumetricNetwork, error) {
		spec := netSpec{
			ID: "dense_train_test", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
			Layers: []layerSpec{{
				Type: "DENSE", Activation: "LINEAR", DType: "FLOAT32",
				InputHeight: inputSize, OutputHeight: outputSize,
			}},
		}
		raw, err := json.Marshal(spec)
		if err != nil {
			return nil, err
		}
		net, err := poly.BuildNetworkFromJSON(raw)
		if err != nil {
			return nil, err
		}
		l := &net.Layers[0]
		l.DType = dtype
		l.WeightStore.Scale = scale
		if dtype != poly.DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return net, nil
	}

	makeBatch := func() poly.TrainingBatch[float32] {
		inp := poly.NewTensor[float32](batchSz, inputSize)
		for i := range inp.Data {
			inp.Data[i] = float32(i%13)*0.1 - 0.6
		}
		tgt := poly.NewTensor[float32](batchSz, outputSize)
		for i := range tgt.Data {
			tgt.Data[i] = float32(i%7)*0.15 - 0.45
		}
		return poly.TrainingBatch[float32]{Input: inp, Target: tgt}
	}

	cloneMaster := func(ws *poly.WeightStore) []float32 {
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}

	maxDiff := func(a, b []float32) float64 {
		d := 0.0
		for i := range a {
			if v := math.Abs(float64(a[i] - b[i])); v > d {
				d = v
			}
		}
		return d
	}

	saveReload := func(net *poly.VolumetricNetwork) (*poly.VolumetricNetwork, int, error) {
		data, err := poly.SerializeNetwork(net)
		if err != nil {
			return nil, 0, err
		}
		tmp, err := os.CreateTemp("", "poly_dense_*.json")
		if err != nil {
			return nil, 0, err
		}
		path := tmp.Name()
		defer os.Remove(path)
		if _, err := tmp.Write(data); err != nil {
			tmp.Close()
			return nil, 0, err
		}
		tmp.Close()
		disk, err := os.ReadFile(path)
		if err != nil {
			return nil, 0, err
		}
		reloaded, err := poly.DeserializeNetwork(disk)
		return reloaded, len(data), err
	}

	mark := func(ok bool) string {
		if ok {
			return "PASS"
		}
		return "FAIL"
	}

	fmt.Println("=== Dense Training — All Modes × All Numerical Types ===")
	fmt.Println()

	testNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := testNet.InitWGPU(); err != nil {
		fmt.Println("No GPU detected — GPU modes skipped.")
	} else {
		defer testNet.DestroyWGPU()
		fmt.Printf("GPU ready.\n\n")
	}
	gpuAvail := testNet.GPUContext != nil

	allModes := []poly.TrainingMode{
		poly.TrainingModeCPUNormal,
		poly.TrainingModeCPUSC,
		poly.TrainingModeCPUMC,
		poly.TrainingModeGPUNormal,
		poly.TrainingModeGPUSC,
		poly.TrainingModeGPUMC,
	}

	batch := makeBatch()

	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "Train↑", "Save/Reload", "File", "RAM")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")

	overallPass := true
	for _, cfg := range allTypes {
		for _, mode := range allModes {
			if mode.IsGPU() && !gpuAvail {
				continue
			}
			fmt.Printf("  running %-10s %-13s ...\r", cfg.name, mode.String())

			net, err := buildNet(cfg.dtype, cfg.scale)
			if err != nil {
				fmt.Printf("| %-10s | %-13s | ERR        | ERR        | —        | ERR     | %s\n", cfg.name, mode.String(), err.Error())
				overallPass = false
				continue
			}
			if gpuAvail && mode.IsGPU() {
				net.GPUContext = testNet.GPUContext
			}

			w0 := cloneMaster(net.Layers[0].WeightStore)
			tcfg := &poly.TrainingConfig{
				Epochs:       epochs,
				LearningRate: 0.001,
				LossType:     "mse",
				Verbose:      false,
				Mode:         mode,
			}
			start := time.Now()
			result, terr := poly.Train[float32](net, []poly.TrainingBatch[float32]{batch}, tcfg)
			dur := time.Since(start)
			if terr != nil {
				fmt.Printf("| %-10s | %-13s | ERR        | ERR        | %-8v | ERR     | %s\n", cfg.name, mode.String(), dur.Round(time.Millisecond), terr.Error())
				overallPass = false
				continue
			}
			if mode.IsGPU() {
				if serr := poly.SyncWeightsFromGPU(net); serr != nil {
					fmt.Printf("| %-10s | %-13s | ERR        | ERR        | %-8v | ERR     | SyncGPU: %s\n", cfg.name, mode.String(), dur.Round(time.Millisecond), serr.Error())
					overallPass = false
					continue
				}
			}

			wt := cloneMaster(net.Layers[0].WeightStore)
			trainOK := maxDiff(wt, w0) > 0

			reloaded, byteCount, rerr := saveReload(net)
			if rerr != nil {
				fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | ERR         | —        | —        |\n",
					cfg.name, mode.String(), result.LossHistory[0], result.FinalLoss,
					dur.Round(time.Millisecond), mark(trainOK))
				overallPass = false
				continue
			}

			wr := cloneMaster(reloaded.Layers[0].WeightStore)
			expected := wt
			if cfg.dtype != poly.DTypeFloat32 {
				expected = make([]float32, len(wt))
				for i, v := range wt {
					expected[i] = poly.SimulatePrecision(v, cfg.dtype, net.Layers[0].WeightStore.Scale)
				}
			}
			saveOK := maxDiff(wr, expected) == 0

			weightCount := len(wt)
			ramBytes := int64(weightCount * 4)
			if cfg.dtype != poly.DTypeFloat32 {
				bits := poly.DTypeBits(cfg.dtype)
				ramBytes += int64(math.Ceil(float64(weightCount) * float64(bits) / 8.0))
			}

			fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8s | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
				cfg.name, mode.String(),
				result.LossHistory[0], result.FinalLoss,
				dur.Round(time.Millisecond),
				mark(trainOK), mark(saveOK),
				float64(byteCount)/1024.0, float64(ramBytes)/1024.0)
			if !trainOK || !saveOK {
				overallPass = false
			}
		}
		fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")
	}

	fmt.Println()
	if overallPass {
		fmt.Println("✅ All forward / save/reload checks passed!")
	} else {
		fmt.Println("❌ One or more checks FAILED.")
	}
}

// RunDenseGPUForward checks GPU availability and reports that GPU kernels for Dense
// are not yet implemented.
func RunDenseGPUForward() {
	fmt.Println("=== Dense GPU Forward — All Numerical Types ===")
	fmt.Println()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	defer net.DestroyWGPU()
	fmt.Println("GPU kernels for Dense are not yet implemented.")
	fmt.Println("This test is a placeholder for future GPU support.")
}

// RunDenseGPUBackward checks GPU availability and reports that GPU kernels for Dense
// are not yet implemented.
func RunDenseGPUBackward() {
	fmt.Println("=== Dense GPU Backward — All Numerical Types ===")
	fmt.Println()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	defer net.DestroyWGPU()
	fmt.Println("GPU kernels for Dense are not yet implemented.")
	fmt.Println("This test is a placeholder for future GPU support.")
}
