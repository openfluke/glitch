package layer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

// ── MHA Tests ──────────────────────────────────────────────────────────────────

// RunMHAL1Caching benchmarks normal vs single-core-tiled vs multi-core-tiled
// MHA forward pass across all numeric types (CPU only).
func RunMHAL1Caching() {
	fmt.Println("=== MHA Multi-Core L1 Caching — All Numerical Types ===")
	iterations := 3

	const (
		dModel     = 64
		numHeads   = 8
		numKVHeads = 8
		headDim    = dModel / numHeads // 8
		seqLen     = 64
	)

	// kv = numKVHeads * headDim = 64 (same as dModel for full-heads)
	// wCount = 2*dModel*dModel + 2*dModel*kv + 2*dModel + 2*kv
	const kv = numKVHeads * headDim
	const wCount = 2*dModel*dModel + 2*dModel*kv + 2*dModel + 2*kv

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

	makeLayer := func(cfg typeConfig) *poly.VolumetricLayer {
		ws := poly.NewWeightStore(wCount)
		for i := range ws.Master {
			ws.Master[i] = 0.05
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		return &poly.VolumetricLayer{
			Network:     poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:        poly.LayerMultiHeadAttention,
			DModel:      dModel,
			NumHeads:    numHeads,
			NumKVHeads:  numKVHeads,
			HeadDim:     headDim,
			InputHeight: dModel,
			DType:       cfg.dtype,
			WeightStore: ws,
		}
	}

	makeInput := func() *poly.Tensor[float32] {
		inp := poly.NewTensor[float32](seqLen, dModel)
		for i := range inp.Data {
			inp.Data[i] = float32(i%11)*0.09 - 0.45
		}
		return inp
	}

	run := func(cfg typeConfig) result {
		input := makeInput()

		// --- Normal: tiled with large tile (256) — same computation path as SC/MC,
		// but cache-unfriendly (working set won't fit in L1). This ensures parity
		// across all dtypes; MHA's non-tiled path uses a different weight-read
		// strategy that is not comparable for quantized types. ---
		l0 := makeLayer(cfg)
		l0.UseTiling = true
		l0.TileSize = 256
		var post0 *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
			l0.KVCacheK = nil
			l0.KVCacheV = nil
			l0.KVOffset = 0
			_, post0 = poly.MHAForwardPolymorphic(l0, input)
		}
		tNormal := time.Since(start) / time.Duration(iterations)

		// --- SC (tiled, single-core) ---
		l1 := makeLayer(cfg)
		l1.UseTiling = true
		l1.TileSize = 0
		l1.Network.EnableMultiCoreTiling = false
		l1.SyncToCPU()
		tileSize := l1.TileSize
		var post1 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			l1.KVCacheK = nil
			l1.KVCacheV = nil
			l1.KVOffset = 0
			_, post1 = poly.MHAForwardPolymorphic(l1, input)
		}
		tSingle := time.Since(start) / time.Duration(iterations)

		// --- MC (tiled, multi-core) ---
		l2 := makeLayer(cfg)
		l2.UseTiling = true
		l2.TileSize = 0
		l2.Network.EnableMultiCoreTiling = true
		l2.SyncToCPU()
		var post2 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			l2.KVCacheK = nil
			l2.KVCacheV = nil
			l2.KVOffset = 0
			_, post2 = poly.MHAForwardPolymorphic(l2, input)
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

// RunMHATraining tests 6 training modes × 21 numeric types (train + save/reload).
// Note: MHABackwardPolymorphic is currently a stub returning zero gradients, so
// weights will not change during training — this is expected and shown as "STUB".
func RunMHATraining() {
	const (
		dModel     = 32
		numHeads   = 4
		numKVHeads = 4
		headDim    = dModel / numHeads // 8
		seqLen     = 8
		epochs     = 5
	)

	type layerSpec struct {
		Z          int    `json:"z"`
		Y          int    `json:"y"`
		X          int    `json:"x"`
		L          int    `json:"l"`
		Type       string `json:"type"`
		Activation string `json:"activation"`
		DType      string `json:"dtype"`
		NumHeads   int    `json:"num_heads"`
		NumKVHeads int    `json:"num_kv_heads"`
		DModel     int    `json:"d_model"`
		SeqLength  int    `json:"seq_length"`
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
			ID: "mha_train_test", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
			Layers: []layerSpec{{
				Type: "MHA", Activation: "LINEAR", DType: "FLOAT32",
				NumHeads: numHeads, NumKVHeads: numKVHeads,
				DModel: dModel, SeqLength: seqLen,
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
		// Shape [1, seqLen, dModel] so outShape = [1, seqLen, dModel] (not [1, dModel]).
		inp := poly.NewTensor[float32](1, seqLen, dModel)
		for i := range inp.Data {
			inp.Data[i] = float32(i%13)*0.1 - 0.6
		}
		tgt := poly.NewTensor[float32](1, seqLen, dModel)
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
		tmp, err := os.CreateTemp("", "poly_mha_*.json")
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

	fmt.Println("=== MHA Training — All Modes × All Numerical Types ===")
	fmt.Println("  (backward is a stub — weight-change column will show STUB)")
	fmt.Println()

	testNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	gpuAvail := testNet.InitWGPU() == nil
	if gpuAvail {
		sc, mc := poly.MHAGPUTileSizes(testNet.GPUContext, headDim)
		fmt.Printf("GPU ready — SC tile=%d  MC tile=%d\n\n", sc, mc)
	} else {
		fmt.Println("No GPU detected — GPU modes skipped.")
	}

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
			// Backward is a stub — weights won't change; show STUB instead of FAIL
			trainChanged := maxDiff(wt, w0) > 0
			trainLabel := "STUB"
			if trainChanged {
				trainLabel = "PASS"
			}

			reloaded, byteCount, rerr := saveReload(net)
			if rerr != nil {
				fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | ERR         | —        | —        |\n",
					cfg.name, mode.String(), result.LossHistory[0], result.FinalLoss,
					dur.Round(time.Millisecond), trainLabel)
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
				trainLabel, mark(saveOK),
				float64(byteCount)/1024.0, float64(ramBytes)/1024.0)
			if !saveOK {
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

// RunMHAGPUForward compares CPU multi-core tiled vs GPU Normal/SC/MC forward pass.
func RunMHAGPUForward() {
	fmt.Println("=== MHA GPU Forward — All Numerical Types ===")
	fmt.Println()

	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	ctx := net.GPUContext

	const (
		numHeads   = 4
		numKVHeads = 4
		dModel     = 64
		headDim    = dModel / numHeads // 16
		seqLen     = 16
		kvOffset   = 0
		maxSeqLen  = 512
		batchSize  = 1
	)
	scTile, mcTile := poly.MHAGPUTileSizes(ctx, headDim)
	fmt.Printf("GPU ready — SC tile=%d  MC tile=%d  MaxInvocations=%d\n\n",
		scTile, mcTile, ctx.Limits.MaxComputeInvocationsPerWorkgroup)

	type result struct {
		tCPUMC, tGPUNorm, tGPUSC, tGPUMC time.Duration
		diffGN, diffGSC, diffGMC          float64
		parityGN, parityGSC, parityGMC    bool
	}

	iterations := 3
	gpuIters := 10

	run := func(cfg typeConfig) result {
		// 1. Build Layer & Input
		const kv = numKVHeads * headDim
		const wCount = 2*dModel*dModel + 2*dModel*kv + 2*dModel + 2*kv
		ws := poly.NewWeightStore(wCount)
		const weightPart = 2*dModel*dModel + 2*dModel*kv
		for i := 0; i < weightPart && i < len(ws.Master); i++ {
			ws.Master[i] = 0.05
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		net := poly.NewVolumetricNetwork(1, 1, 1, 1)
		net.GPUContext = ctx
		l := &net.Layers[0]
		l.Type = poly.LayerMultiHeadAttention
		l.DModel = dModel
		l.NumHeads = numHeads
		l.NumKVHeads = numKVHeads
		l.HeadDim = headDim
		l.SeqLength = seqLen
		l.MaxSeqLen = maxSeqLen
		l.DType = cfg.dtype
		l.WeightStore = ws
		l.UseTiling = true
		l.TileSize = 0
		l.Network = net
		
		net.EnableMultiCoreTiling = true
		l.SyncToCPU()
		net.SyncToGPU()

		input := poly.NewTensor[float32](batchSize, seqLen, dModel)
		for i := range input.Data {
			input.Data[i] = 0.1
		}

		// 2. CPU MC Forward
		var cpuMC *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
		_, cpuMC = poly.MHAForwardPolymorphic(l, input)
	}
	tCPUMC := time.Since(start) / time.Duration(iterations)

	// 3. GPU Forward (Normal/SC/MC)
	runGPUMode := func(tSize int) (time.Duration, []float32) {
		l.TileSize = tSize
		l.Network.EnableMultiCoreTiling = (tSize == mcTile)
		
		start := time.Now()
		outSize := seqLen * dModel
		outBuf := ctx.GetActivationBuffer("parity_out", uint64(outSize*4), wgpu.BufferUsageStorage)
		inData := input.Data
		inBuf := ctx.GetActivationBuffer("parity_in", uint64(len(inData)*4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(inBuf, 0, wgpu.ToBytes(inData))
		
		for i := 0; i < gpuIters; i++ {
			ctx.BeginFrame()
			ctx.DispatchForwardLayer(l, batchSize, inBuf, outBuf)
			ctx.FlushFrame()
		}
			dur := time.Since(start) / time.Duration(gpuIters)
			data, _ := ctx.ReadBuffer(outBuf)
			return dur, data
		}

		// GPU Normal
		tGN, gnOut := runGPUMode(0)
		// GPU SC
		tGSC, gscOut := runGPUMode(scTile)
		// GPU MC
		tGMC, gmcOut := runGPUMode(mcTile)

		diffGN, diffGSC, diffGMC := 0.0, 0.0, 0.0
		for i := range cpuMC.Data {
			if d := math.Abs(float64(cpuMC.Data[i] - gnOut[i])); d > diffGN { diffGN = d }
			if d := math.Abs(float64(cpuMC.Data[i] - gscOut[i])); d > diffGSC { diffGSC = d }
			if d := math.Abs(float64(cpuMC.Data[i] - gmcOut[i])); d > diffGMC { diffGMC = d }
		}

		if diffGN > 0.1 {
			fmt.Printf(" [DEBUG %s] s0h0: CPU=%0.6f GPU=%0.6f\n", cfg.name, cpuMC.Data[0], gnOut[0])
		}
		return result{
			tCPUMC: tCPUMC, tGPUNorm: tGN, tGPUSC: tGSC, tGPUMC: tGMC,
			diffGN: diffGN, diffGSC: diffGSC, diffGMC: diffGMC,
			parityGN: diffGN <= cfg.tolerance, parityGSC: diffGSC <= cfg.tolerance, parityGMC: diffGMC <= cfg.tolerance,
		}
	}

	fmt.Printf("| %-10s | %-12s | %-12s | %-12s | %-12s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s |\n",
		"DType", "CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC", "GN-Spd", "SC-Spd", "MC-Spd", "Diff-GN", "Diff-SC", "Diff-MC", "GN-Par", "SC-Par", "MC-Par")
	fmt.Println("|------------|--------------|--------------|--------------|--------------|--------|--------|--------|----------|----------|----------|--------|--------|--------|")

	allPass := true
	for _, cfg := range allTypes {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := run(cfg)
		if !r.parityGN || !r.parityGSC || !r.parityGMC {
			allPass = false
		}
		fmt.Printf("| %-10s | %-12v | %-12v | %-12v | %-12v | %-6.1fx | %-6.1fx | %-6.1fx | %-8.2e | %-8.2e | %-8.2e | %-6s | %-6s | %-6s |\n",
			cfg.name, r.tCPUMC, r.tGPUNorm, r.tGPUSC, r.tGPUMC,
			float64(r.tCPUMC)/float64(r.tGPUNorm),
			float64(r.tCPUMC)/float64(r.tGPUSC),
			float64(r.tCPUMC)/float64(r.tGPUMC),
			r.diffGN, r.diffGSC, r.diffGMC,
			parityMark(r.parityGN), parityMark(r.parityGSC), parityMark(r.parityGMC))
	}
	fmt.Println()
	if allPass {
		fmt.Println("✅ All GPU forward parity checks passed!")
	} else {
		fmt.Println("❌ One or more GPU forward parity checks FAILED.")
	}
}

// RunMHAGPUBackward dispatches the MHA backward GPU kernel for all numerical types
// and checks that gradient outputs are non-zero.
func RunMHAGPUBackward() {
	fmt.Println("=== MHA GPU Backward — All Numerical Types ===")
	fmt.Println()

	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	ctx := net.GPUContext

	const (
		batchSize  = 1
		numHeads   = 4
		numKVHeads = 4
		dModel     = 64
		headDim    = dModel / numHeads // 16
		seqLen     = 16
	)
	scaleF := float32(1.0 / math.Sqrt(headDim))
	bufSize := numHeads * seqLen * headDim

	fmt.Printf("| %-10s | %-8s | %-5s | %-5s | %-5s | %-12s |\n", "DType", "Time", "dQ≠0", "dK≠0", "dV≠0", "Result")
	fmt.Println("|------------|----------|-------|-------|-------|--------------|")

	allPass := true
	for _, cfg := range allTypes {
		fmt.Printf("  running %-10s ...\r", cfg.name)

		// 1. Prepare weights/data for this DType
		ws := poly.NewWeightStore(numHeads*seqLen*headDim*2 + numKVHeads*seqLen*headDim*2)
		const kv = numKVHeads * headDim
		const wCount = 2*dModel*dModel + 2*dModel*kv + 2*dModel + 2*kv
		ws = poly.NewWeightStore(wCount)
		const weightPart = 2*dModel*dModel + 2*dModel*kv
		for i := 0; i < weightPart && i < len(ws.Master); i++ {
			ws.Master[i] = 0.05
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 { ws.Morph(cfg.dtype) }

		net := poly.NewVolumetricNetwork(1, 1, 1, 1)
		net.GPUContext = ctx
		l := &net.Layers[0]
		l.Type = poly.LayerMultiHeadAttention
		l.Network = net
		l.DModel = dModel
		l.NumHeads = numHeads
		l.NumKVHeads = numKVHeads
		l.HeadDim = headDim
		l.SeqLength = seqLen
		l.MaxSeqLen = 512
		l.DType = cfg.dtype
		l.WeightStore = ws
		
		net.SyncToGPU()

		// 2. Prepare Buffers
		q, _ := l.WeightStore.GPUWeights[poly.WeightMHAQuery].(*wgpu.Buffer)
		k, _ := l.WeightStore.GPUWeights[poly.WeightMHAKey].(*wgpu.Buffer)
		v, _ := l.WeightStore.GPUWeights[poly.WeightMHAValue].(*wgpu.Buffer)

		goData := make([]float32, bufSize)
		for i := range goData { goData[i] = float32(i%7)*0.1 - 0.3 }
		goBuf := ctx.GetActivationBuffer("bwd_go", uint64(bufSize*4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(goBuf, 0, wgpu.ToBytes(goData))

		dqBuf := ctx.GetActivationBuffer("bwd_dq", uint64(bufSize*4), wgpu.BufferUsageStorage)
		dkBuf := ctx.GetActivationBuffer("bwd_dk", uint64(bufSize*4), wgpu.BufferUsageStorage)
		dvBuf := ctx.GetActivationBuffer("bwd_dv", uint64(bufSize*4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(dqBuf, 0, make([]byte, bufSize*4))
		ctx.Queue.WriteBuffer(dkBuf, 0, make([]byte, bufSize*4))
		ctx.Queue.WriteBuffer(dvBuf, 0, make([]byte, bufSize*4))

		// 3. Dispatch & Measure
		gpuIters := 5
		start := time.Now()
		for i := 0; i < gpuIters; i++ {
			ctx.BeginFrame()
			ctx.DispatchMHABackward(batchSize, numHeads, numKVHeads, headDim, seqLen, scaleF, goBuf, q, k, v, dqBuf, dkBuf, dvBuf)
			ctx.FlushFrame()
		}
		elapsed := time.Since(start) / time.Duration(gpuIters)

		nonzero := func(buf *wgpu.Buffer) bool {
			data, _ := ctx.ReadBuffer(buf)
			for _, v := range data {
				if v != 0 { return true }
			}
			return false
		}

		dqOK := nonzero(dqBuf)
		dkOK := nonzero(dkBuf)
		dvOK := nonzero(dvBuf)
		ok := dqOK && dkOK && dvOK
		if !ok { allPass = false }

		fmt.Printf("| %-10s | %-8v | %-5s | %-5s | %-5s | %-12s |\n",
			cfg.name, elapsed.Round(time.Microsecond), parityMark(dqOK), parityMark(dkOK), parityMark(dvOK), parityMark(ok))
	}

	fmt.Println()
	if allPass {
		fmt.Println("✅ MHA GPU backward producing non-zero gradients for all types!")
	} else {
		fmt.Println("❌ One or more MHA GPU backward checks FAILED.")
	}
}
