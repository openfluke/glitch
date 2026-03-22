// Package layer provides layer-level tests for glitch testing mode.
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

// ── shared helpers ────────────────────────────────────────────────────────────

type typeConfig struct {
	name      string
	dtype     poly.DType
	scale     float32
	tolerance float64
}

var allTypes = []typeConfig{
	{"Float64", poly.DTypeFloat64, 1.0, 1e-3},
	{"Float32", poly.DTypeFloat32, 1.0, 1e-5},
	{"Float16", poly.DTypeFloat16, 1.0, 1e-3},
	{"BFloat16", poly.DTypeBFloat16, 1.0, 1e-3},
	{"FP8-E4M3", poly.DTypeFP8E4M3, 0.01, 1e-3},
	{"FP8-E5M2", poly.DTypeFP8E5M2, 0.01, 1e-3},
	{"Int64", poly.DTypeInt64, 0.01, 1e-3},
	{"Uint64", poly.DTypeUint64, 0.01, 1e-3},
	{"Int32", poly.DTypeInt32, 0.01, 1e-3},
	{"Uint32", poly.DTypeUint32, 0.01, 1e-3},
	{"Int16", poly.DTypeInt16, 0.01, 1e-3},
	{"Uint16", poly.DTypeUint16, 0.01, 1e-3},
	{"Int8", poly.DTypeInt8, 0.01, 1e-3},
	{"Uint8", poly.DTypeUint8, 0.01, 1e-3},
	{"Int4", poly.DTypeInt4, 0.01, 1e-3},
	{"Uint4", poly.DTypeUint4, 0.01, 1e-3},
	{"FP4", poly.DTypeFP4, 0.01, 1e-3},
	{"Int2", poly.DTypeInt2, 0.01, 1e-3},
	{"Uint2", poly.DTypeUint2, 0.01, 1e-3},
	{"Ternary", poly.DTypeTernary, 0.1, 1e-3},
	{"Binary", poly.DTypeBinary, 0.1, 1e-3},
}

func parityMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func maxAbsDiff(a, b []float32) float64 {
	var d float64
	for i := range a {
		if v := math.Abs(float64(a[i] - b[i])); v > d {
			d = v
		}
	}
	return d
}

// rawF32 returns the active weight buffer as []float32 without applying scale.
func rawF32(ws *poly.WeightStore, dtype poly.DType) []float32 {
	active := ws.GetActive(dtype)
	if active == nil {
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}
	switch w := active.(type) {
	case []float32:
		out := make([]float32, len(w))
		copy(out, w)
		return out
	case []float64:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int64:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int32:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int16:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int8:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	default:
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}
}

func zeroF32Buf(ctx *poly.WGPUContext, size int, label string) (*wgpu.Buffer, error) {
	zeros := make([]float32, size)
	return ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label,
		Contents: wgpu.ToBytes(zeros),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
}

// ── CNN3 Tests ────────────────────────────────────────────────────────────────

// RunCNN3L1Caching benchmarks normal vs single-core-tiled vs multi-core-tiled
// forward pass across all numeric types (CPU only).
func RunCNN3L1Caching() {
	fmt.Println("=== CNN3 Multi-Core L1 Caching — All Numerical Types ===")
	iterations := 3

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
		ws := poly.NewWeightStore(32 * 32 * 3 * 3 * 3)
		for i := range ws.Master {
			ws.Master[i] = 0.1
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		l := poly.VolumetricLayer{
			Network:       poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:          poly.LayerCNN3,
			InputChannels: 32, InputDepth: 32, InputHeight: 32, InputWidth: 32,
			Filters: 32, OutputDepth: 32, OutputHeight: 32, OutputWidth: 32,
			KernelSize: 3, Stride: 1, Padding: 1,
			DType: cfg.dtype, WeightStore: ws,
		}
		input := poly.NewTensor[float32](1, 32, 32, 32, 32)
		for i := range input.Data {
			input.Data[i] = 0.5
		}

		l.UseTiling = false
		var post0 *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, post0 = poly.CNN3ForwardPolymorphic(&l, input)
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
			_, post1 = poly.CNN3ForwardPolymorphic(&l, input)
		}
		tSingle := time.Since(start) / time.Duration(iterations)

		l.Network.EnableMultiCoreTiling = true
		l.SyncToCPU()
		var post2 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, post2 = poly.CNN3ForwardPolymorphic(&l, input)
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

// RunCNN3Training tests 6 training modes × 21 numeric types (train + save/reload).
func RunCNN3Training() {
	const (
		batchSz  = 1
		inC      = 4
		inD, inH, inW = 4, 4, 4
		filters  = 4
		kSize    = 3
		stride   = 1
		padding  = 1
		outD, outH, outW = inD, inH, inW
		epochs   = 5
	)

	type layerSpec struct {
		Z int    `json:"z"`
		Y int    `json:"y"`
		X int    `json:"x"`
		L int    `json:"l"`
		Type          string `json:"type"`
		Activation    string `json:"activation"`
		DType         string `json:"dtype"`
		InputChannels int    `json:"input_channels"`
		InputDepth    int    `json:"input_depth"`
		InputHeight   int    `json:"input_height"`
		InputWidth    int    `json:"input_width"`
		Filters       int    `json:"filters"`
		OutputDepth   int    `json:"output_depth"`
		OutputHeight  int    `json:"output_height"`
		OutputWidth   int    `json:"output_width"`
		KernelSize    int    `json:"kernel_size"`
		Stride        int    `json:"stride"`
		Padding       int    `json:"padding"`
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
			ID: "cnn3_train_test", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
			Layers: []layerSpec{{
				Type: "CNN3", Activation: "LINEAR", DType: "FLOAT32",
				InputChannels: inC, InputDepth: inD, InputHeight: inH, InputWidth: inW,
				Filters: filters, OutputDepth: outD, OutputHeight: outH, OutputWidth: outW,
				KernelSize: kSize, Stride: stride, Padding: padding,
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
		inp := poly.NewTensor[float32](batchSz, inC, inD, inH, inW)
		for i := range inp.Data {
			inp.Data[i] = float32(i%13)*0.1 - 0.6
		}
		tgt := poly.NewTensor[float32](batchSz, filters, outD, outH, outW)
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
		tmp, err := os.CreateTemp("", "poly_cnn3_*.json")
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

	fmt.Println("=== CNN3 Training — All Modes × All Numerical Types ===")
	fmt.Println()

	testNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	gpuAvail := testNet.InitWGPU() == nil
	if gpuAvail {
		sc, mc := poly.CNN3GPUTileSizes(testNet.GPUContext)
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
		fmt.Println("✅ All training + save/reload checks passed!")
	} else {
		fmt.Println("❌ One or more checks FAILED.")
	}
}

// RunCNN3GPUForward compares CPU multi-core tiled vs GPU Normal/SC/MC forward pass.
func RunCNN3GPUForward() {
	fmt.Println("=== CNN3 GPU Forward — All Numerical Types ===")
	fmt.Println()

	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	ctx := net.GPUContext
	scTile, mcTile := poly.CNN3GPUTileSizes(ctx)
	fmt.Printf("GPU ready — SC tile=%d  MC tile=%d  MaxInvocations=%d\n\n",
		scTile, mcTile, ctx.Limits.MaxComputeInvocationsPerWorkgroup)

	type result struct {
		tCPUMC, tGPUNorm, tGPUSC, tGPUMC time.Duration
		tileSize, scTile, mcTile          int
		diffGN, diffGSC, diffGMC          float64
		parityGN, parityGSC, parityGMC    bool
	}

	const (
		inC, inD, inH, inW     = 32, 32, 32, 32
		outC, outD, outH, outW = 32, 32, 32, 32
		kD, kH, kW             = 3, 3, 3
		sD, sH, sW             = 1, 1, 1
		pD, pH, pW             = 1, 1, 1
		batchSize              = 1
		outputSize             = batchSize * outC * outD * outH * outW
		kernelVol              = inC * kD * kH * kW
	)

	iterations := 3

	run := func(cfg typeConfig) result {
		ws := poly.NewWeightStore(32 * 32 * 3 * 3 * 3)
		for i := range ws.Master {
			ws.Master[i] = 0.1
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		l := poly.VolumetricLayer{
			Network:       poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:          poly.LayerCNN3,
			InputChannels: inC, InputDepth: inD, InputHeight: inH, InputWidth: inW,
			Filters: outC, OutputDepth: outD, OutputHeight: outH, OutputWidth: outW,
			KernelSize: kD, Stride: 1, Padding: 1,
			DType: cfg.dtype, WeightStore: ws, UseTiling: true, TileSize: 0,
		}
		l.Network.EnableMultiCoreTiling = true
		l.SyncToCPU()
		tileSize := l.TileSize

		input := poly.NewTensor[float32](batchSize, inC, inD, inH, inW)
		for i := range input.Data {
			input.Data[i] = 0.5
		}

		var cpuMC *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, cpuMC = poly.CNN3ForwardPolymorphic(&l, input)
		}
		tCPUMC := time.Since(start) / time.Duration(iterations)

		raw := rawF32(ws, cfg.dtype)

		inputBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: "FwdInput", Contents: wgpu.ToBytes(input.Data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			fmt.Printf("  input buf: %v\n", err)
			return result{}
		}
		defer inputBuf.Destroy()

		rawWeightBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: "FwdWeights", Contents: wgpu.ToBytes(raw),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			fmt.Printf("  weight buf: %v\n", err)
			return result{}
		}
		defer rawWeightBuf.Destroy()

		outputBuf, err := ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "FwdOutput", Size: uint64(outputSize * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			fmt.Printf("  output buf: %v\n", err)
			return result{}
		}
		defer outputBuf.Destroy()

		// Warmup
		ctx.DispatchCNN3Scaled(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
		ctx.DispatchCNN3Tiled(scTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
		ctx.DispatchCNN3Tiled(mcTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
		ctx.Device.Poll(true, nil)

		gpuIters := 10

		start = time.Now()
		for i := 0; i < gpuIters; i++ {
			ctx.DispatchCNN3Scaled(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
		}
		ctx.Device.Poll(true, nil)
		tGPUNorm := time.Since(start) / time.Duration(gpuIters)
		gpuNormOut, _ := ctx.ReadBuffer(outputBuf)

		start = time.Now()
		for i := 0; i < gpuIters; i++ {
			ctx.DispatchCNN3Tiled(scTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
		}
		ctx.Device.Poll(true, nil)
		tGPUSC := time.Since(start) / time.Duration(gpuIters)
		gpuSCOut, _ := ctx.ReadBuffer(outputBuf)

		start = time.Now()
		for i := 0; i < gpuIters; i++ {
			ctx.DispatchCNN3Tiled(mcTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
		}
		ctx.Device.Poll(true, nil)
		tGPUMC := time.Since(start) / time.Duration(gpuIters)
		gpuMCOut, _ := ctx.ReadBuffer(outputBuf)

		diffGN, diffGSC, diffGMC := 0.0, 0.0, 0.0
		for i := range cpuMC.Data {
			if d := math.Abs(float64(cpuMC.Data[i] - gpuNormOut[i])); d > diffGN {
				diffGN = d
			}
			if d := math.Abs(float64(cpuMC.Data[i] - gpuSCOut[i])); d > diffGSC {
				diffGSC = d
			}
			if d := math.Abs(float64(cpuMC.Data[i] - gpuMCOut[i])); d > diffGMC {
				diffGMC = d
			}
		}

		return result{
			tCPUMC: tCPUMC, tGPUNorm: tGPUNorm, tGPUSC: tGPUSC, tGPUMC: tGPUMC,
			tileSize: tileSize, scTile: scTile, mcTile: mcTile,
			diffGN: diffGN, diffGSC: diffGSC, diffGMC: diffGMC,
			parityGN: diffGN <= cfg.tolerance, parityGSC: diffGSC <= cfg.tolerance, parityGMC: diffGMC <= cfg.tolerance,
		}
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s |\n",
		"DType", "Tile", "CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd", "Diff-GN", "Diff-SC", "Diff-MC", "GN-Par", "SC-Par", "MC-Par")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|--------|--------|--------|----------|----------|----------|--------|--------|--------|")

	allPass := true
	for _, cfg := range allTypes {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := run(cfg)
		if !r.parityGN || !r.parityGSC || !r.parityGMC {
			allPass = false
		}
		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-6.1fx | %-6.1fx | %-6.1fx | %-8.2e | %-8.2e | %-8.2e | %-6s | %-6s | %-6s |\n",
			cfg.name, r.tileSize, r.tCPUMC, r.tGPUNorm, r.tGPUSC, r.tGPUMC,
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

// RunCNN3GPUBackward compares CPU multi-core tiled vs GPU Normal/SC/MC backward pass.
func RunCNN3GPUBackward() {
	fmt.Println("=== CNN3 GPU Backward — All Numerical Types ===")
	fmt.Println()

	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	ctx := net.GPUContext
	scTile, mcTile := poly.CNN3GPUTileSizes(ctx)
	fmt.Printf("GPU ready — SC tile=%d  MC tile=%d\n\n", scTile, mcTile)

	type result struct {
		tCPUMC, tGPUNorm, tGPUSC, tGPUMC time.Duration
		scTile, mcTile, tileSize          int
		diffDXNorm, diffDWNorm            float64
		diffDXSC, diffDWSC                float64
		diffDXMC, diffDWMC                float64
		parityNorm, paritySC, parityMC    bool
	}

	const (
		inC, inD, inH, inW     = 32, 32, 32, 32
		outC, outD, outH, outW = 32, 32, 32, 32
		kSize, stride, padding = 3, 1, 1
		kD, kH, kW             = 3, 3, 3
		sD, sH, sW             = 1, 1, 1
		pD, pH, pW             = 1, 1, 1
		batchSize              = 1
		inputSize              = batchSize * inC * inD * inH * inW
		outputSize             = batchSize * outC * outD * outH * outW
		kernelVol              = inC * kD * kH * kW
		weightSize             = outC * kernelVol
	)

	run := func(cfg typeConfig) result {
		ws := poly.NewWeightStore(32 * 32 * 3 * 3 * 3)
		for i := range ws.Master {
			ws.Master[i] = 0.1
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		l := poly.VolumetricLayer{
			Network:       poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:          poly.LayerCNN3,
			InputChannels: inC, InputDepth: inD, InputHeight: inH, InputWidth: inW,
			Filters: outC, OutputDepth: outD, OutputHeight: outH, OutputWidth: outW,
			KernelSize: kD, Stride: 1, Padding: 1,
			Activation: poly.ActivationLinear,
			DType:      cfg.dtype, WeightStore: ws, UseTiling: true, TileSize: 8,
		}
		l.Network.EnableMultiCoreTiling = true
		l.SyncToCPU()
		tileSize := l.TileSize

		gradOut := poly.NewTensor[float32](batchSize, outC, outD, outH, outW)
		for i := range gradOut.Data {
			gradOut.Data[i] = 1.0
		}
		input := poly.NewTensor[float32](batchSize, inC, inD, inH, inW)
		for i := range input.Data {
			input.Data[i] = 0.5
		}
		preAct := poly.NewTensor[float32](batchSize, outC, outD, outH, outW)
		for i := range preAct.Data {
			preAct.Data[i] = 0.5
		}

		var cpuDX, cpuDW *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < 3; i++ {
			cpuDX, cpuDW = poly.CNN3BackwardTiledParallel(&l, gradOut, input, preAct)
		}
		tCPUMC := time.Since(start) / 3

		raw := rawF32(ws, cfg.dtype)
		act := poly.ActivationLinear

		gradOutBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "BwdGO", Contents: wgpu.ToBytes(gradOut.Data), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		weightBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "BwdW", Contents: wgpu.ToBytes(raw), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		inputBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "BwdIn", Contents: wgpu.ToBytes(input.Data), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		preActBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "BwdPA", Contents: wgpu.ToBytes(preAct.Data), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		defer gradOutBuf.Destroy()
		defer weightBuf.Destroy()
		defer inputBuf.Destroy()
		defer preActBuf.Destroy()

		timingGI, _ := zeroF32Buf(ctx, inputSize, "TimingGI")
		timingGW, _ := zeroF32Buf(ctx, weightSize, "TimingGW")
		defer timingGI.Destroy()
		defer timingGW.Destroy()

		// Warmup
		{
			gi, _ := zeroF32Buf(ctx, inputSize, "WarmGI")
			gw, _ := zeroF32Buf(ctx, weightSize, "WarmGW")
			ctx.DispatchCNN3BackwardDX(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kSize, stride, padding, act, gradOutBuf, weightBuf, preActBuf, gi)
			ctx.DispatchCNN3BackwardDW(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kSize, stride, padding, act, gradOutBuf, inputBuf, preActBuf, gw)
			ctx.DispatchCNN3TiledBackwardDX(scTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
			ctx.DispatchCNN3TiledBackwardDW(scTile, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
			ctx.DispatchCNN3TiledBackwardDX(mcTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
			ctx.DispatchCNN3TiledBackwardDW(mcTile, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
			ctx.Device.Poll(true, nil)
			gi.Destroy()
			gw.Destroy()
		}

		gpuIters := 10

		var normTotal time.Duration
		for i := 0; i < gpuIters; i++ {
			t0 := time.Now()
			ctx.DispatchCNN3BackwardDX(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kSize, stride, padding, act, gradOutBuf, weightBuf, preActBuf, timingGI)
			ctx.DispatchCNN3BackwardDW(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kSize, stride, padding, act, gradOutBuf, inputBuf, preActBuf, timingGW)
			ctx.Device.Poll(true, nil)
			normTotal += time.Since(t0)
		}
		tGPUNorm := normTotal / time.Duration(gpuIters)

		var scTotal time.Duration
		for i := 0; i < gpuIters; i++ {
			t0 := time.Now()
			ctx.DispatchCNN3TiledBackwardDX(scTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, timingGI)
			ctx.DispatchCNN3TiledBackwardDW(scTile, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, timingGW)
			ctx.Device.Poll(true, nil)
			scTotal += time.Since(t0)
		}
		tGPUSC := scTotal / time.Duration(gpuIters)

		var mcTotal time.Duration
		for i := 0; i < gpuIters; i++ {
			t0 := time.Now()
			ctx.DispatchCNN3TiledBackwardDX(mcTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, timingGI)
			ctx.DispatchCNN3TiledBackwardDW(mcTile, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, timingGW)
			ctx.Device.Poll(true, nil)
			mcTotal += time.Since(t0)
		}
		tGPUMC := mcTotal / time.Duration(gpuIters)

		readParity := func(dx func(*wgpu.Buffer), dw func(*wgpu.Buffer)) ([]float32, []float32) {
			gi, _ := zeroF32Buf(ctx, inputSize, "PGI")
			gw, _ := zeroF32Buf(ctx, weightSize, "PGW")
			defer gi.Destroy()
			defer gw.Destroy()
			dx(gi)
			dw(gw)
			ctx.Device.Poll(true, nil)
			giData, _ := ctx.ReadBuffer(gi)
			gwData, _ := ctx.ReadBuffer(gw)
			return giData, gwData
		}

		giNorm, gwNorm := readParity(
			func(gi *wgpu.Buffer) {
				ctx.DispatchCNN3BackwardDX(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kSize, stride, padding, act, gradOutBuf, weightBuf, preActBuf, gi)
			},
			func(gw *wgpu.Buffer) {
				ctx.DispatchCNN3BackwardDW(batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kSize, stride, padding, act, gradOutBuf, inputBuf, preActBuf, gw)
			},
		)
		giSC, gwSC := readParity(
			func(gi *wgpu.Buffer) {
				ctx.DispatchCNN3TiledBackwardDX(scTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
			},
			func(gw *wgpu.Buffer) {
				ctx.DispatchCNN3TiledBackwardDW(scTile, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
			},
		)
		giMC, gwMC := readParity(
			func(gi *wgpu.Buffer) {
				ctx.DispatchCNN3TiledBackwardDX(mcTile, kernelVol, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
			},
			func(gw *wgpu.Buffer) {
				ctx.DispatchCNN3TiledBackwardDW(mcTile, batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
			},
		)

		dxNorm := maxAbsDiff(cpuDX.Data, giNorm)
		dwNorm := maxAbsDiff(cpuDW.Data, gwNorm)
		dxSC := maxAbsDiff(cpuDX.Data, giSC)
		dwSC := maxAbsDiff(cpuDW.Data, gwSC)
		dxMC := maxAbsDiff(cpuDX.Data, giMC)
		dwMC := maxAbsDiff(cpuDW.Data, gwMC)

		return result{
			tCPUMC: tCPUMC, tGPUNorm: tGPUNorm, tGPUSC: tGPUSC, tGPUMC: tGPUMC,
			scTile: scTile, mcTile: mcTile, tileSize: tileSize,
			diffDXNorm: dxNorm, diffDWNorm: dwNorm,
			diffDXSC: dxSC, diffDWSC: dwSC,
			diffDXMC: dxMC, diffDWMC: dwMC,
			parityNorm: math.Max(dxNorm, dwNorm) <= cfg.tolerance,
			paritySC:   math.Max(dxSC, dwSC) <= cfg.tolerance,
			parityMC:   math.Max(dxMC, dwMC) <= cfg.tolerance,
		}
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-7s | %-7s | %-7s | %-9s | %-9s | %-9s | %-9s | %-6s | %-6s | %-6s |\n",
		"DType", "Tile", "CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd",
		"Diff-DX/N", "Diff-DW/N", "Diff-DX/SC", "Diff-DW/SC",
		"GN", "SC", "MC")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|---------|---------|---------|-----------|-----------|-----------|-----------|--------|--------|--------|")

	allPass := true
	for _, cfg := range allTypes {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := run(cfg)
		if !r.parityNorm || !r.paritySC || !r.parityMC {
			allPass = false
		}
		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-7.1fx | %-7.1fx | %-7.1fx | %-9.2e | %-9.2e | %-9.2e | %-9.2e | %-6s | %-6s | %-6s |\n",
			cfg.name, r.tileSize, r.tCPUMC, r.tGPUNorm, r.tGPUSC, r.tGPUMC,
			float64(r.tCPUMC)/float64(r.tGPUNorm),
			float64(r.tCPUMC)/float64(r.tGPUSC),
			float64(r.tCPUMC)/float64(r.tGPUMC),
			r.diffDXNorm, r.diffDWNorm, r.diffDXSC, r.diffDWSC,
			parityMark(r.parityNorm), parityMark(r.paritySC), parityMark(r.parityMC))
	}
	fmt.Println()
	if allPass {
		fmt.Println("✅ All GPU backward parity checks passed!")
	} else {
		fmt.Println("❌ One or more GPU backward parity checks FAILED.")
	}
}
