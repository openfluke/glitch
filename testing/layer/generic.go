package layer

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

type TestMode int

const (
	TestForward TestMode = 1 << iota
	TestBackward
	TestSaveLoad
	TestTraining
	TestAll = TestForward | TestBackward | TestSaveLoad | TestTraining
)

// TestSpec defines a standardized layer test configuration.
type TestSpec struct {
	Name       string
	Layer      poly.PersistenceLayerSpec
	InputShape []int // [Batch, ...]
}

// RunGenericLayerSuite executes forward/backward parity, save/reload, and training tests.
func RunGenericLayerSuite(spec TestSpec, mode TestMode) bool {
	fmt.Printf("\n--- [%s] Generic Layer Suite ---\n", spec.Name)
	stats.StartLayer()

	fullSpec := poly.PersistenceNetworkSpec{
		ID: "test_net", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
		Layers: []poly.PersistenceLayerSpec{spec.Layer},
	}
	fullSpec.Layers[0].Z, fullSpec.Layers[0].Y, fullSpec.Layers[0].X, fullSpec.Layers[0].L = 0, 0, 0, 0

	js, _ := json.Marshal(fullSpec)
	net, err := poly.DeserializeNetwork(js)
	if err != nil {
		fmt.Printf("Deserialization failed: %v\n", err)
		return false
	}
	l := net.GetLayer(0, 0, 0, 0)

	allPass := true

	if mode&TestForward != 0 {
		stats.ResetSub()
		allPass = runForwardSuite(spec, l) && allPass
		stats.ReportSub("Forward Parity")
	}

	if mode&TestBackward != 0 {
		stats.ResetSub()
		allPass = runBackwardSuite(spec, l) && allPass
		stats.ReportSub("Backward Parity")
	}

	if mode&TestSaveLoad != 0 {
		stats.ResetSub()
		allPass = runSaveReloadSuite(spec, l) && allPass
		stats.ReportSub("Save/Reload")
	}

	if mode&TestTraining != 0 {
		stats.ResetSub()
		allPass = runTrainingSuite(spec, l) && allPass
		stats.ReportSub("Training Matrix")
	}

	stats.ReportLayer(spec.Name)
	return allPass
}

func runForwardSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("\n=== %s GPU Forward - All Numerical Types ===\n\n", spec.Name)
	input := genInput(spec.InputShape)

	ctx := l.Network.GPUContext
	if ctx == nil {
		l.Network.InitWGPU()
		ctx = l.Network.GPUContext
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-6s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-8s | %-8s |\n",
		"DType", "Tile", "CPU Norm", "CPU Tile SC", "CPU Tile MC", "GPU Normal", "GPU Tile SC", "GPU Tile MC",
		"N->SC", "SC->MC", "MC->GN", "MC->GMC", "Diff-SC", "Diff-MC", "Diff-GN", "Diff-GSC", "Diff-GMC")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|--------------|--------------|--------|--------|--------|--------|----------|----------|----------|----------|----------|")

	allPass := true
	for _, cfg := range allTypes {
		l.DType = cfg.dtype
		if l.WeightStore != nil {
			l.WeightStore.Morph(cfg.dtype)
			l.WeightStore.Scale = cfg.scale
			l.SyncToCPU()
		}

		l.UseTiling = false
		l.EnableMultiCoreTiling = false
		t0 := time.Now()
		_, postCPU := poly.DispatchLayer(l, input, nil)
		tCPUNorm := time.Since(t0)

		l.UseTiling = true
		l.EnableMultiCoreTiling = false
		t0 = time.Now()
		_, postSC := poly.DispatchLayer(l, input, nil)
		tCPUSC := time.Since(t0)

		l.UseTiling = true
		l.EnableMultiCoreTiling = true
		t0 = time.Now()
		_, postMC := poly.DispatchLayer(l, input, nil)
		tCPUMC := time.Since(t0)

		l.Network.SyncToGPU()
		inBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "FwdIn",
			Contents: wgpu.ToBytes(input.Data),
			Usage:    wgpu.BufferUsageStorage,
		})
		outSize := len(postCPU.Data)
		outBufNorm, _ := zeroF32Buf(ctx, outSize, "FwdOutN")
		outBufSC, _ := zeroF32Buf(ctx, outSize, "FwdOutSC")
		outBufMC, _ := zeroF32Buf(ctx, outSize, "FwdOutMC")
		defer inBuf.Destroy()
		defer outBufNorm.Destroy()
		defer outBufSC.Destroy()
		defer outBufMC.Destroy()

		ctx.GPUTileSize = 0
		t0 = time.Now()
		ctx.DispatchForwardLayer(l, spec.InputShape[0], inBuf, outBufNorm)
		ctx.Device.Poll(true, nil)
		tGPUNorm := time.Since(t0)
		gpuNormData, _ := ctx.ReadBuffer(outBufNorm)

		ctx.GPUTileSize = l.GetGPUSCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchForwardLayer(l, spec.InputShape[0], inBuf, outBufSC)
		ctx.Device.Poll(true, nil)
		tGPUSC := time.Since(t0)
		gpuSCData, _ := ctx.ReadBuffer(outBufSC)

		ctx.GPUTileSize = l.GetGPUMCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchForwardLayer(l, spec.InputShape[0], inBuf, outBufMC)
		ctx.Device.Poll(true, nil)
		tGPUMC := time.Since(t0)
		gpuMCData, _ := ctx.ReadBuffer(outBufMC)

		diffSC := maxAbsDiff(postCPU.Data, postSC.Data)
		diffMC := maxAbsDiff(postCPU.Data, postMC.Data)
		diffGN := maxAbsDiff(postCPU.Data, gpuNormData)
		diffGSC := maxAbsDiff(postCPU.Data, gpuSCData)
		diffGMC := maxAbsDiff(postCPU.Data, gpuMCData)

		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-12v | %-12v | %-6.1fx | %-6.1fx | %-6.1fx | %-6.1fx | %-8.2e | %-8.2e | %-8.2e | %-8.2e | %-8.2e |\n",
			cfg.name, l.GetCPUTileSize(cfg.dtype), tCPUNorm, tCPUSC, tCPUMC, tGPUNorm, tGPUSC, tGPUMC,
			float64(tCPUNorm)/float64(tCPUSC),
			float64(tCPUSC)/float64(tCPUMC),
			float64(tCPUMC)/float64(tGPUNorm),
			float64(tCPUMC)/float64(tGPUMC),
			diffSC, diffMC, diffGN, diffGSC, diffGMC)

		if diffSC > 1e-10 || diffMC > 1e-10 || diffGN >= cfg.tolerance || diffGSC > 1e-10 || diffGMC > 1e-10 {
			allPass = false
		}
		stats.AddSpectrum(spectrumMark(diffSC, 1e-10, postSC.Data, postCPU.Data))
		stats.AddSpectrum(spectrumMark(diffMC, 1e-10, postMC.Data, postCPU.Data))
		stats.AddSpectrum(spectrumMark(diffGN, cfg.tolerance, gpuNormData, postCPU.Data))
		stats.AddSpectrum(spectrumMark(diffGSC, 1e-10, gpuSCData, postCPU.Data))
		stats.AddSpectrum(spectrumMark(diffGMC, 1e-10, gpuMCData, postCPU.Data))
		stats.AddPerf(spec.Name, cfg.name, "Forward", tCPUNorm, tGPUMC)
	}
	return allPass
}

func runBackwardSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("\n=== %s GPU Backward - All Numerical Types ===\n\n", spec.Name)
	input := genInput(spec.InputShape)
	l.DType = poly.DTypeFloat32
	l.WeightStore.Morph(poly.DTypeFloat32)
	l.SyncToCPU()

	pre, post := poly.DispatchLayer(l, input, nil)
	gradOut := genInput(post.Shape)

	l.EnableMultiCoreTiling = true

	ctx := l.Network.GPUContext
	if ctx == nil {
		l.Network.InitWGPU()
		ctx = l.Network.GPUContext
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-7s | %-7s | %-7s | %-9s | %-9s | %-9s | %-9s | %-4s | %-4s | %-4s |\n",
		"DType", "Tile", "CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd", "D-DX/N", "D-DW/N", "D-DX/SC", "D-DW/SC", "GN", "SC", "MC")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|---------|---------|---------|-----------|-----------|-----------|-----------|------|------|------|")

	allPass := true
	for _, cfg := range allTypes {
		l.DType = cfg.dtype
		if l.WeightStore != nil {
			l.WeightStore.Morph(cfg.dtype)
			l.WeightStore.Scale = cfg.scale
		}
		l.Network.SyncToGPU()

		l.UseTiling = true
		l.EnableMultiCoreTiling = true
		t0 := time.Now()
		cpuDX, cpuDW := poly.DispatchLayerBackward(l, gradOut, input, nil, pre)
		tCPUMC := time.Since(t0)

		inBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "BwdIn",
			Contents: wgpu.ToBytes(input.Data),
			Usage:    wgpu.BufferUsageStorage,
		})
		goBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "BwdGO",
			Contents: wgpu.ToBytes(gradOut.Data),
			Usage:    wgpu.BufferUsageStorage,
		})
		preBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "BwdPre",
			Contents: wgpu.ToBytes(pre.Data),
			Usage:    wgpu.BufferUsageStorage,
		})

		dxBufN, _ := zeroF32Buf(ctx, len(cpuDX.Data), "dxN")
		dwSize := len(cpuDW.Data)
		if l.Type == poly.LayerResidual {
			dwSize = len(cpuDX.Data)
		}
		dwBufN, _ := zeroF32Buf(ctx, dwSize, "dwN")
		dxBufSC, _ := zeroF32Buf(ctx, len(cpuDX.Data), "dxSC")
		dwBufSC, _ := zeroF32Buf(ctx, dwSize, "dwSC")
		dxBufMC, _ := zeroF32Buf(ctx, len(cpuDX.Data), "dxMC")
		dwBufMC, _ := zeroF32Buf(ctx, dwSize, "dwMC")
		defer inBuf.Destroy()
		defer goBuf.Destroy()
		defer preBuf.Destroy()
		defer dxBufN.Destroy()
		defer dwBufN.Destroy()
		defer dxBufSC.Destroy()
		defer dwBufSC.Destroy()
		defer dxBufMC.Destroy()
		defer dwBufMC.Destroy()

		ctx.GPUTileSize = 0
		t0 = time.Now()
		ctx.DispatchBackwardLayer(l, spec.InputShape[0], goBuf, inBuf, preBuf, dxBufN, dwBufN)
		ctx.Device.Poll(true, nil)
		tGPUNorm := time.Since(t0)
		gDXN, _ := ctx.ReadBuffer(dxBufN)
		gDWN, _ := ctx.ReadBuffer(dwBufN)

		ctx.GPUTileSize = l.GetGPUSCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchBackwardLayer(l, spec.InputShape[0], goBuf, inBuf, preBuf, dxBufSC, dwBufSC)
		ctx.Device.Poll(true, nil)
		tGPUSC := time.Since(t0)
		gDXSC, _ := ctx.ReadBuffer(dxBufSC)
		gDWSC, _ := ctx.ReadBuffer(dwBufSC)

		ctx.GPUTileSize = l.GetGPUMCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchBackwardLayer(l, spec.InputShape[0], goBuf, inBuf, preBuf, dxBufMC, dwBufMC)
		ctx.Device.Poll(true, nil)
		tGPUMC := time.Since(t0)
		gDXMC, _ := ctx.ReadBuffer(dxBufMC)
		gDWMC, _ := ctx.ReadBuffer(dwBufMC)

		dxDiffN := maxAbsDiff(cpuDX.Data, gDXN)
		dwDiffN := maxAbsDiff(cpuDW.Data, gDWN)
		dxDiffSC := maxAbsDiff(cpuDX.Data, gDXSC)
		dwDiffSC := maxAbsDiff(cpuDW.Data, gDWSC)
		dxDiffMC := maxAbsDiff(cpuDX.Data, gDXMC)
		dwDiffMC := maxAbsDiff(cpuDW.Data, gDWMC)

		okN := dxDiffN < cfg.tolerance*5 && dwDiffN < cfg.tolerance*5
		okSC := dxDiffSC < cfg.tolerance*5 && dwDiffSC < cfg.tolerance*5
		okMC := dxDiffMC < cfg.tolerance*5 && dwDiffMC < cfg.tolerance*5

		mN := spectrumMark(dxDiffN+dwDiffN, cfg.tolerance*10, gDXN, cpuDX.Data)
		mSC := spectrumMark(dxDiffSC+dwDiffSC, cfg.tolerance*10, gDXSC, cpuDX.Data)
		mMC := spectrumMark(dxDiffMC+dwDiffMC, cfg.tolerance*10, gDXMC, cpuDX.Data)

		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-7.1fx | %-7.1fx | %-7.1fx | %-9.2e | %-9.2e | %-9.2e | %-9.2e | %-8s | %-8s | %-8s |\n",
			cfg.name, l.GetCPUTileSize(cfg.dtype), tCPUMC, tGPUNorm, tGPUSC, tGPUMC,
			float64(tCPUMC)/float64(tGPUNorm),
			float64(tCPUMC)/float64(tGPUSC),
			float64(tCPUMC)/float64(tGPUMC),
			dxDiffN, dwDiffN, dxDiffSC, dwDiffSC,
			mN, mSC, mMC)

		if !okN || !okSC || !okMC {
			allPass = false
		}
		stats.AddSpectrum(mN)
		stats.AddSpectrum(mSC)
		stats.AddSpectrum(mMC)
		stats.AddPerf(spec.Name, cfg.name, "Backward", tCPUMC, tGPUMC)
	}
	return allPass
}

func runSaveReloadSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("  [Save/Reload %s] ", spec.Name)
	input := genInput(spec.InputShape)
	l.DType = poly.DTypeFloat32
	_, post1 := poly.DispatchLayer(l, input, nil)

	js, _ := poly.SerializeNetwork(l.Network)
	net2, err := poly.DeserializeNetwork(js)
	if err != nil {
		fmt.Printf("FAIL: %v\n", err)
		return false
	}

	l2 := net2.GetLayer(0, 0, 0, 0)
	_, post2 := poly.DispatchLayer(l2, input, nil)

	weightsMatch := true
	if l.WeightStore != nil && l2.WeightStore != nil {
		weightsMatch = maxAbsDiff(l.WeightStore.Master, l2.WeightStore.Master) < 1e-6
	} else if (l.WeightStore == nil) != (l2.WeightStore == nil) {
		weightsMatch = false
	}
	diff := maxAbsDiff(post1.Data, post2.Data)

	ok := diff < 1e-6 && weightsMatch
	if ok {
		fmt.Println("PASS")
	} else {
		fmt.Printf("FAIL (Diff: %.2e, Weights: %v)\n", diff, weightsMatch)
	}
	stats.AddSpectrum(spectrumMark(diff, 1e-6, post1.Data, post2.Data))
	return ok
}

func runTrainingSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("\n=== %s Training - All Modes x All Numerical Types ===\n\n", spec.Name)
	input := genInput(spec.InputShape)
	_, postBaseline := poly.DispatchLayer(l, input, nil)
	target := genInput(postBaseline.Shape)

	batch := poly.TrainingBatch[float32]{Input: input, Target: target}
	allModes := []poly.TrainingMode{
		poly.TrainingModeCPUNormal, poly.TrainingModeCPUSC, poly.TrainingModeCPUMC,
		poly.TrainingModeGPUNormal, poly.TrainingModeGPUSC, poly.TrainingModeGPUMC,
	}

	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "TrainOK", "Save/Reload", "File", "RAM")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")

	overallPass := true
	for _, cfg := range allTypes {
		for _, mode := range allModes {
			l.DType = cfg.dtype
			if l.WeightStore != nil {
				l.WeightStore.Morph(cfg.dtype)
				l.WeightStore.Scale = cfg.scale
				l.SyncToCPU()
			}
			if mode.IsGPU() && l.Network.GPUContext == nil {
				continue
			}

			var w0 []float32
			if l.WeightStore != nil {
				w0 = make([]float32, len(l.WeightStore.Master))
				copy(w0, l.WeightStore.Master)
			}

			tcfg := poly.DefaultTrainingConfig()
			tcfg.Epochs = 5
			tcfg.Mode = mode
			tcfg.Verbose = false
			tcfg.LearningRate = 0.01

			start := time.Now()
			res, err := poly.Train(l.Network, []poly.TrainingBatch[float32]{batch}, tcfg)
			dur := time.Since(start)

			if err != nil {
				fmt.Printf("| %-10s | %-13s | ERR        | ERR        | %-8v | ERR     | %s\n", cfg.name, mode.String(), dur.Round(time.Millisecond), err)
				overallPass = false
				continue
			}

			var trainOK, saveOK bool
			if l.WeightStore != nil {
				if mode.IsGPU() {
					poly.SyncWeightsFromGPU(l.Network)
				}
				wt := l.WeightStore.Master
				trainNoNaN := !math.IsNaN(res.FinalLoss) && !math.IsNaN(res.LossHistory[0])
				trainImproved := res.FinalLoss < res.LossHistory[0]
				trainOK = trainNoNaN && trainImproved

				js, _ := poly.SerializeNetwork(l.Network)
				net2, _ := poly.DeserializeNetwork(js)
				l2 := net2.GetLayer(0, 0, 0, 0)
				wr := l2.WeightStore.Master

				expected := wt
				if cfg.dtype != poly.DTypeFloat32 {
					expected = make([]float32, len(wt))
					for i, v := range wt {
						expected[i] = poly.SimulatePrecision(v, cfg.dtype, l.WeightStore.Scale)
					}
				}
				saveOK = maxAbsDiff(wr, expected) < 1e-4

				ramBytes := int64(len(wt)*4) + int64(math.Ceil(float64(len(wt)*poly.DTypeBits(cfg.dtype))/8.0))
				fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
					cfg.name, mode.String(), res.LossHistory[0], res.FinalLoss, dur.Round(time.Millisecond),
					markMark(trainOK), markMark(saveOK), float64(len(js))/1024.0, float64(ramBytes)/1024.0)
			} else {
				trainOK, saveOK = true, true
				fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7v | %-11v | %-8s | %-8s |\n",
					cfg.name, mode.String(), res.LossHistory[0], res.FinalLoss, dur.Round(time.Millisecond),
					"N/A", "N/A", "0KB", "0KB")
			}

			if !trainOK || !saveOK {
				overallPass = false
			}

			if trainOK {
				stats.AddSpectrum(SpecExact)
			} else {
				stats.AddSpectrum(SpecBroken)
			}
			if saveOK {
				stats.AddSpectrum(SpecExact)
			} else {
				stats.AddSpectrum(SpecBroken)
			}
		}
		fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")
	}
	return overallPass
}

func markMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
