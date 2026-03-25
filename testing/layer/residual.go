package layer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
)

// ── Residual Tests ─────────────────────────────────────────────────────────────

// RunResidualL1Caching benchmarks normal vs single-core-tiled vs multi-core-tiled
// Residual (skip-connection add) forward pass across all numeric types (CPU only).
func RunResidualL1Caching() {
	fmt.Println("=== Residual Multi-Core L1 Caching — All Numerical Types ===")
	iterations := 3

	const size = 1024

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
		// Residual has no learnable weights; a minimal WeightStore avoids nil panics.
		ws := poly.NewWeightStore(1)
		ws.Scale = cfg.scale

		l := poly.VolumetricLayer{
			Network:     poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:        poly.LayerResidual,
			DType:       cfg.dtype,
			WeightStore: ws,
		}
		input := poly.NewTensor[float32](size)
		skip := poly.NewTensor[float32](size)
		for i := range input.Data {
			input.Data[i] = float32(i%11)*0.09 - 0.45
			skip.Data[i] = float32(i%7)*0.1 - 0.3
		}

		l.UseTiling = false
		var post0 *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, post0 = poly.ResidualForwardPolymorphic(&l, input, skip)
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
			_, post1 = poly.ResidualForwardPolymorphic(&l, input, skip)
		}
		tSingle := time.Since(start) / time.Duration(iterations)

		l.Network.EnableMultiCoreTiling = true
		l.SyncToCPU()
		var post2 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, post2 = poly.ResidualForwardPolymorphic(&l, input, skip)
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

// RunResidualGPUForward checks GPU availability and reports that GPU kernels for Residual
// are not yet implemented.
func RunResidualGPUForward() {
	fmt.Println("=== Residual GPU Forward — All Numerical Types ===")
	fmt.Println()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	defer net.DestroyWGPU()
	fmt.Println("GPU kernels for Residual are not yet implemented.")
	fmt.Println("This test is a placeholder for future GPU support.")
}

// RunResidualGPUBackward checks GPU availability and reports that GPU kernels for Residual
// are not yet implemented.
func RunResidualGPUBackward() {
	fmt.Println("=== Residual GPU Backward — All Numerical Types ===")
	fmt.Println()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	defer net.DestroyWGPU()
	fmt.Println("GPU kernels for Residual are not yet implemented.")
	fmt.Println("This test is a placeholder for future GPU support.")
}
