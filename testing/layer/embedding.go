package layer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
)

// ── Embedding Tests ────────────────────────────────────────────────────────────

// RunEmbeddingL1Caching benchmarks normal vs single-core-tiled vs multi-core-tiled
// Embedding forward pass across all numeric types (CPU only).
func RunEmbeddingL1Caching() {
	fmt.Println("=== Embedding Multi-Core L1 Caching — All Numerical Types ===")
	iterations := 3

	const (
		vocabSize    = 100
		embeddingDim = 32
		seqLen       = 16
	)
	const wCount = vocabSize * embeddingDim

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
		ws := poly.NewWeightStore(wCount)
		for i := range ws.Master {
			ws.Master[i] = 0.05
		}
		ws.Scale = cfg.scale
		if cfg.dtype != poly.DTypeFloat32 {
			ws.Morph(cfg.dtype)
		}
		l := poly.VolumetricLayer{
			Network:      poly.NewVolumetricNetwork(1, 1, 1, 1),
			Type:         poly.LayerEmbedding,
			VocabSize:    vocabSize,
			EmbeddingDim: embeddingDim,
			DType:        cfg.dtype,
			WeightStore:  ws,
		}
		// Input contains token IDs as float32 values (0..vocabSize-1)
		input := poly.NewTensor[float32](seqLen)
		for i := range input.Data {
			input.Data[i] = float32(i % vocabSize)
		}

		l.UseTiling = false
		var post0 *poly.Tensor[float32]
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, post0 = poly.EmbeddingForwardPolymorphic(&l, input)
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
			_, post1 = poly.EmbeddingForwardPolymorphic(&l, input)
		}
		tSingle := time.Since(start) / time.Duration(iterations)

		l.Network.EnableMultiCoreTiling = true
		l.SyncToCPU()
		var post2 *poly.Tensor[float32]
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, post2 = poly.EmbeddingForwardPolymorphic(&l, input)
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

// RunEmbeddingGPUForward checks GPU availability and reports that GPU kernels for Embedding
// are not yet implemented.
func RunEmbeddingGPUForward() {
	fmt.Println("=== Embedding GPU Forward — All Numerical Types ===")
	fmt.Println()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	defer net.DestroyWGPU()
	fmt.Println("GPU kernels for Embedding are not yet implemented.")
	fmt.Println("This test is a placeholder for future GPU support.")
}

// RunEmbeddingGPUBackward checks GPU availability and reports that GPU kernels for Embedding
// are not yet implemented.
func RunEmbeddingGPUBackward() {
	fmt.Println("=== Embedding GPU Backward — All Numerical Types ===")
	fmt.Println()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\nThis test requires a WebGPU-capable GPU.\n", err)
		return
	}
	defer net.DestroyWGPU()
	fmt.Println("GPU kernels for Embedding are not yet implemented.")
	fmt.Println("This test is a placeholder for future GPU support.")
}
