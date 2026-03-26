// Package layer provides layer-level tests for glitch testing mode.
package layer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

// ── Shared Helper Types ───────────────────────────────────────────────────────

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

// ── Shared Utilities ─────────────────────────────────────────────────────────

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

func genInput(shape []int) *poly.Tensor[float32] {
	t := poly.NewTensor[float32](shape...)
	for i := range t.Data {
		t.Data[i] = float32(i%13)*0.1 - 0.6
	}
	return t
}

// ── Standardized Results ─────────────────────────────────────────────────────

type CachingResult struct {
	TNormal, TSingle, TMulti time.Duration
	TileSize                 int
	Parity01, Parity02       bool
}

type TrainingResult struct {
	Mode      poly.TrainingMode
	LossInit  float32
	LossFinal float32
	Dur       time.Duration
	TrainOK   bool
	SaveOK    bool
	ByteCount int
	RamBytes  int64
	Err       error
}

type ParityResult struct {
	TCPUMC, TGPUNorm, TGPUSC, TGPUMC time.Duration
	DiffGN, DiffGSC, DiffGMC         float64
	ParityGN, ParityGSC, ParityGMC   bool
	TileSize, SCTile, MCTile         int
}

// ── Standardized Table Printing ──────────────────────────────────────────────

func PrintCachingHeader() {
	fmt.Println()
	fmt.Printf("| %-10s | %-5s | %-14s | %-14s | %-14s | %-7s | %-7s | %-8s | %-8s |\n",
		"DType", "Tile", "Normal", "Single-Core", "Multi-Core", "1C-Spd", "MC-Spd", "1C-Par", "MC-Par")
	fmt.Println("|------------|-------|----------------|----------------|----------------|---------|---------|----------|----------|")
}

func PrintCachingRow(cfg typeConfig, r CachingResult) {
	fmt.Printf("| %-10s | %-5d | %-14v | %-14v | %-14v | %-7.2fx | %-7.2fx | %-8s | %-8s |\n",
		cfg.name, r.TileSize, r.TNormal, r.TSingle, r.TMulti,
		float64(r.TNormal)/float64(r.TSingle),
		float64(r.TNormal)/float64(r.TMulti),
		parityMark(r.Parity01), parityMark(r.Parity02))
}

func PrintTrainingHeader() {
	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "Train↑", "Save/Reload", "File", "RAM")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")
}

func PrintTrainingRow(cfg typeConfig, r TrainingResult) {
	if r.Err != nil {
		fmt.Printf("| %-10s | %-13s | ERR        | ERR        | %-8v | ERR     | %s\n", cfg.name, r.Mode.String(), r.Dur.Round(time.Millisecond), r.Err.Error())
		return
	}
	fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8s | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
		cfg.name, r.Mode.String(),
		r.LossInit, r.LossFinal,
		r.Dur.Round(time.Millisecond),
		parityMark(r.TrainOK), parityMark(r.SaveOK),
		float64(r.ByteCount)/1024.0, float64(r.RamBytes)/1024.0)
}

func PrintParityHeader() {
	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s |\n",
		"DType", "Tile", "CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd", "Diff-GN", "Diff-SC", "Diff-MC", "GN-Par", "SC-Par", "MC-Par")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|--------|--------|--------|----------|----------|----------|--------|--------|--------|")
}

func PrintParityRow(cfg typeConfig, r ParityResult) {
	fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-6.1fx | %-6.1fx | %-6.1fx | %-8.2e | %-8.2e | %-8.2e | %-6s | %-6s | %-6s |\n",
		cfg.name, r.TileSize, r.TCPUMC, r.TGPUNorm, r.TGPUSC, r.TGPUMC,
		float64(r.TCPUMC)/float64(r.TGPUNorm),
		float64(r.TCPUMC)/float64(r.TGPUSC),
		float64(r.TCPUMC)/float64(r.TGPUMC),
		r.DiffGN, r.DiffGSC, r.DiffGMC,
		parityMark(r.ParityGN), parityMark(r.ParityGSC), parityMark(r.ParityGMC))
}

// ── Global Stats ─────────────────────────────────────────────────────────────

type GlobalStats struct {
	TotalTests  int
	PassedTests int
}

func (s *GlobalStats) Add(ok bool) {
	s.TotalTests++
	if ok {
		s.PassedTests++
	}
}

func (s *GlobalStats) Report() {
	if s.TotalTests == 0 {
		return
	}
	pct := float64(s.PassedTests) / float64(s.TotalTests) * 100
	fmt.Printf("\n=== GLOBAL PASS RATE: %.1f%% (%d/%d) ===\n", pct, s.PassedTests, s.TotalTests)
}

var stats = &GlobalStats{}

// ── Generic Test Runner ──────────────────────────────────────────────────────

type LayerTask func() bool

var registry []LayerTask

func RegisterTask(t LayerTask) {
	registry = append(registry, t)
}

func RunAllLayers() {
	fmt.Println("🚀 Running All Layer Tests...")
	start := time.Now()
	for _, task := range registry {
		ok := task()
		stats.Add(ok)
	}
	stats.Report()
	fmt.Printf("Total Time: %v\n", time.Since(start).Round(time.Millisecond))
}
