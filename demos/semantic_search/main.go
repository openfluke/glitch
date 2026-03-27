package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

type EmbeddingEngine struct {
	Network *poly.VolumetricNetwork
	Vocab   map[string]int
	Dim     int
}

func NewEmbeddingEngine(modelPath, vocabPath string) (*EmbeddingEngine, error) {
	fmt.Printf("⏳ Loading %s and %s into Poly Bedrock...\n", modelPath, vocabPath)

	// Load Vocab
	vocabFile, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer vocabFile.Close()

	vocab := make(map[string]int)
	scanner := bufio.NewScanner(vocabFile)
	idx := 0
	for scanner.Scan() {
		w := strings.ToLower(scanner.Text())
		vocab[w] = idx
		idx++
	}

	// Load Network
	net, err := poly.LoadUniversal(modelPath)
	if err != nil {
		return nil, err
	}

	dim := net.Layers[0].OutputHeight
	if dim == 0 {
		dim = net.Layers[0].EmbeddingDim
	}

	return &EmbeddingEngine{
		Network: net,
		Vocab:   vocab,
		Dim:     dim,
	}, nil
}

// GetVector uses Loom's native polymorphic forward pass to extract meanings.
func (e *EmbeddingEngine) GetVector(text string) []float32 {
	words := strings.Fields(strings.ToLower(text))
	var tokenIDs []float32
	for _, w := range words {
		if id, ok := e.Vocab[w]; ok {
			tokenIDs = append(tokenIDs, float32(id))
		}
	}

	if len(tokenIDs) == 0 {
		return make([]float32, e.Dim)
	}

	input := poly.NewTensorFromSlice(tokenIDs, len(tokenIDs))
	
	// Native Poly call: uses current DType (FP32, INT8, etc.)
	_, postAct := poly.EmbeddingForwardPolymorphic(e.Network.GetLayer(0,0,0,0), input)
	
	// Mean Pooling
	res := make([]float32, e.Dim)
	for i := 0; i < len(tokenIDs); i++ {
		for d := 0; d < e.Dim; d++ {
			res[d] += float32(postAct.Data[i*e.Dim+d])
		}
	}
	for d := 0; d < e.Dim; d++ {
		res[d] /= float32(len(tokenIDs))
	}
	return res
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func main() {
	engine, err := NewEmbeddingEngine("glove_300d.safetensors", "vocab.txt")
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		fmt.Println("Did you run converter/main.go first?")
		return
	}

	fmt.Println("✅ Corpus Online. Ready for semantic correlation.")
	
	// Pre-calculate FP32 embeddings for the KnowledgeBase
	fmt.Printf("⏳ Pre-calculating %d KnowledgeBase embeddings in FP32...\n", len(KnowledgeBase))
	kbEmbeds := make([][]float32, len(KnowledgeBase))
	for i, text := range KnowledgeBase {
		kbEmbeds[i] = engine.GetVector(text)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\n🔍 Query (or 'exit'): ")
		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)
		if query == "exit" || query == "quit" {
			break
		}

		// Showcase Polymorphism
		types := []poly.DType{
			poly.DTypeFloat32,
			poly.DTypeInt8,
			poly.DTypeBinary,
		}

		for _, dt := range types {
			start := time.Now()
			
			// Morph the model
			fmt.Printf("🧬 Mode: %v... ", dt)
			poly.MorphLayer(engine.Network.GetLayer(0,0,0,0), dt)

			qVec := engine.GetVector(query)
			
			type Result struct {
				Text string
				Score float32
			}
			var results []Result

			for i, kbVec := range kbEmbeds {
				// We use the local cosine similarity for correlation
				// Note: kbEmbeds were pre-calc in FP32, qVec is current DType.
				score := cosineSimilarity(qVec, kbVec)
				results = append(results, Result{KnowledgeBase[i], score})
			}

			sort.Slice(results, func(i, j int) bool {
				return results[i].Score > results[j].Score
			})

			elapsed := time.Since(start)
			fmt.Printf("(%v)\n", elapsed)
			for i := 0; i < 2; i++ {
				fmt.Printf("  %.4f: %s\n", results[i].Score, results[i].Text)
			}
		}
	}
}
