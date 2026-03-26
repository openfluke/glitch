package layer

import (
	"github.com/openfluke/loom/poly"
)

var embeddingSpec = TestSpec{
	Name: "Embedding",
	Layer: poly.PersistenceLayerSpec{
		Type:         "Embedding",
		InputHeight:  100, // Vocab size
		OutputHeight: 64,  // Embedding dim
	},
	InputShape: []int{8, 1}, // Batch of 8 indices
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(embeddingSpec, TestAll)
	})
}

func RunEmbeddingL1Caching() { RunGenericLayerSuite(embeddingSpec, TestForward) }
