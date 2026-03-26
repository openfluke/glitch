package layer

import (
	"github.com/openfluke/loom/poly"
)

var residualSpec = TestSpec{
	Name: "Residual",
	Layer: poly.PersistenceLayerSpec{
		Type:         "Residual",
		InputHeight:  128,
		OutputHeight: 128,
	},
	InputShape: []int{8, 128},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(residualSpec, TestAll)
	})
}

func RunResidualL1Caching() { RunGenericLayerSuite(residualSpec, TestForward) }
