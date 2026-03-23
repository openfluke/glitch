# Systolic 3D Target Propagation Demo

This demo illustrates a running deployment of `poly.SystolicForward` and `poly.SystolicApplyTargetProp` to model "Predictive Coding" / continuous-time biological neural learning without a traditional sequential backward pass.

## What is happening here?
1. The `VolumetricNetwork` is constructed physically as a 3D grid with `UseTiling = true`. This forces the engine to dispatch parallel Go routines, pinning sub-cubes of the neural grid to specific CPU caches.
2. The network learns in a **streaming** manner. Instead of gathering a large static "batch" and halting inference to execute `backwards()`, we push data through the network 1 spatial hop per clock tick (`SystolicForward`).
3. We apply `SystolicApplyTargetProp` at the output. The mesh relies on `target_prop.go`'s Link Budgeting (Cosine Similarity) and Local Error Gaps to naturally adjust its internal weights to match the biological prediction goal. 

## To run:
```bash
go run main.go
```
You should see the network's `Running Error Avg` decrease rapidly over a few thousand continuous spatial ticks as it learns to perfectly map the incoming stream to the non-linear target metric autonomously.
