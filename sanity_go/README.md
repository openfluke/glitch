# Loom v0.75.0 — High-Fidelity Go Sanity Check

This folder contains a standalone Go verification tool designed to validate the **v0.75.0 "Multi-Core Symphony"** milestone before merging into the main branch.

## 🎯 Verification Goals
1. **GPU Training Integrity**: Confirm that `poly.Train` with `TrainingModeGPUNormal` correctly updates weights in VRAM.
2. **Synchronization Reliability**: Confirm that `SyncWeightsFromGPU` successfully retrieves the trained state into the CPU `Master` FP32 store.
3. **Serialization Fidelity**: Confirm that `SerializeNetwork` (JSON) and `DeserializeNetwork` produce bit-perfect outputs compared to the pre-serialized state.
4. **Consistency**: Confirm that the model behaves identically after being reloaded from disk.

## 🚀 How to Run

### Initial Run (Training Mode)
This will create a new model, train it on the GPU, and save the result to `sanity_model.json`.
```bash
go run main.go
```

### Subsequent Runs (Reload Mode)
If `sanity_model.json` exists, the script will automatically bypass training and verify the existing model's output.
```bash
go run main.go
```

## 📊 Expected Output
A successful run should print:
```text
=== Loom v0.75.0 Go Sanity Check ===
[RELOAD MODE] Found existing model: sanity_model.json
[1/1] Verification (Reloaded Output)...
      Reloaded Output: [0.041971732 0.15299916]

=== SANITY CHECK COMPLETE: RELOAD VERIFIED ===
```

## 🛠️ Components Tested
- `poly.NewVolumetricNetwork`: Network construction.
- `poly.Train`: GPU-accelerated training pipeline.
- `poly.SyncWeightsFromGPU`: Unified memory synchronization.
- `poly.ForwardPolymorphic`: CPU-side verification pass.
- `poly.SerializeNetwork`: JSON weight serialization.
- `poly.DeserializeNetwork`: JSON bit-perfect reconstruction.

---
**Verified by Antigravity AI for Loom v0.75.0 Stable Release.**
