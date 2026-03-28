/**
 * @openfluke/welvet — Node.js AI Showcase (Loom v0.75.0)
 * ----------------------------------------------------
 * Comprehensive command-line example for backend inference,
 * training, and evolutionary AI using the Loom engine.
 *
 * Usage: npm run start
 */

import welvet from '@openfluke/welvet';

// Beautiful CLI Logging
const log = (msg: string, colorCode: string = '32') => {
    console.log(`\x1b[90m[${new Date().toLocaleTimeString()}]\x1b[0m \x1b[${colorCode}m${msg}\x1b[0m`);
};

async function runNodeDemo() {
    log("=== WELVET NODE.JS ENGINE INITIALIZATION ===", "36");
    
    try {
        // 1. Loader & Discovery
        await welvet.init();
        const parity = welvet.getInternalParity ? welvet.getInternalParity() : [];
        log(`Runtime Loaded. ${parity.length} internal symbols acquired.`, "32");

        // 2. High-Fidelity Network Creation
        log("\n--- PHASE 1: Architecting Polymorphic Networks ---", "36");
        const net = welvet.createNetwork({
            depth: 1, rows: 1, cols: 1, layers_per_cell: 2,
            layers: [
                { z: 0, y: 0, x: 0, l: 0, type: "Dense", input_height: 8, output_height: 32, activation: "SiLU", dtype: "F32" },
                { z: 0, y: 0, x: 0, l: 1, type: "Dense", input_height: 32, output_height: 1, activation: "Sigmoid", dtype: "F32" }
            ]
        });
        log(`Network Alpha Build ID: ${net._id}`, "32");

        // 3. Functional Inference Pass
        const input = new Float32Array(8).fill(0.1);
        const output = net.sequentialForward(input);
        log(`Forward pass result: ${output[0].toFixed(6)}`, "32");

        // 4. Parity IO: Blueprints & SafeTensors
        log("\n--- PHASE 2: IO & Serialization (SafeTensors) ---", "36");
        const blueprint = net.extractNetworkBlueprint("demo_model");
        log(`Blueprint extracted (${blueprint.length} chars).`, "32");

        // Demonstration of save/load (Note: Node-only features)
        const weights = welvet.saveNetwork(net);
        log(`Network serialized to SafeTensors buffer: ${weights.length} bytes.`, "32");

        // 5. Training Loop (Manual Parity Verification)
        log("\n--- PHASE 3: Training (Backprop Performance) ---", "36");
        const batches = [
            { input: [0, 0, 0, 0, 0, 0, 0, 0], target: [0] },
            { input: [1, 1, 1, 1, 1, 1, 1, 1], target: [1] }
        ];
        
        log("Starting supervised training (10 epochs)...", "90");
        const result = welvet.trainNetwork(net, batches, 10, 0.05);
        log(`Training Result: Final Loss = ${result.final_loss.toFixed(6)}`, "32");

        // 6. Genetic Crossover (NEAT Lab)
        log("\n--- PHASE 4: Genetic Evolution ---", "36");
        const pop = welvet.createNEATPopulation(net, 10);
        log(`Created NEAT Population of size ${pop.size}.`, "32");

        log("Evaluating generation 1...", "90");
        pop.evolveWithFitnesses(Array.from(new Float64Array(10).fill(1.0)));
        log(`${pop.summary(1)}`, "32");

        // 7. Transformer Module (LLM)
        log("\n--- PHASE 5: Transformer Logic (LLM Context) ---", "36");
        const tnet = welvet.createNetwork({
            depth: 1, rows: 1, cols: 1, layers_per_cell: 1,
            layers: [{ type: "MHA", d_model: 64, num_heads: 4, seq_length: 128 }]
        });
        
        const embeds = new Float32Array(100 * 64).fill(0.01);
        const head = new Float32Array(64 * 100).fill(0.01);
        const norm = new Float32Array(64).fill(1.0);
        
        const model = welvet.createTransformer(tnet, embeds, head, norm);
        log("Transformer Context initialized (d_model=64, n_heads=4).", "32");

        const logits = model.forwardFull([1, 2, 3, 4]);
        log(`LLM Prefill pass token results: ${logits.length} float values.`, "32");

        log("\n=== NODE.JS DEMO COMPLETE (100% PARITY) ===", "36");
        // @ts-ignore
        process.exit(0);

    } catch (e) {
        // @ts-ignore
        console.error(`\x1b[31mCRITICAL FAILURE: ${(e as Error).message}\x1b[0m`);
        // @ts-ignore
        process.exit(1);
    }
}

runNodeDemo();
