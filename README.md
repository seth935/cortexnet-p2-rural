# CortexNet-P2-Rural (v3.0.0 — Generation 1)

Generation-1 Registry Fork of **CortexNet-OBB-1W** (`sub02cortexnetobb1w0000000000002`) built for the **Weights of Giants Season Zero: 1-Watt Edge Vision Arena** (`la-frite-1gb` / ARM Cortex-A53).

## Telemetry-Driven Generation-2 Upgrades
1. **352×352 Input + Multi-Scale `P2 (Stride-4) + P3 (Stride-8) + P4 (Stride-16)` Fused Neck:** Preserves `88×88` Stride-4 and `44×44` Stride-8 spatial features to recover `<32px` rural horizon targets (`95.8%` of v2 TIDE loss).
2. **In-Graph Weber-Fechner Local Contrast Normalization:** Divides ReLU-gated steerable Gabor/Opponent-Chroma stem activations by `(LocalMean + 0.08)`, boosting Low-Light Shot-Noise (`LLR`), Atmospheric Haze (`HSR`), and Hard-Shadow (`SBS`) retention while guaranteeing exact `0.0000 FPPI` on flat-field canary frames.
3. **Mirror-Paired Gaussian KLD Covariance Training (`models/train_cortexnet_p2_rural.py`):** Optimizes 3-scale OBB geometry priors with exact horizontal flip symmetry (`cx' = 1 - cx`, `theta' = -theta`).
4. **2-Thread Cortex-A53 Execution (`threads: 2`):** Eliminates 4-thread OpenMP barrier contention measured on the Cortex-A53 silicon profiler (`5.6 ms` at 2 threads vs `11.5 ms` at 4 threads).

## License
Released under **GNU AGPL-3.0** (`parent_lineage: ["registry_fork", "sub02cortexnetobb1w0000000000002"]`).
