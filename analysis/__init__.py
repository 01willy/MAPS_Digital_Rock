"""
Evaluation drivers behind the paper's tables and figures.

Modules (paper artifact each reproduces):
    benchmark_cubes         -- shared test-cube selection (all drivers)
    inference_tiled         -- tiled 2D/3D inference + parity reconstruction
    ksweep_eval             -- k-sweep stress evaluation (Fig. 6)
    morphology3d_eval       -- 3D morphology suite (Tables S3, S5-S7)
    aggregation_ablation    -- GT-free aggregation variants + oracle (Table 2)
    parity_matrix_eval      -- 3x3 Stage1 x Stage2 seed matrix (Table S2,
                               reported dphi = 0.00134 +/- 0.00033)
    lbm_multiseed_eval      -- test-only multi-seed LBM run (Table 3, S10)
    lbm_multiseed_aggregate -- aggregates the above into long/summary CSVs
    lbm_8domain_eval        -- 8-domain k=1 LBM trace campaign (Table S15)
    failure_regression      -- failure-map OLS regression (Figs. 12-13)
    compute_cost            -- end-to-end latency + FLOPs (Table 4)
    gpu_benchmark           -- per-GPU / DDP scaling suite (Tables S11-S12)

All drivers take --volume_path / --checkpoint / --out-style arguments; no
internal experiment-directory layout is assumed.
"""
