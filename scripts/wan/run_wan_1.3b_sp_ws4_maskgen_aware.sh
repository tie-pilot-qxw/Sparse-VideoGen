#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/Sparse-VideoGen

FLASHINFER_WORKSPACE_BASE=/tmp \
TRITON_CACHE_DIR=/tmp/triton-cache \
torchrun --nproc_per_node=4 scripts/wan/bench_wan_sp_all2all_attention.py \
  --input result/wan/t2v/sap_1.3b/attn_core_step30_layer5.pt \
  --warmup 1 \
  --iters 3 \
  --balance greedy \
  --density-log result/wan/t2v/sap_1.3b/Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10/1-0.jsonl \
  --cost-model-json result/wan/t2v/sap_1.3b/maskgen_aware_cost_step30_layer5.json \
  --rank-csv result/wan/t2v/sap_1.3b/sp_rank_times_step30_layer5_ws4_maskgen_aware.csv \
  --density-csv result/wan/t2v/sap_1.3b/sp_head_density_step30_layer5_ws4_maskgen_aware.csv \
  --q-chunk-density-csv result/wan/t2v/sap_1.3b/sp_q_chunk_density_step30_layer5_ws4_maskgen_aware.csv \
  --q-density-chunks 8
