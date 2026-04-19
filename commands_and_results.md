rank heads      seq           a2a_in  mask     qkv_ag  ag_wait  attention inv_perm a2a_out  total
-------------------------------------------------------------------------------------------------
   0  0-2          0-47189     1.56     4.86    0.00     0.00     22.93     0.10   20.12   49.58
   1  3-5      47190-94379     1.54     5.21    0.00     0.00     42.34     0.10    0.34   49.53
   2  6-8      94380-141569    1.27     4.85    0.00     0.00     13.49     0.10   29.54   49.24
   3  9-11    141570-188759    1.53     4.88    0.00     0.00     17.73     0.10   25.31   49.55

   rank heads      seq           a2a_in  mask     qkv_ag  ag_wait  attention inv_perm a2a_out  total
-------------------------------------------------------------------------------------------------
   0  0-2          0-47189     1.24     5.21    0.00     0.00     35.96     0.10    0.33   42.85
   1  3-5      47190-94379     1.41     4.99    0.00     0.00     22.26     0.10   14.24   43.00
   2  6-8      94380-141569    1.38     4.95    0.00     0.00     19.04     0.10   17.55   43.01
   3  9-11    141570-188759    1.41     4.97    0.00     0.00     18.86     0.10   17.70   43.04

   rank heads      seq           a2a_in  mask     qkv_ag  ag_wait  attention inv_perm a2a_out  total
-------------------------------------------------------------------------------------------------
   0  0-3          0-47189     1.59     2.66    0.00     0.00     27.04     0.03    3.20   34.53
   1  4-7      47190-94379     1.59     4.98    0.00     0.00     22.88     0.10    5.02   34.57
   2  8-11     94380-141569    1.58     7.12    0.00     0.00     24.26     0.14    1.69   34.78
   3 12-15    141570-188759    1.59     7.17    0.00     0.00     23.82     0.14    2.04   34.77


   SVG_WAN_ATTN_EXPORT_PATH=result/wan/t2v/sap_1.3b/attn_core_step20_240frames_layer21.pt \
  SVG_WAN_ATTN_EXPORT_MAX=1 \
  SVG_WAN_ATTN_EXPORT_LAYER=21 \
  SVG_WAN_ATTN_EXPORT_STEP=20 \
  SVG_WAN_ATTN_EXPORT_REQUIRE_CACHE=1 \
  python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "$(cat examples/1/prompt.txt)" \
    --height 480 \
    --width 832 \
    --seed 0 \
    --num_inference_steps 50 \
    --pattern SAP \
    --num_q_centroids 300 \
    --num_k_centroids 1000 \
    --top_p_kmeans 0.9 \
    --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 \
    --kmeans_iter_step 2 \
    --first_times_fp 0.2 \
    --first_layers_fp 0.03 \
    --output_file "result/wan/t2v/sap_1.3b/Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10/1-0.mp4" \
    --logging_file "result/wan/t2v/sap_1.3b/Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_240frames/1-0.jsonl" --num_frames 240



torchrun --nproc_per_node=4 scripts/wan/bench_wan_sp_all2all_attention.py     --input result/wan/t2v/sap_1.3b/attn_core_step20_480frames_layer21.pt     --density-log result/wan/t2v/sap_1.3b/Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_480frames/1-0.jsonl    --balance contiguous  --min-heads-per-rank 1     --iters 100 --warmup 50

torchrun --nproc_per_node=4 scripts/wan/bench_wan_sp_all2all_attention.py     --input result/wan/t2v/sap_1.3b/attn_core_step20_480frames_layer21.pt     --density-log result/wan/t2v/sap_1.3b/Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_480frames/1-0.jsonl    --balance greedy  --min-heads-per-rank 1     --iters 100 --warmup 50

torchrun --nproc_per_node=4 scripts/wan/bench_wan_sp_all2all_attention.py     --input result/wan/t2v/sap_1.3b/attn_core_step20_480frames_layer21.pt     --density-log result/wan/t2v/sap_1.3b/Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_480frames/1-0.jsonl    --balance greedy_unequal  --min-heads-per-rank 1     --iters 100 --warmup 50