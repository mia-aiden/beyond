# Analysis

Scripts that turn the evaluation runs into the statistics reported in the thesis.
They read the experiment directory (`EXPERIMENTS_DIR`, see `training/paths.py`)
and write to an output directory; the runs themselves are never modified.

| Thesis | Script |
|---|---|
| 5.1 Synthetic data validation (embedding audit, t-SNE) | `rerun_synthetic_validation_mpnet.py` |
| 5.5 Cross-validation across inference backends, 5.6 narrative length | `backend_consistency.py` |
| 5.7 Model A/B against the four λ settings | `compare_model_abc.py` |
| 5.7 Three-seed emotion test and McNemar test | `emotion_significance.py` |
| 5.7 Paired bootstrap, λ=0.8 and λ=0.3 against Model B | `paired_bootstrap_narrative.py` |
| 5.8.1 Three-seed comparison with Model B | `summarize_model_b_seqjoint_3seed.py` |
| 5.8.2 Emotion-semantics controls and manipulation check | `summarize_format_semantics_multiseed.py` |
| 5.8.3 Model-by-prompt interaction | `summarize_prompt_cross_3seed.py` |
| Appendix, narrative-filter threshold distribution | `threshold_distribution.py` |

`compare_joint_lambdas.py` builds the λ ablation report from the validation
losses, and `threshold_sweep.py` is a threshold sensitivity check that is not
reported in the thesis.

## Stage1 Test Set: 984 comments

The Stage1 Test Set was drawn as 1,000 comments, and every model was evaluated
on all of them. Sixteen of these comments also belong to the 200-comment Manual
Gold Standard, so the thesis excludes them and reports Stage1 results on the
remaining 984. The Manual results are unchanged.

`filter_stage1_overlap.py` applies the exclusion without touching the original
runs. It mirrors the experiment directory with symlinks, checks that every Stage1
summary can be recomputed exactly from its per-sample files, and then rewrites
the per-sample and summary files of the 70 Stage1 runs without the 16 comments.
`run_stage1_analyses.sh` then runs every script in the table on one tree:

```bash
python analysis/filter_stage1_overlap.py --dst /root/autodl-fs/sft_experiments_984
analysis/run_stage1_analyses.sh /root/autodl-fs/sft_experiments_984 /root/autodl-fs/stage1_984_results
```

Running `run_stage1_analyses.sh` on the original experiment directory instead
reproduces the results on all 1,000 Stage1 comments.

The hierarchical bootstraps in 5.7 and 5.8 process Stage1 before Manual with one
random stream, so the Manual intervals (by at most 0.002) and p-values (by at most
0.04) also shift slightly when Stage1 shrinks, although the Manual data are the
same; the thesis reports the values of the 984 run. The paired bootstraps use
one random stream per test set and are unaffected.

## Environments

The analyses run in the `vllm-eval` environment (pandas, SciPy, scikit-learn,
sacreBLEU, Matplotlib). `rerun_synthetic_validation_mpnet.py` also needs PyTorch
and sentence-transformers, and the threshold scripts need pyarrow and spaCy with
`en_core_web_sm`; on the server they ran in the `llamafactory` environment.
