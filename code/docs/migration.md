# Migration map

All Python commands now run through `uv run jlens`; see [commands](commands.md).
Direct module execution is also available as `uv run python -m jlens.MODULE`.
The old script paths are retired.

| Previous module | Current module |
| --- | --- |
| `paper_stats.py` | `jlens/paper_statistics.py` |
| `build_paper_figure_data.py` | `jlens/figure_data.py` |
| `figure_fact_ranks.py` | `jlens/plot_fact_ranks.py` |
| `analysis/relsplit.py` | `jlens/splits.py` |
| `analysis/lens_vs_faith.py` | `jlens/measure_jacobian_lens.py` |
| `analysis/lre_faithfulness.py` | `jlens/measure_faithfulness.py` |
| `analysis/lre_subject_ranks.py` | `jlens/measure_subject_ranks.py` |
| `analysis/logit_lens_baseline.py` | `jlens/measure_logit_lens.py` |
| `analysis/object_prior.py` | `jlens/measure_answer_prior.py` |
| `analysis/prominence2.py` | `jlens/measure_prominence.py` |
| `analysis/prominence_stats.py` | `jlens/prominence_statistics.py` |
| `analysis/lens_probe_names.py` | `jlens/probe_names.py` |
| `analysis/expansion_stats.py` | `jlens/relation_statistics.py` |
| `analysis/subject_level.py` | `jlens/subject_statistics.py` |
| `analysis/outlier_ranking.py` | `jlens/rank_outliers.py` |
| `analysis/make_figure_rank_rank.py` | `jlens/plot_relation_ranks.py` |
| `analysis/panels_subject_level.py` | `jlens/plot_subject_outcomes.py` |
| `analysis/panels_subject_scatter.py` | `jlens/plot_subject_ranks.py` |
| `repro/make_table.py` | `jlens/table_relations.py` |
| `repro/make_prominence_table.py` | `jlens/table_prominence.py` |
| `repro/make_paper_figure.py` | `jlens/plot_relation_agreement.py` |

| Previous data/result path | Current path |
| --- | --- |
| `data/lre_full/{bias,factual}/`, `data/lre_all/` | `data/relations/` |
| `e2/uniform/` | `results/main/` |
| `e2/distinct/` | `results/shared_split/` |
| `e2/unfilt/` | `results/unfiltered/` |
| `lrefaithT_*.json` | `faithfulness_*.json` |
| `lresubjrank_*.json` | `subject_ranks_*.json` |
| `lens_vs_faith_full.json` | `jacobian_lens.json` |
| `lens_vs_faith_years_ts.json` | `jacobian_lens_years.json` |
| `lens_aligned.json`, `lens_unfiltered.json` | `jacobian_lens.json` in the corresponding protocol directory |
| `object_prior_full.json` | `answer_prior.json` |
| `prominence2.json` | `prominence.json` |
| `prominence_stats.json` | `prominence_statistics.json` |
| `logit_lens_baseline.json` | `logit_lens.json` |

The root-level `run_gpu_reproduction.sh` and five cluster launcher scripts under
`analysis/` are replaced by `jlens rerun`. This runner respects
`CUDA_VISIBLE_DEVICES` and uses the active uv environment; GPU selection is no
longer tied to a particular cluster. `requirements-gpu.txt` is replaced by
`pyproject.toml` and `uv.lock`.
