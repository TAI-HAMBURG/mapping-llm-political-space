## Description
This is the repository for the paper "Mapping LLMs’ Political Space through Multi-Axis Representation Steering" (Zaberer and Lauscher, to appear at EACL 2027). It contains all the code to reproduce the results from the paper as well as intermediate files, data and steering vectors. 

# Reproducing experiments
1. Install requirements: 
- Python, PyTorch, transformers
- xrfm in an old version as specified in https://github.com/dmbeaglehole/neural_controllers

2. Set the directories in directory_config and your HF token. Make sure you have access to Llama models via HF
3. If you want to re-run any LLM judge setup, put OpenAI API key there as well

## §3: Limits of Single-Dimension Representations
```
python RepresentationPoliticalLLM/mapping_pol_space.py --model_type [llama/llama-big/mistral-medium] --task congress_groups
```

## §4: Multi-Axis steering
```
python RepresentationPoliticalLLM/mapping_pol_space.py --model_type [llama/llama-big/mistral-medium] --task TASK
```
- Generating steered values (e.g. Table 2): `--task generate_8values_statements --prompt values` (creates file in `outputs/8values/generations/`, e.g. `llama_generations_values.json`)
- Generating steered statements: `--task generate_8values_statements --prompt issues` (creates file in `outputs/8values/generations/`, e.g. `llama_generations_issues.json`)
- Judging steered statements: `--task judge_all_generations` (creates/adds to file in `outputs/8values/`, e.g. `llama_steering_eval_LLMjudge_gpt5.json`)
- LLM judge validation via human annotations: `--task annotator_scores_and_agreement` will first print agreement between humans, then between human and LLM judge.

## §5: Scaling of manifesto values
```
python RepresentationPoliticalLLM/mapping_pol_space.py --model_type [llama/llama-big/mistral-medium] --task TASK
```
- Party value scaling: `--task projection --baseline axes_combined`
- Robustness checks: 
    - For other baselines set baseline arg to `nayak` (political_ideologies), `politune` and `hs116` (Voteview)
    - For more variables but fewer parties set `countries = ["Germany"]` in `projection` 
    - For prompt robustness check set `--task robustness_prompt`

## §6: Relationships Between Steering Axes
```
python RepresentationPoliticalLLM/mapping_pol_space.py --model_type [llama/llama-big/mistral-medium] --task TASK
```
- Generating statements for all combinations of two steered axes: `--task steer_multiaxis` (creates file in `outputs/8values/generations/`, e.g. `llama-big_generations_multiaxis.json`)
- Judging all combinations of two steered axes: `--task judge_combinations` (creates/adds to file in `outputs/8values/`, e.g. `llama_multiaxis_eval_LLMjudge_gpt5.json`)
- Get ratio of desired steering of all setups in that file: `--task measure_entanglement`

(some analysis in `--task evaluate_multiaxis_steering` and `plot_pol_space.py` as well)

### Mitigations
Same generation/judging tasks as in §6, but for orthogonalization set argument `--orth subspace` or `--orth löwdin`, for Contrastive+ set `--all8 True`

## §7: Cube visualizations
`--task get_axis_predictions --category CATEGORY`, e.g. `party_de` (saves predictions to `outputs/8values/pol_maps/`), then plot in `plot_pol_space.py`

# Credit
All code not specified below is (c) 2026 Urs Zaberer. If you used any of our code in your project, please cite
```
TBA
```


The file repeng_scripts/repeng/control_rfm.py is adapted from https://github.com/dmbeaglehole/neural_controllers, (c) 2025 Daniel Beaglehole, MIT license. If you use it, please cite [their paper](https://www.science.org/doi/10.1126/science.aea6792).

All other code in the repeng_scripts directory is (c) 2024 Theia Vogel, MIT license, from https://github.com/vgel/repeng/.
If used, please cite 

```
@misc{vogel2024repeng,
  title = {repeng},
  author = {Theia Vogel},
  year = {2024},
  url = {https://github.com/vgel/repeng/}
}
```
and [Zou et al. 2023](https://github.com/andyzoujm/representation-engineering?tab=readme-ov-file#citation).

RepresentationPoliticalLLM/8values.json is from https://github.com/8values/8values.github.io, copyright (c) 2020 8values, MIT license.

## Contact 
urs.zaberer[at]uni-hamburg.de
 



