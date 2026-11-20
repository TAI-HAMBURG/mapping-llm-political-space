## Llama-70B
Prompt: "Generate a statement by the German party {}."
AfD: {'econ': 0.017774173617362977, 'govt': 0.0003783881664276123, 'scty': 0.003494676947593689, 'dipl': 0.0027465413014094034}
CDU: {'econ': 0.036950488885243736, 'govt': 0.0056517769893010454, 'scty': 0.0017276068528493246, 'dipl': 0.018111673990885417}
FDP: {'econ': 0.03748884399731954, 'govt': 0.012447421749432882, 'scty': 0.006517452001571655, 'dipl': 0.02043415407339732}
The Greens: {'econ': 0.053087926904360455, 'govt': 0.01566182772318522, 'scty': 0.010289743542671204, 'dipl': 0.03391708532969157}
BSW (past knowledge cutoff): {'econ': 0.019580731789271038, 'govt': 0.0054017513990402225, 'scty': 0.0005693733692169189, 'dipl': 0.01751015881697337}

## Llama-8B
[{'econ': 0.0381130494854667, 'govt': -0.019841448827223343, 'scty': 0.007115117528221824, 'dipl': -0.009088638153943148}, 
{'econ': 0.048313663764433426, 'govt': -0.011875808238983154, 'scty': 0.011473761363462969, 'dipl': 0.008825402368198742}, 
{'econ': 0.0453508198261261, 'govt': -0.013851344585418701, 'scty': 0.010103843428871849, 'dipl': 0.0046974122524261475}, 
{'econ': 0.05234864083203403, 'govt': -0.004143674265254627, 'scty': 0.015017344193025068, 'dipl': 0.012385316870429298}, 
{'econ': 0.05655464800921353, 'govt': -0.005751436406915838, 'scty': 0.016622321172194046, 'dipl': 0.014288883317600597}, 
{'econ': 0.06590999527411028, 'govt': 0.0016075047579678621, 'scty': 0.01565498113632202, 'dipl': 0.024208475242961536}, 
{'econ': 0.04826933958313682, 'govt': -0.0074027343229814005, 'scty': 0.014643636616793547, 'dipl': 0.011605474081906405}, 
{'econ': 0.0695267536423423, 'govt': 0.002462525259364735, 'scty': 0.015918463468551636, 'dipl': 0.020263235677372326}]


{"model_type": "llama", "strength": 0.5, "all8": false, "results": {
    "AfD": {"econ": 0.0381130494854667, "govt": -0.019841448827223343, "scty": 0.007115117528221824, "dipl": -0.009088638153943148}, 
    "CDU": {"econ": 0.048313663764433426, "govt": -0.011875808238983154, "scty": 0.011473761363462969, "dipl": 0.008825402368198742}, 
    "CSU": {"econ": 0.0453508198261261, "govt": -0.013851344585418701, "scty": 0.010103843428871849, "dipl": 0.0046974122524261475}, 
    "FDP": {"econ": 0.05234864083203403, "govt": -0.004143674265254627, "scty": 0.015017344193025068, "dipl": 0.012385316870429298}, 
    "SPD": {"econ": 0.05655464800921353, "govt": -0.005751436406915838, "scty": 0.016622321172194046, "dipl": 0.014288883317600597}, 
    "Bündnis '90/Die Grünen": {"econ": 0.06590999527411028, "govt": 0.0016075047579678621, "scty": 0.01565498113632202, "dipl": 0.024208475242961536}, 
    "BSW": {"econ": 0.045238294384696266, "govt": -0.006666730750690807, "scty": 0.016128456050699406, "dipl": 0.009557905522259798}, 
    "Die Linke": {"econ": 0.0695267536423423, "govt": 0.002462525259364735, "scty": 0.015918463468551636, "dipl": 0.020263235677372326}}}

{"baseline": true, "model_type": "llama-big", "strength": 1, "results": {"AfD": {"right": 0.01570175588130951}, "CDU": {"right": 0.02231767773628235}, "CSU": {"right": 0.02093265950679779}, "FDP": {"right": 0.022965365648269655}, "SPD": {"right": 0.027796561519304912}, "B\u00fcndnis '90/Die Gr\u00fcnen": {"right": 0.03063608705997467}, "BSW": {"right": 0.02314061721165975}, "Die Linke": {"right": 0.02892240583896637}}}

{"baseline": true, "model_type": "llama-big", "strength": 1, "results": {"AfD": {"axes_combined": 0.001622837781906128}, "CDU": {"axes_combined": 0.005780808130900065}, "CSU": {"axes_combined": 0.004389822483062744}, "FDP": {"axes_combined": 0.010535163680712382}, "SPD": {"axes_combined": 0.01087781588236491}, "B\u00fcndnis '90/Die Gr\u00fcnen": {"axes_combined": 0.017189092437426248}, "BSW": {"axes_combined": 0.005703008174896241}, "Die Linke": {"axes_combined": 0.015673877795537312}}}

## Correlations: Llama-70B
### HS 116 (no BSW)
right, rile: r = -0.87, pvalue 0.02
right, planeco: r = 0.66, pvalue 0.15
right, markeco: r = -0.82, pvalue 0.05
right, welfare: r = 0.86, pvalue 0.03
right, intpeace: r = 0.59, pvalue 0.21
right, Military-pro: r = -0.14, pvalue 0.8
right, Internationalism-pro: r = 0.79, pvalue 0.06
right, EU-pro: r = 0.54, pvalue 0.27
right, Freedoms: r = -0.69, pvalue 0.13
right, Environment: r = 0.51, pvalue 0.3
right, Equality: r = 0.88, pvalue 0.02
### combined (no BSW)
axes_combined, rile: r = -0.83, pvalue 0.04
axes_combined, planeco: r = 0.67, pvalue 0.14
axes_combined, markeco: r = -0.69, pvalue 0.13
axes_combined, welfare: r = 0.81, pvalue 0.05
axes_combined, intpeace: r = 0.63, pvalue 0.18
axes_combined, per104: r = -0.32, pvalue 0.53
axes_combined, per107: r = 0.58, pvalue 0.22
axes_combined, per108: r = 0.32, pvalue 0.53
axes_combined, per201: r = -0.52, pvalue 0.29
axes_combined, per501: r = 0.6, pvalue 0.21
axes_combined, per503: r = 0.86, pvalue 0.03
### axes (no BSW)
econ, rile: r = -0.92, pvalue 0.01
econ, planeco: r = 0.79, pvalue 0.06
econ, markeco: r = -0.8, pvalue 0.06
econ, welfare: r = 0.89, pvalue 0.02
econ, intpeace: r = 0.71, pvalue 0.11
econ, per104: r = -0.21, pvalue 0.69
econ, per107: r = 0.67, pvalue 0.15
econ, per108: r = 0.4, pvalue 0.44
econ, per201: r = -0.69, pvalue 0.13
econ, per501: r = 0.46, pvalue 0.36
econ, per503: r = 0.92, pvalue 0.01
govt, rile: r = -0.78, pvalue 0.07
govt, planeco: r = 0.66, pvalue 0.16
govt, markeco: r = -0.58, pvalue 0.23
govt, welfare: r = 0.74, pvalue 0.1
govt, intpeace: r = 0.64, pvalue 0.17
govt, per104: r = -0.43, pvalue 0.4
govt, per107: r = 0.42, pvalue 0.4
govt, per108: r = 0.16, pvalue 0.76
govt, per201: r = -0.38, pvalue 0.45
govt, per501: r = 0.62, pvalue 0.18
govt, per503: r = 0.81, pvalue 0.05
scty, rile: r = -0.47, pvalue 0.35
scty, planeco: r = 0.18, pvalue 0.74
scty, markeco: r = -0.41, pvalue 0.42
scty, welfare: r = 0.49, pvalue 0.32
scty, intpeace: r = 0.15, pvalue 0.78
scty, per104: r = -0.09, pvalue 0.86
scty, per107: r = 0.63, pvalue 0.18
scty, per108: r = 0.55, pvalue 0.25
scty, per201: r = -0.4, pvalue 0.43
scty, per501: r = 0.53, pvalue 0.28
scty, per503: r = 0.51, pvalue 0.3
dipl, rile: r = -0.78, pvalue 0.07
dipl, planeco: r = 0.59, pvalue 0.21
dipl, markeco: r = -0.71, pvalue 0.11
dipl, welfare: r = 0.75, pvalue 0.09
dipl, intpeace: r = 0.54, pvalue 0.27
dipl, per104: r = -0.17, pvalue 0.75
dipl, per107: r = 0.63, pvalue 0.18
dipl, per108: r = 0.41, pvalue 0.41
dipl, per201: r = -0.58, pvalue 0.23
dipl, per501: r = 0.64, pvalue 0.17
### Win rates:
34 parties, 23 variables (<20% zeros): {'econ': 8, 'govt': 2, 'scty': 8, 'dipl': 4, 'axes_combined': 1}
OLD AND DIFFERENT: German parties, 58 variables: {'econ': 16, 'govt': 5, 'scty': 31, 'dipl': 6, 'axes_combined': 0}
CORRECT: {'econ': 0.3448275862068966, 'govt': 0.06896551724137931, 'scty': 0.3448275862068966, 'dipl': 0.2413793103448276, 'axes_combined': 0.0}

## Desired steering ratios
[]
Average ratio desired steering:  0.346
['orthogonal: löwdin']
Average ratio desired steering:  0.32
['orthogonal: False', 'all8: True']
Average ratio desired steering:  0.396
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge']
Average ratio desired steering:  0.347
['orthogonal: subspace', 'all8: False', 'prompt: construct_prompt_LLMjudge', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.406
['orthogonal: subspace', 'all8: False', 'prompt: construct_prompt_LLMjudge_old', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.395
[] ['orthogonal: löwdin'] 0.38
[] ['orthogonal: False', 'all8: True'] 0.2
[] ['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge'] 0.95
[] ['orthogonal: subspace', 'all8: False', 'prompt: construct_prompt_LLMjudge', 'only_judge_relevant_statements: False'] 0.25
### Robustness check over 5 prompts:
econ: Average winrate 0.3448275862068965, sd 0.06071316159192762
govt: Average winrate 0.07931034482758621, sd 0.048275862068965524
scty: Average winrate 0.3172413793103448, sd 0.09910419215999941
dipl: Average winrate 0.2482758620689655, sd 0.11975003692541612
axes_combined: Average winrate 0.010344827586206896, sd 0.008446516354424751
{1: 21, 2: 26, 3: 10, 4: 1, 5: 0}
{0: {'econ': 0.3448275862068966, 'govt': 0.06896551724137931, 'scty': 0.3448275862068966, 'dipl': 0.2413793103448276, 'axes_combined': 0.0}, 
1: {'econ': 0.3275862068965517, 'govt': 0.05172413793103448, 'scty': 0.3620689655172414, 'dipl': 0.2413793103448276, 'axes_combined': 0.017241379310344827}, 
2: {'econ': 0.3448275862068966, 'govt': 0.06896551724137931, 'scty': 0.20689655172413793, 'dipl': 0.3620689655172414, 'axes_combined': 0.017241379310344827}, 
3: {'econ': 0.4482758620689655, 'govt': 0.034482758620689655, 'scty': 0.46551724137931033, 'dipl': 0.034482758620689655, 'axes_combined': 0.017241379310344827}, 
4: {'econ': 0.25862068965517243, 'govt': 0.1724137931034483, 'scty': 0.20689655172413793, 'dipl': 0.3620689655172414, 'axes_combined': 0.0}}
### Other baselines OLD:
{'econ': 0.1724137931034483, 'govt': 0.06896551724137931, 'scty': 0.3448275862068966, 'dipl': 0.10344827586206896, 'hs116': 0.3103448275862069}
{'econ': 0.29310344827586204, 'govt': 0.05172413793103448, 'scty': 0.08620689655172414, 'dipl': 0.10344827586206896, 'nayak': 0.46551724137931033}
politune: BL .43, Econ .31, Civil .05, Soc .10, Dipl .10\\

### Scaling using subspace OLD:
{'econ': 0.3275862068965517, 'govt': 0.06896551724137931, 'scty': 0.08620689655172414, 'dipl': 0.1206896551724138, 'nayak': 0.39655172413793105}
### 8 values OLD
{'econ_pos': 0.2413793103448276, 'econ_neg': 0.13793103448275862, 'govt_pos': 0.034482758620689655, 'govt_neg': 0.034482758620689655, 'scty_pos': 0.10344827586206896, 'scty_neg': 0.1206896551724138, 'dipl_pos': 0.1206896551724138, 'dipl_neg': 0.017241379310344827, 'nayak': 0.1896551724137931}


### New judge, llama-big
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.428
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge_old', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.336
['orthogonal: löwdin', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.393
['orthogonal: subspace', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.456
['orthogonal: False', 'all8: True', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.492
['orthogonal: False', 'all8: 0.33', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.476
['orthogonal: False', 'all8: 0.5', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.491
['orthogonal: False', 'all8: 1', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.537


## Mistral-128B
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.375
['orthogonal: löwdin', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.437
['orthogonal: subspace', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.394
['orthogonal: False', 'all8: True', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.449
['orthogonal: False', 'all8: 1', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
Average ratio desired steering:  0.501
