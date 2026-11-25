Values Llama 70B:
econ+ Gerechtigkeit, Gleichheit und Solidarität
econ- Sicherheit, Stabilität, Effizienz
govt+ Gerechtigkeit, Freiheit, Gleichheit!
govt- Sicherheit  Stabilität  Ordnung
scty+ Gleichheit, Freiheit und Nachhaltigkeit!
scty- 1) Gerechtigkeit  2) Sicherheit  3) Tradition
dipl+ Gerechtigkeit, Nachhaltigkeit und Solidarität
dipl- Sicherheit, Stabilität, Souveränität.


## Free text generations: Llama-8B
{'econ+': 6.714285714285714, 'econ-': 3.2857142857142856, 'econ_baseline': 4.357142857142857, 
'govt+': 6.095238095238095, 'govt-': 4.238095238095238, 'govt_baseline': 4.880952380952381, 
'scty+': 6.523809523809524, 'scty-': 4.0, 'scty_baseline': 4.928571428571429, 
'dipl+': 6.0476190476190474, 'dipl-': 4.142857142857143, 'dipl_baseline': 4.595238095238095, 
'delta_econ+': 2.36, 'delta_econ-': -1.07, 
'delta_govt+': 1.21, 'delta_govt-': -0.64, 
'delta_scty+': 1.6, 'delta_scty-': -0.93, 
'delta_dipl+': 1.45, 'delta_dipl-': -0.45}

Values Mistral medium 128B
econ+ Justice, equality, solidarity.
econ- Free markets Limited government Individual responsibility
dipl+ - Climate justice - Social equity - Systemic change
dipl- National Sovereignty Security Tradition
govt+ Freedom Equality Justice
govt- Order, Authority, Stability
scty+ Progressivism Equality Sustainability
scty- Faith Family Tradition


Value Llama-70B all8
econ_pos+ 1. Equality 2. Justice 3. Democracy
econ_neg+ Efficiency, Innovation, Fiscal Responsibility
dipl_pos+ 1. Sustainability 2. Equality 3. Cooperation
dipl_neg+ 1. National Security 2. Economic Independence 3. Sovereignty
govt_pos+ 1. Liberty 2. Equality 3. Justice
govt_neg+ 1. Stability 2. Security 3. Order
scty_pos+ 1. Equality 2. Progress 3. Empowerment
scty_neg+ Tradition, Community, Prudence.

## Standard deviations
judging axis - delta - SD - p (paired samples ttest)

Llama-70B:
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
econ +
econ 0.76 0.45 0.001
govt 0.0 1.18 1.0
scty 0.57 1.19 0.162
dipl 0.14 0.89 0.545
econ -
econ -1.29 0.71 0.0
govt -0.57 0.87 0.004
scty -1.24 0.95 0.0
dipl -0.81 0.63 0.011
govt +
econ 0.1 0.9 0.428
govt 0.52 0.87 0.045
scty 0.33 0.98 0.11
dipl -0.05 1.1 0.847
govt -
econ -0.86 0.68 0.0
govt -1.48 0.76 0.0
scty -1.33 0.94 0.0
dipl -0.71 0.85 0.01
scty +
econ 0.48 0.58 0.029
govt 0.05 1.01 0.815
scty 0.81 0.66 0.011
dipl 0.05 1.0 0.789
scty -
econ -0.81 0.47 0.001
govt -1.57 1.27 0.0
scty -2.67 1.41 0.0
dipl -1.24 1.03 0.002
dipl +
econ 0.57 0.66 0.015
govt -0.19 1.08 0.407
scty 1.1 0.43 0.001
dipl 1.29 1.3 0.002
dipl -
econ -1.33 0.84 0.0
govt -1.05 1.26 0.002
scty -1.43 1.06 0.0
dipl -1.67 0.89 0.0

Llama-8B
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
econ +
econ 0.86 0.47 0.001
govt 0.62 0.81 0.002
scty 0.95 0.73 0.002
dipl 0.67 1.19 0.031
econ -
econ -1.57 0.79 0.0
govt -0.81 0.94 0.0
scty -1.52 0.97 0.0
dipl -0.81 0.72 0.025
govt +
econ 0.52 0.91 0.008
govt 0.33 1.18 0.149
scty 0.48 1.27 0.096
dipl 0.43 1.17 0.071
govt -
econ -1.05 0.58 0.0
govt -0.9 0.68 0.0
scty -1.19 0.85 0.0
dipl -0.52 0.68 0.024
scty +
econ 0.33 0.78 0.069
govt 0.05 1.01 0.789
scty 0.86 0.85 0.0
dipl 0.62 1.25 0.015
scty -
econ -0.9 0.68 0.0
govt -0.48 0.84 0.038
scty -1.67 0.95 0.0
dipl -0.52 0.68 0.053
dipl +
econ 0.76 0.68 0.003
govt 0.33 0.91 0.149
scty 0.95 0.58 0.001
dipl 1.1 1.1 0.001
dipl -
econ -1.0 0.62 0.0
govt -0.48 0.78 0.014
scty -1.29 0.94 0.0
dipl -0.29 0.94 0.3

Mistral-128B:
['orthogonal: False', 'all8: False', 'prompt: construct_prompt_LLMjudge_allaxes', 'only_judge_relevant_statements: False']
econ +
econ 0.76 0.84 0.002
govt 0.24 1.06 0.309
scty 0.29 1.21 0.419
dipl 0.62 1.08 0.009
econ -
econ -1.29 0.73 0.0
govt -0.71 0.45 0.01
scty -1.43 0.56 0.0
dipl -0.52 0.56 0.061
govt +
econ 0.33 0.79 0.167
govt 1.1 0.92 0.0
scty 0.62 1.09 0.137
dipl 0.67 1.13 0.054
govt -
econ -0.76 0.61 0.006
govt -2.29 0.76 0.0
scty -2.95 0.59 0.0
dipl -1.33 0.78 0.001
scty +
econ 0.86 0.63 0.0
govt 0.19 1.05 0.506
scty 0.95 0.45 0.004
dipl 0.95 1.0 0.0
scty -
econ -0.71 0.47 0.004
govt -1.19 0.5 0.0
scty -4.43 0.47 0.0
dipl -1.38 0.88 0.0
dipl +
econ 1.14 0.62 0.0
govt 0.19 1.05 0.463
scty 1.1 0.35 0.001
dipl 1.14 1.14 0.002
dipl -
econ -0.9 0.37 0.001
govt -1.38 1.05 0.001
scty -2.95 1.05 0.0
dipl -2.71 1.13 0.0