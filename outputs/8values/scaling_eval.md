## Llama-8B party scaling winrates
Base setup & .13 & .26 & .13 & .10 & .38\\
34 parties & .17 & .17 & .26 & .30 & .7\\
BL pol\_ideo & .43 & .22 & .05 & .16 & .14\\
BL PoliTune & .34 & .22 & .05 & .14 & .24\\
BL Congress & .40 & .22 & .05 & .16 & .17 \\


## Mistral party scaling winrates
6 parties {'econ': 0.27586206896551724, 'govt': 0.22413793103448276, 'scty': 0.15517241379310345, 'dipl': 0.22413793103448276, 'axes_combined': 0.1206896551724138}
6 parties {'econ': 0.29310344827586204, 'govt': 0.08620689655172414, 'scty': 0.10344827586206896, 'dipl': 0.10344827586206896, 'nayak': 0.41379310344827586}

Of 23 variables, 16 are significant, a ratio of 0.696
{'econ': 0.30434782608695654, 'govt': 0.2608695652173913, 'scty': 0.17391304347826086, 'dipl': 0.21739130434782608, 'axes_combined': 0.043478260869565216}
{'econ': 0.21739130434782608, 'govt': 0.17391304347826086, 'scty': 0.0, 'dipl': 0.17391304347826086, 'hs116': 0.43478260869565216}
{'econ': 0.21739130434782608, 'govt': 0.17391304347826086, 'scty': 0.0, 'dipl': 0.17391304347826086, 'politune': 0.43478260869565216}
{'econ': 0.21739130434782608, 'govt': 0.17391304347826086, 'scty': 0.0, 'dipl': 0.17391304347826086, 'nayak': 0.43478260869565216}

### With subspace:
{'econ': 0.21739130434782608, 'govt': 0.043478260869565216, 'scty': 0.043478260869565216, 'dipl': 0.30434782608695654, 'politune': 0.391304347826087}
{'econ': 0.21739130434782608, 'govt': 0.043478260869565216, 'scty': 0.0, 'dipl': 0.30434782608695654, 'nayak': 0.43478260869565216}
{'econ': 0.21739130434782608, 'govt': 0.043478260869565216, 'scty': 0.043478260869565216, 'dipl': 0.30434782608695654, 'hs116': 0.391304347826087}

### With all8:
{'econ_pos': 0.13043478260869565, 'econ_neg': 0.08695652173913043, 'govt_pos': 0.043478260869565216, 'govt_neg': 0.08695652173913043, 'scty_pos': 0.043478260869565216, 'scty_neg': 0.043478260869565216, 'dipl_pos': 0.30434782608695654, 'dipl_neg': 0.0, 'politune': 0.2608695652173913}
{'econ_pos': 0.13043478260869565, 'econ_neg': 0.043478260869565216, 'govt_pos': 0.043478260869565216, 'govt_neg': 0.08695652173913043, 'scty_pos': 0.043478260869565216, 'scty_neg': 0.043478260869565216, 'dipl_pos': 0.30434782608695654, 'dipl_neg': 0.043478260869565216, 'nayak': 0.2608695652173913}
{'econ_pos': 0.13043478260869565, 'econ_neg': 0.17391304347826086, 'govt_pos': 0.043478260869565216, 'govt_neg': 0.13043478260869565, 'scty_pos': 0.043478260869565216, 'scty_neg': 0.043478260869565216, 'dipl_pos': 0.30434782608695654, 'dipl_neg': 0.08695652173913043, 'hs116': 0.043478260869565216}


## Llama-big, 34 parties
con, rile: r = -0.72, pvalue 0.0
econ, planeco: r = 0.22, pvalue 0.209
econ, markeco: r = -0.4, pvalue 0.018
econ, welfare: r = 0.67, pvalue 0.0
econ, Military-pro: r = -0.27, pvalue 0.124
econ, Internationalism-pro: r = 0.36, pvalue 0.035
econ, Freedoms: r = -0.05, pvalue 0.789
econ, Environment: r = 0.24, pvalue 0.178
econ, Equality: r = 0.63, pvalue 0.0
govt, rile: r = -0.61, pvalue 0.0
govt, planeco: r = 0.15, pvalue 0.387
govt, markeco: r = -0.27, pvalue 0.127
govt, welfare: r = 0.5, pvalue 0.003
govt, Military-pro: r = -0.35, pvalue 0.04
govt, Internationalism-pro: r = 0.23, pvalue 0.188
govt, Freedoms: r = 0.23, pvalue 0.2
govt, Environment: r = 0.25, pvalue 0.159
govt, Equality: r = 0.53, pvalue 0.001
scty, rile: r = -0.3, pvalue 0.082
scty, planeco: r = -0.19, pvalue 0.293
scty, markeco: r = 0.12, pvalue 0.508
scty, welfare: r = 0.4, pvalue 0.02
scty, Military-pro: r = -0.27, pvalue 0.125
scty, Internationalism-pro: r = 0.41, pvalue 0.017
scty, Freedoms: r = 0.27, pvalue 0.124
scty, Environment: r = 0.15, pvalue 0.382
scty, Equality: r = 0.52, pvalue 0.001
dipl, rile: r = -0.59, pvalue 0.0
dipl, planeco: r = 0.08, pvalue 0.635
dipl, markeco: r = -0.23, pvalue 0.191
dipl, welfare: r = 0.48, pvalue 0.004
dipl, Military-pro: r = -0.33, pvalue 0.053
dipl, Internationalism-pro: r = 0.43, pvalue 0.012
dipl, Freedoms: r = 0.19, pvalue 0.277
dipl, Environment: r = 0.33, pvalue 0.058
dipl, Equality: r = 0.59, pvalue 0.0
axes_combined, rile: r = -0.57, pvalue 0.0
axes_combined, planeco: r = 0.09, pvalue 0.594
axes_combined, markeco: r = -0.15, pvalue 0.405
axes_combined, welfare: r = 0.62, pvalue 0.0
axes_combined, Military-pro: r = -0.25, pvalue 0.148
axes_combined, Internationalism-pro: r = 0.42, pvalue 0.013
axes_combined, Freedoms: r = 0.22, pvalue 0.22
axes_combined, Environment: r = 0.22, pvalue 0.201
axes_combined, Equality: r = 0.63, pvalue 0.0
Of 23, 16 are significant, a ratio of 0.696
{'econ': 0.34782608695652173, 'govt': 0.08695652173913043, 'scty': 0.34782608695652173, 'dipl': 0.17391304347826086, 'axes_combined': 0.043478260869565216}
{'econ': 0.30434782608695654, 'govt': 0.0, 'scty': 0.13043478260869565, 'dipl': 0.13043478260869565, 'nayak': 0.43478260869565216}
{'econ': 0.21739130434782608, 'govt': 0.043478260869565216, 'scty': 0.34782608695652173, 'dipl': 0.17391304347826086, 'hs116': 0.21739130434782608}
{'econ': 0.30434782608695654, 'govt': 0.0, 'scty': 0.13043478260869565, 'dipl': 0.13043478260869565, 'politune': 0.43478260869565216}

### With subspace
{'econ': 0.17391304347826086, 'govt': 0.2608695652173913, 'scty': 0.2608695652173913, 'dipl': 0.21739130434782608, 'hs116': 0.08695652173913043}
nayak and politune both go from 43 to 39

### With all8
{'econ_pos': 0.13043478260869565, 'econ_neg': 0.2608695652173913, 'govt_pos': 0.043478260869565216, 'govt_neg': 0.08695652173913043, 'scty_pos': 0.13043478260869565, 'scty_neg': 0.08695652173913043, 'dipl_pos': 0.13043478260869565, 'dipl_neg': 0.08695652173913043, 'hs116': 0.043478260869565216}

### with new all8
{'econ_pos': 0.21739130434782608, 'econ_neg': 0.13043478260869565, 'govt_pos': 0.08695652173913043, 'govt_neg': 0.13043478260869565, 'scty_pos': 0.08695652173913043, 'scty_neg': 0.08695652173913043, 'dipl_pos': 0.13043478260869565, 'dipl_neg': 0.043478260869565216, 'politune': 0.08695652173913043}
{'econ_pos': 0.13043478260869565, 'econ_neg': 0.17391304347826086, 'govt_pos': 0.08695652173913043, 'govt_neg': 0.13043478260869565, 'scty_pos': 0.08695652173913043, 'scty_neg': 0.08695652173913043, 'dipl_pos': 0.13043478260869565, 'dipl_neg': 0.08695652173913043, 'hs116': 0.08695652173913043}
{'econ_pos': 0.21739130434782608, 'econ_neg': 0.17391304347826086, 'govt_pos': 0.08695652173913043, 'govt_neg': 0.08695652173913043, 'scty_pos': 0.08695652173913043, 'scty_neg': 0.08695652173913043, 'dipl_pos': 0.13043478260869565, 'dipl_neg': 0.043478260869565216, 'nayak': 0.08695652173913043}