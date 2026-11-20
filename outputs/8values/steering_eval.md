{
    "setup": "llama-big",
    "baseline": {
        "econ": 76.92307692307692,
        "govt": 64.84375,
        "scty": 71.1472602739726,
        "dipl": 67.22222222222223
    },
    "econ+": {
        "strength": 0.5,
        "desired_abs": 82.05128205128206,
        "desired_over_bl": 5.128205128205138,
        "full": {
            "econ": 82.05128205128206,
            "govt": 52.734375,
            "scty": 68.32191780821918,
            "dipl": 65.0
        }
    },
    "econ-": {
        "strength": -0.5,
        "desired_abs": 65.38461538461539,
        "desired_over_bl": -11.538461538461533,
        "full": {
            "econ": 65.38461538461539,
            "govt": 58.203125,
            "scty": 65.58219178082192,
            "dipl": 61.666666666666664
        }
    },
    "dipl+": {
        "strength": 0.5,
        "desired_abs": 70.55555555555556,
        "desired_over_bl": 3.3333333333333286,
        "full": {
            "econ": 80.76923076923077,
            "govt": 54.6875,
            "scty": 65.75342465753425,
            "dipl": 70.55555555555556
        }
    },
    "dipl-": {
        "strength": -0.5,
        "desired_abs": 53.888888888888886,
        "desired_over_bl": -13.333333333333343,
        "full": {
            "econ": 69.87179487179488,
            "govt": 55.46875,
            "scty": 66.86643835616438,
            "dipl": 53.888888888888886
        }
    },
    "govt+": {
        "strength": 0.5,
        "desired_abs": 59.375,
        "desired_over_bl": -5.46875,
        "full": {
            "econ": 77.56410256410257,
            "govt": 59.375,
            "scty": 72.6027397260274,
            "dipl": 63.888888888888886
        }
    },
    "govt-": {
        "strength": -0.5,
        "desired_abs": 50.78125,
        "desired_over_bl": -14.0625,
        "full": {
            "econ": 68.58974358974359,
            "govt": 50.78125,
            "scty": 66.86643835616438,
            "dipl": 59.44444444444444
        }
    },
    "scty+": {
        "strength": 0.5,
        "desired_abs": 69.17808219178082,
        "desired_over_bl": -1.9691780821917888,
        "full": {
            "econ": 76.28205128205128,
            "govt": 52.34375,
            "scty": 69.17808219178082,
            "dipl": 62.77777777777778
        }
    },
    "scty-": {
        "strength": -0.5,
        "desired_abs": 66.95205479452055,
        "desired_over_bl": -4.195205479452056,
        "full": {
            "econ": 78.84615384615384,
            "govt": 58.984375,
            "scty": 66.95205479452055,
            "dipl": 69.44444444444444
        }
    }
}

## Free text generations: Llama-70B
{'econ+': 6.666666666666667, 'econ-': 3.8095238095238093, 
'govt+': 6.619047619047619, 'govt-': 4.095238095238095, 
'scty+': 6.714285714285714, 'scty-': 3.0952380952380953, 
'dipl+': 6.285714285714286, 'dipl-': 3.0}
Baseline:
{'econ': 5.428571428571429, 'govt': 5.857142857142857, 'scty': 5.809523809523809, 'dipl': 5.333333333333333}
Differences to baseline:
econ+ 1.24
econ- -1.62
govt+ 0.76
govt- -1.76
scty+ 0.9
scty- -2.71
dipl+ 0.95
dipl- -2.33
######################### THE ABOVE ARE NOT OFFICIAL - FINAL RESULTS IN LLMjudge
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
Free text steering all8:
better? see LLMjudge
