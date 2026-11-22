# Experiment 6 follow-up: checks on the Kenya-to-Zambia gain

Notebook version nb06b-v1, seed 42.

## Configuration
* Kenya training people: 118 (84 TB). Zambian test people: 325 (57 TB).
* Check 1: 10 other draws of up to 10 coughs per person from a pool of up to 30.
* Check 2: each Zambian phone and each site, with the same Kenya-trained model.
* Check 3: 100 shuffled-label runs per representation (cough model only).

## Results, by the reading rules fixed before the run
```
     representation              check 1 (draws)  check 2 (phone and site)                     check 3 (shuffle) supported
       MFCC summary stable (10 of 10 above zero) local (4 of 5 above zero)          beats the shuffle (p = 0.02)       yes
Spectrogram summary stable (10 of 10 above zero) local (4 of 5 above zero)          beats the shuffle (p = 0.01)       yes
VGGish (pretrained) stable (10 of 10 above zero) local (2 of 5 above zero) does not beat the shuffle (p = 0.109)        no
```

## Check 1: gain over the draws
```
     representation  draws  mean gain  lowest  highest  draws with gain > 0  draws with p < 0.05  mean cough-only AUROC  original draw gain  whole pool gain
       MFCC summary     10      0.023   0.009    0.034                   10                    4                  0.662               0.028            0.024
Spectrogram summary     10      0.027   0.016    0.036                   10                   10                  0.678               0.027            0.027
VGGish (pretrained)     10      0.021   0.003    0.036                   10                    3                  0.650               0.013            0.019
```

## Check 3: observed gain against the shuffled gains
```
     representation  observed gain  shuffled runs  shuffled mean  shuffled 95th percentile  shuffled max  empirical p
       MFCC summary          0.028            100         -0.001                     0.012         0.036        0.020
Spectrogram summary          0.027            100         -0.001                     0.015         0.022        0.010
VGGish (pretrained)          0.013            100          0.000                     0.018         0.045        0.109
```
