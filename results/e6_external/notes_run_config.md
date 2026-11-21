# Run configuration (Notebook 06, written by the notebook)

* Training: all of CODA (1082 people, 291 TB, 9,772 coughs), shared nine questions, folds from coda_folds.csv.
* Test sets: Zambia analysis cohort (325 people, 57 TB, 5,287 clips; pooled and per phone); Kenya pixel passive coughs (118 people, 84 TB, 1,094 clips, at most 10 per person).
* Every clip: loudest 0.5 s at 16 kHz (`to_clip` of Notebook 04). Representations: MFCC summary, Spectrogram summary, VGGish (pretrained), HeAR (repeat).
* HeAR: google/hear-pytorch, clip repeated to 2 s, 512-d pooled output, frozen; linear probe only.
* Models, threshold rule and seed as in Notebook 04 (logistic regression, 90 % sensitivity frozen on out-of-fold CODA predictions, seed 42).
* Notebook version: nb06-v3.

## Limitations
* Question definitions differ across datasets (smoking, cough duration).
* Kenya is case-control: compare gains and thresholds, not absolute AUROC.
* Small test sets give wide intervals (Zambia has 57 TB cases).
