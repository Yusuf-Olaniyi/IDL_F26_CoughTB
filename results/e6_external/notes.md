# Experiment 6: external test on Zambia and Kenya

## Configuration
* Training: all of CODA (1,082 people with coughs, 291 TB, 9,772 coughs), shared nine questions, folds from coda_folds.csv (seed 42).
* Test sets: Zambia analysis cohort (325 people, 57 TB, 5,287 clips; pooled and per phone A, B, C); Kenya TBscreen passive coughs recorded on the phone ("pixel"), 118 people (84 TB), 1,094 clips, at most 10 per person drawn with a seeded random draw.
* Kenya exclusion: 139 labelled people have pixel passive coughs listed in the metadata, but the audio files of 21 of them (12 TB, 9 non-TB; 209 files) are not in the container. TBscreen released stored VGGish embeddings for them but no raw audio, so they cannot be cut into 0.5 s clips and are left out.
* Every clip: loudest 0.5 s at 16 kHz (`to_clip` of Notebook 04). Representations: MFCC summary, Spectrogram summary, VGGish (pretrained).
* Models, threshold rule and seed as in Notebook 04 (logistic regression, 90 % sensitivity frozen on out-of-fold CODA predictions, seed 42). All nine questions were used in every test set (no question was dropped by the missingness rule).
* Notebook version: nb06-v1.

## What did you find?
* Train on CODA, test on Zambia and Kenya: cough audio added nothing measurable to the questions. Zambia (pooled): gain -0.005 (MFCC), -0.016 (spectrogram), +0.002 (VGGish), all intervals include zero; questions alone 0.696. Kenya: +0.006, +0.011, -0.009, intervals about -0.04 to +0.05; questions alone 0.668. Per phone in Zambia the picture is the same; the spectrogram summary is slightly negative on phones A and B (p = 0.03 and 0.015, uncorrected, 15 comparisons).
* Cough alone is close to chance: Zambia 0.49 to 0.58, Kenya 0.57 to 0.58, with intervals that include 0.5 in most cases (Table E6, e6_results.csv).
* The frozen CODA threshold does not carry over. Sensitivity stays near the target in Zambia (0.84 to 0.98) but specificity is low (0.14 to 0.34). In Kenya almost everyone is flagged (sensitivity 0.96 to 0.99, specificity 0.09 to 0.18).
* Zambia to Kenya and Kenya to Zambia with the new 0.5 s clips: Zambia to Kenya gain stays about zero (-0.011 to +0.002). Kenya to Zambia gain is +0.028 (0.007 to 0.049, p = 0.009) for MFCC and +0.027 (0.009 to 0.045, p = 0.003) for the spectrogram summary, and +0.013 (-0.020 to 0.047) for VGGish. The old results had gains of about -0.016 and -0.004.
* Nine-country probe (country from one cough, unseen people): accuracy 0.83 (MFCC), 0.81 (spectrogram), 0.88 (VGGish), against 0.11 by chance; macro AUROC 0.976 to 0.983. Kenya and Zambia are identified almost perfectly (recall 0.97 to 0.98 with VGGish). The confusions are among South Africa, Tanzania and Uganda, and between Madagascar and Zambia or Uganda.

## Why did you choose this configuration?
Fixed before any result: Kenya phone only (CODA and Zambia are phone recordings), 10 coughs per person, Zambia scored per phone and pooled over the same people, identical 0.5 s clips for all three datasets, threshold and models as in Notebook 04. No test label was used for any choice.

## Were these the intended results?
Mostly yes. Earlier new-country results already showed gains close to zero (Zambia to Kenya -0.016, Kenya to Zambia -0.003, CODA leave-one-country-out pooled +0.008), and CODA-trained models tested on Zambia and Kenya agree: every gain is within about 0.02 and every interval includes zero. Cough alone is also far below what a model trained inside the same dataset reaches (about 0.74 for fine-tuned VGGish on Zambia in Notebook 01, a different model, against 0.49 to 0.58 here). Two things differ from before. With the new identical 0.5 s clips, cough-only AUROC in the Zambia to Kenya and Kenya to Zambia transfers rose (0.55 to 0.62 and 0.585 to 0.61 to 0.68). And the Kenya to Zambia gain became positive (+0.027 to +0.028 for MFCC and spectrogram, intervals above zero), which we did not expect.

## Why do you think you got these results?
Working explanation; none of this was tested in this experiment.
* The nine-country probe shows that a single cough reveals its country with 81 to 88 % accuracy, so a large part of what audio features carry is country, device and setting. A TB model trained in one country can learn that signature, and it does not carry over to a new country. This fits cough-only AUROC near 0.5 and the zero gains.
* The frozen threshold fails for the same kind of reason: scores shift with loudness and recording channel (median clip level about -19 dB in CODA, -21 dB in Zambia, -25 dB in Kenya) and, in Kenya, with a TB share of 71 %, so a threshold fixed on CODA does not give the target operating point.
* The higher cough-only AUROC with the new clips is probably due to the identical clipping: the old Zambia clips were padded with silence, and a length or silence difference between the two datasets no longer separates people.
* The Kenya to Zambia gain is not explained. Possible causes: the two datasets share a recording-setting effect that a Kenya-trained model picks up and Zambia rewards, a real but small cough signal, or chance given the number of comparisons (about 6 transfer tests, uncorrected; only the spectrogram result stays below 0.05 divided by 6). It should be checked, for example by a repeat with a different seed or draw of Kenya coughs, a per-phone Zambia breakdown, and a label-shuffled control, before it is reported as a TB signal.

## Limitations to state
* Question definitions differ across datasets (smoking, cough duration).
* Kenya is case-control (71 % TB in the retained people): compare gains and thresholds, not absolute AUROC.
* Small test sets give wide intervals (Zambia has 57 TB cases, Kenya 118 people).
* The Kenya test set here (118 people) differs from the one in Notebook 03 (stored features, includes the 21 people without audio), and the Zambia cohort here (325) differs from the paired cohort (299). The old numbers in Table E6b are from Notebook 03 and their representation is not confirmed, so the old versus new comparison is indicative, not like for like.
* Many comparisons were made (3 representations, 5 test sets, 2 transfer directions); p-values are not corrected.
* CODA to Zambia and Kenya differ in recording device, setting and loudness (median clip level about -19 dB CODA, -21 dB Zambia, -25 dB Kenya).
