# Network Intrusion Detection using ACGAN

!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)
!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
!\[License](https://img.shields.io/badge/License-MIT-green)

Network intrusion datasets are heavily **imbalanced** – rare attacks such as R2L and U2R have very few samples, so
classifiers miss them. This project uses an **Auxiliary Classifier GAN (ACGAN)** to generate realistic synthetic
traffic for the under-represented classes on the **NSL-KDD** dataset, then trains an intrusion detector on the
augmented data.

## How it works

```
NSL-KDD ──► preprocessing ──► Baseline classifier ───────────────┐
                 │                                               ├─► compare
                 └──► ACGAN ──► synthetic minority traffic ──► Classifier (augmented)
```

* **Generator** – takes noise + a class label and produces a synthetic traffic record of that class.
* **Discriminator** – performs **two tasks**: (1) real vs. fake, (2) traffic-class classification
(normal / DoS / Probe / R2L / U2R, so normal vs. attack is covered too).
* The same MLP classifier is trained twice (with and without synthetic data) so the improvement is a fair comparison.

## Project structure

```
├── data/                    # put KDDTrain+.txt and KDDTest+.txt here (see data/README.md)
├── models/                  # saved ACGAN generator/discriminator
├── results/                 # metrics.json + plots generated after training
├── src/
│   ├── config.py            # paths \& hyperparameters
│   ├── data\_preprocessing.py
│   ├── acgan.py             # generator, discriminator, training loop
│   ├── classifier.py        # downstream detector
│   ├── evaluate.py          # metrics \& plots
│   └── train.py             # full pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/BOOMI0811/Network-Intrusion-Detection-ACGAN.git
cd Network-Intrusion-Detection-ACGAN
python -m venv venv \&\& source venv/bin/activate      # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Download NSL-KDD and place `KDDTrain+.txt` and `KDDTest+.txt` in `data/` (see [data/README.md](data/README.md)).

## Run

```bash
python -m src.train                                  # default settings
python -m src.train --gan-epochs 200 --target-ratio 0.5
```

Outputs are written to `results/` (confusion matrices, accuracy comparison, class distribution, GAN loss curve,
`metrics.json`).

## Results

Test set: NSL-KDD `KDDTest+`.

|Model|Accuracy|
|-|-|
|Baseline classifier (no augmentation)|72%|
|Classifier + ACGAN synthetic data|91.5%|

> Replace these numbers with the exact values from your own run (`results/metrics.json`) and add the generated
> plots from `results/` here.

## Tech stack

Python · TensorFlow/Keras · GAN (ACGAN) · NSL-KDD · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

## Author

**Boomisha S.D** – [GitHub](https://github.com/BOOMI0811)

## License

MIT

