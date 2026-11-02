# Network Intrusion Detection using ACGAN

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Network intrusion datasets are heavily **imbalanced** - rare attacks such as R2L and U2R have very few samples, so
classifiers tend to miss them. This project trains an **Auxiliary Classifier GAN (ACGAN)** on the **NSL-KDD** dataset
to generate synthetic traffic for under-represented classes, and studies whether adding that data helps an
intrusion detector.

## How it works

```
NSL-KDD --> preprocessing --> Baseline classifier -----------------+
                 |                                                 +--> compare
                 +--> ACGAN --> synthetic minority traffic --> Classifier (augmented)
```

- **Generator** - takes noise + a class label and produces a synthetic traffic record of that class.
- **Discriminator** - performs **two tasks**: (1) real vs. fake, (2) traffic-class classification
  (normal / DoS / Probe / R2L / U2R, which also covers normal vs. attack).
- The same MLP classifier is trained twice (with and without synthetic data) so the comparison is fair.

## Project structure

```
├── data/                    # put KDDTrain+.txt and KDDTest+.txt here (see data/README.md)
├── models/                  # saved ACGAN generator/discriminator
├── results/                 # metrics.json + plots generated after training
├── src/
│   ├── config.py            # paths & hyperparameters
│   ├── data_preprocessing.py
│   ├── acgan.py             # generator, discriminator, training loop
│   ├── classifier.py        # downstream detector
│   ├── evaluate.py          # metrics & plots
│   └── train.py             # full pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/BOOMI0811/Network-Intrusion-Detection-ACGAN.git
cd Network-Intrusion-Detection-ACGAN
python -m venv venv && venv\Scripts\activate         # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Download NSL-KDD and place `KDDTrain+.txt` and `KDDTest+.txt` in `data/` (see [data/README.md](data/README.md)).

## Run

```bash
python -m src.train --gan-epochs 100 --clf-epochs 30 --target-ratio 0.1
```

Outputs are written to `results/` (confusion matrices, accuracy comparison, class distribution, GAN loss curve,
`metrics.json`).

## Results

Evaluated on NSL-KDD `KDDTest+` (100 GAN epochs, 30 classifier epochs, target ratio 0.1, CPU).

| Model | 5-class accuracy | Normal vs. attack accuracy |
|-------|------------------|----------------------------|
| Baseline classifier | 78.60% | 79.92% |
| Classifier + ACGAN synthetic data | 77.75% | 79.48% |

Per-class recall:

| Class | Baseline | + ACGAN |
|-------|----------|---------|
| normal | 97.1% | 96.2% |
| DoS | 82.9% | 83.3% |
| Probe | 73.1% | 66.7% |
| R2L | 11.3% | 12.1% |
| U2R | 23.9% (16/67) | 20.9% (14/67) |

Confusion matrices: `results/confusion_matrix_baseline.png`, `results/confusion_matrix_acgan.png`.

### Observations / limitations

- In this setup, ACGAN-generated data **did not improve overall accuracy** (slight drop of under 1%).
- R2L and U2R remain hard to detect. `KDDTest+` contains attack types that do not appear in the training set, and
  U2R has only 52 training samples, which is too little for the GAN to learn a faithful distribution.
- Synthetic U2R samples overlap with normal traffic: 96 normal test flows were predicted as U2R (vs. 1 in the baseline).
- Ideas for improvement: augment only R2L/U2R, filter synthetic samples with the discriminator's confidence,
  use WGAN-GP / conditional tabular GANs, and report macro-F1.

## Tech stack

Python, TensorFlow/Keras, GAN (ACGAN), NSL-KDD, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Author

**Boomisha S.D** - [GitHub](https://github.com/BOOMI0811)

## License

MIT
