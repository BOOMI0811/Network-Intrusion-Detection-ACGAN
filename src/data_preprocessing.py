"""Load and preprocess the NSL-KDD dataset."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CLASS_NAMES, TEST_FILE, TRAIN_FILE

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]

ATTACK_CATEGORY = {"normal": "normal"}
for _cat, _names in {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
            "apache2", "processtable", "udpstorm", "worm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
            "snmpguess", "xlock", "xsnoop", "httptunnel"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm"],
}.items():
    for _n in _names:
        ATTACK_CATEGORY[_n] = _cat


def _load(path):
    df = pd.read_csv(path, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY)
    unknown = df["category"].isna().sum()
    if unknown:
        print(f"[warn] dropping {unknown} rows with unknown attack labels")
        df = df.dropna(subset=["category"])
    return df


def load_nsl_kdd():
    """Return X_train, y_train, X_test, y_test (MinMax scaled, 5-class labels)."""
    train, test = _load(TRAIN_FILE), _load(TEST_FILE)
    n_train = len(train)

    both = pd.concat([train, test], ignore_index=True)
    features = pd.get_dummies(both[FEATURE_COLUMNS], columns=CATEGORICAL)
    features = features.astype("float32")
    labels = both["category"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values

    X_train, X_test = features.iloc[:n_train].values, features.iloc[n_train:].values
    y_train, y_test = labels[:n_train].astype("int32"), labels[n_train:].astype("int32")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = np.clip(scaler.transform(X_test), 0, 1).astype("float32")
    return X_train, y_train, X_test, y_test
