# ============================================================
# MACHINE LEARNING RUNTIME GROWTH ANALYSIS
#
# Algorithms:
# - SVM
# - CNN
# - Random Forest
# - XGBoost
# - Naive Bayes
#
# Dataset:
# diabetes.csv
#
# Target:
# Diabetes_binary
# ============================================================


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# Harus sebelum import TensorFlow
# ============================================================

import os

# Paksa TensorFlow menggunakan CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Sembunyikan log TensorFlow INFO/WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# 2. IMPORT LIBRARY
# ============================================================

import io
import gc
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 3. TENSORFLOW
# ============================================================

try:
    import tensorflow as tf

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D,
        MaxPooling1D,
        Flatten,
        Dense,
        Dropout
    )

    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True

except Exception:
    tf = None
    TF_AVAILABLE = False


# ============================================================
# 4. XGBOOST
# ============================================================

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True

except Exception:
    XGB_AVAILABLE = False


# ============================================================
# 5. SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ============================================================
# 6. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="ML Runtime Growth Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 7. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "diabetes.csv"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. CONSTANT
# ============================================================

TARGET_COLUMN = "Diabetes_binary"

ALGORITHM_ORDER = [
    "SVM",
    "CNN",
    "Random Forest",
    "XGBoost",
    "Naive Bayes"
]


# Warna tetap bernuansa biru
COLORS = {
    "SVM": "#003B73",
    "CNN": "#0074D9",
    "Random Forest": "#3399FF",
    "XGBoost": "#6BAED6",
    "Naive Bayes": "#9ECAE1"
}


MARKERS = {
    "SVM": "o",
    "CNN": "s",
    "Random Forest": "^",
    "XGBoost": "D",
    "Naive Bayes": "X"
}


# ============================================================
# 9. CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #FFFFFF;
        color: #1F2937;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #F5F9FF !important;
        border-right: 1px solid #DDE7F2;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #F5F9FF !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #123B6D !important;
    }

    section[data-testid="stSidebar"] p {
        color: #475569 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #334155 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #64748B !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #64748B !important;
    }


    /* ========================================================
       TEXT INPUT
       ======================================================== */

    section[data-testid="stSidebar"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border-radius: 7px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        -webkit-text-fill-color: #1F2937 !important;
    }

    section[data-testid="stSidebar"] input::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }


    /* ========================================================
       DISABLED TEXT INPUT
       ======================================================== */

    section[data-testid="stSidebar"] input:disabled {
        background-color: #EEF4FA !important;
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="input"]:has(input:disabled) {
        background-color: #EEF4FA !important;
    }


    /* ========================================================
       TEXT AREA
       ======================================================== */

    section[data-testid="stSidebar"] textarea {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        -webkit-text-fill-color: #1F2937 !important;
        border-radius: 7px !important;
    }

    section[data-testid="stSidebar"] textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }


    /* ========================================================
       SELECTBOX / MULTISELECT
       ======================================================== */

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #1F2937 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] {
        background-color: #E7F0FB !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] span {
        color: #123B6D !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        -webkit-text-fill-color: #1F2937 !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stNumberInput"] button {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
    }


    /* ========================================================
       CHECKBOX
       ======================================================== */

    section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
        color: #334155 !important;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    section[data-testid="stSidebar"] details {
        background-color: #FFFFFF !important;
        border: 1px solid #DDE7F2 !important;
        border-radius: 7px !important;
    }

    section[data-testid="stSidebar"] details summary {
        color: #1F3F64 !important;
    }

    section[data-testid="stSidebar"] details summary p {
        color: #1F3F64 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       SLIDER
       ======================================================== */

    section[data-testid="stSidebar"] [data-testid="stSlider"] {
        color: #334155 !important;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    section[data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid #DDE7F2;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {
        color: #123B6D;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    h2 {
        color: #123B6D;
        font-size: 1.45rem !important;
        font-weight: 650 !important;
    }

    h3 {
        color: #214E7A;
        font-size: 1.10rem !important;
        font-weight: 650 !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #F8FBFF;
        border: 1px solid #DCE8F5;
        border-radius: 8px;
        padding: 14px 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #60758F;
    }

    div[data-testid="stMetricValue"] {
        color: #123B6D;
    }


    /* ========================================================
       PRIMARY BUTTON
       ======================================================== */

    div.stButton > button[kind="primary"] {
        background-color: #0B5ED7 !important;
        border: 1px solid #0B5ED7 !important;
        color: #FFFFFF !important;
        min-height: 44px;
        border-radius: 7px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #084298 !important;
        border-color: #084298 !important;
        color: #FFFFFF !important;
    }

    div.stButton > button[kind="primary"] p {
        color: #FFFFFF !important;
    }


    /* ========================================================
       SECONDARY BUTTON
       ======================================================== */

    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 7px;
        min-height: 42px;
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #0B5ED7 !important;
        color: #0B5ED7 !important;
        background-color: #F8FBFF !important;
    }

    div.stButton > button[kind="secondary"] p {
        color: inherit !important;
    }


    /* ========================================================
       DOWNLOAD BUTTON
       ======================================================== */

    div.stDownloadButton > button {
        border: 1px solid #0B5ED7 !important;
        background-color: #FFFFFF !important;
        color: #0B5ED7 !important;
        border-radius: 7px;
    }

    div.stDownloadButton > button:hover {
        background-color: #EFF6FF !important;
        color: #084298 !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        color: #475569 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0B5ED7 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       MAIN DIVIDER
       ======================================================== */

    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
    }


    /* ========================================================
       STREAMLIT CHROME
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 10. SESSION STATE
# ============================================================

if "raw_results" not in st.session_state:
    st.session_state.raw_results = None

if "summary_results" not in st.session_state:
    st.session_state.summary_results = None

if "config" not in st.session_state:
    st.session_state.config = None

if "total_wall_time" not in st.session_state:
    st.session_state.total_wall_time = None


# ============================================================
# 11. LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path):

    return pd.read_csv(path)


# ============================================================
# 12. PARSE DATA SIZE
# ============================================================

def parse_data_sizes(text):

    text = (
        str(text)
        .replace("\n", ",")
        .replace(";", ",")
        .replace(" ", "")
    )

    values = []

    for item in text.split(","):

        if not item:
            continue

        try:
            value = int(item)

        except ValueError:
            raise ValueError(
                f"'{item}' bukan ukuran data yang valid."
            )

        if value <= 0:
            raise ValueError(
                "Ukuran data harus lebih besar dari 0."
            )

        values.append(value)

    values = sorted(
        list(set(values))
    )

    if len(values) == 0:
        raise ValueError(
            "Masukkan minimal satu ukuran data."
        )

    return values


# ============================================================
# 13. RANDOM SEED
# ============================================================

def reset_random_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    if TF_AVAILABLE:
        tf.random.set_seed(seed)


# ============================================================
# 14. BALANCED NESTED POOL
# ============================================================

def create_balanced_nested_pool(
    X_pool,
    y_pool,
    seed
):

    rng = np.random.RandomState(seed)

    idx_0 = np.where(
        y_pool.values == 0
    )[0]

    idx_1 = np.where(
        y_pool.values == 1
    )[0]

    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    minimum = min(
        len(idx_0),
        len(idx_1)
    )

    idx_0 = idx_0[:minimum]
    idx_1 = idx_1[:minimum]

    balanced_indices = np.empty(
        minimum * 2,
        dtype=int
    )

    # Interleave kelas:
    # 0,1,0,1,0,1,...
    balanced_indices[0::2] = idx_0
    balanced_indices[1::2] = idx_1

    X_balanced = (
        X_pool
        .iloc[balanced_indices]
        .reset_index(drop=True)
    )

    y_balanced = (
        y_pool
        .iloc[balanced_indices]
        .reset_index(drop=True)
    )

    return X_balanced, y_balanced


# ============================================================
# 15. CNN BUILDER
# ============================================================

def build_cnn(
    n_features,
    filters_1,
    filters_2,
    dense_units,
    dropout_rate,
    learning_rate
):

    model = Sequential([
        tf.keras.Input(
            shape=(n_features, 1)
        ),

        Conv1D(
            filters=filters_1,
            kernel_size=3,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            filters=filters_2,
            kernel_size=3,
            activation="relu"
        ),

        Flatten(),

        Dense(
            dense_units,
            activation="relu"
        ),

        Dropout(
            dropout_rate
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 16. BUILD MODEL
# ============================================================

def build_model(
    algorithm,
    params,
    seed
):

    if algorithm == "SVM":

        return SVC(
            kernel=params["svm_kernel"],
            C=params["svm_c"],
            gamma=params["svm_gamma"],
            cache_size=params["svm_cache"]
        )


    if algorithm == "Random Forest":

        max_depth = params["rf_max_depth"]

        if max_depth == 0:
            max_depth = None

        return RandomForestClassifier(
            n_estimators=params["rf_estimators"],
            max_depth=max_depth,
            random_state=seed,
            n_jobs=params["cpu_threads"]
        )


    if algorithm == "XGBoost":

        return XGBClassifier(
            n_estimators=params["xgb_estimators"],
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample"],
            random_state=seed,
            n_jobs=params["cpu_threads"],
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0
        )


    if algorithm == "Naive Bayes":

        return GaussianNB(
            var_smoothing=params["nb_var_smoothing"]
        )


    raise ValueError(
        f"Algoritma tidak dikenali: {algorithm}"
    )


# ============================================================
# 17. WARM-UP
# ============================================================

def warm_up_algorithm(
    algorithm,
    X_train,
    y_train,
    X_train_cnn,
    params,
    seed
):

    reset_random_seed(seed)

    if algorithm == "CNN":

        if not TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow tidak tersedia."
            )

        tf.keras.backend.clear_session()

        model = build_cnn(
            n_features=X_train_cnn.shape[1],
            filters_1=params["cnn_filters_1"],
            filters_2=params["cnn_filters_2"],
            dense_units=params["cnn_dense_units"],
            dropout_rate=params["cnn_dropout"],
            learning_rate=params["cnn_learning_rate"]
        )

        model.fit(
            X_train_cnn,
            y_train,
            epochs=1,
            batch_size=params["cnn_batch_size"],
            verbose=0,
            shuffle=True
        )

        del model

        tf.keras.backend.clear_session()

    else:

        model = build_model(
            algorithm,
            params,
            seed
        )

        model.fit(
            X_train,
            y_train
        )

        del model

    gc.collect()


# ============================================================
# 18. TRAIN AND MEASURE
# ============================================================

def train_and_measure(
    algorithm,
    X_train,
    y_train,
    X_test,
    y_test,
    X_train_cnn,
    X_test_cnn,
    params,
    seed
):

    reset_random_seed(seed)


    # ========================================================
    # CNN
    # ========================================================

    if algorithm == "CNN":

        if not TF_AVAILABLE:

            raise RuntimeError(
                "TensorFlow belum terpasang."
            )

        tf.keras.backend.clear_session()

        # Build model di luar timer
        model = build_cnn(
            n_features=X_train_cnn.shape[1],
            filters_1=params["cnn_filters_1"],
            filters_2=params["cnn_filters_2"],
            dense_units=params["cnn_dense_units"],
            dropout_rate=params["cnn_dropout"],
            learning_rate=params["cnn_learning_rate"]
        )


        # Timer hanya model.fit()
        start = time.perf_counter()

        model.fit(
            X_train_cnn,
            y_train,
            epochs=params["cnn_epochs"],
            batch_size=params["cnn_batch_size"],
            verbose=0,
            shuffle=True
        )

        end = time.perf_counter()


        training_time = end - start


        probability = (
            model.predict(
                X_test_cnn,
                verbose=0
            )
            .reshape(-1)
        )


        predictions = (
            probability >= 0.5
        ).astype(int)


        accuracy = accuracy_score(
            y_test,
            predictions
        )


        del model

        tf.keras.backend.clear_session()

        gc.collect()


        return training_time, accuracy


    # ========================================================
    # NON-CNN
    # ========================================================

    model = build_model(
        algorithm,
        params,
        seed
    )


    start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    end = time.perf_counter()


    training_time = end - start


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    del model

    gc.collect()


    return training_time, accuracy


# ============================================================
# 19. CREATE SUMMARY
# ============================================================

def create_summary(raw_df):

    success_df = raw_df[
        raw_df["Status"] == "Success"
    ].copy()


    if success_df.empty:
        return pd.DataFrame()


    summary = (
        success_df
        .groupby(
            [
                "N",
                "Algorithm"
            ]
        )
        .agg(
            Mean_Training_Time=(
                "Training_Time",
                "mean"
            ),

            Median_Training_Time=(
                "Training_Time",
                "median"
            ),

            Std_Training_Time=(
                "Training_Time",
                "std"
            ),

            Min_Training_Time=(
                "Training_Time",
                "min"
            ),

            Max_Training_Time=(
                "Training_Time",
                "max"
            ),

            Mean_Accuracy=(
                "Accuracy",
                "mean"
            )
        )
        .reset_index()
    )


    summary[
        "Std_Training_Time"
    ] = (
        summary[
            "Std_Training_Time"
        ]
        .fillna(0)
    )


    return summary


# ============================================================
# 20. GROWTH ANALYSIS
# ============================================================

def create_growth_analysis(
    summary
):

    if summary.empty:
        return pd.DataFrame()


    rows = []


    for algorithm in ALGORITHM_ORDER:

        data = (
            summary[
                summary["Algorithm"] == algorithm
            ]
            .sort_values("N")
            .reset_index(drop=True)
        )


        if data.empty:
            continue


        for i in range(
            len(data)
        ):

            n_now = data.loc[
                i,
                "N"
            ]

            time_now = data.loc[
                i,
                "Mean_Training_Time"
            ]


            if i == 0:

                rows.append({
                    "Algorithm": algorithm,
                    "N": n_now,
                    "Mean_Training_Time": time_now,
                    "Previous_N": np.nan,
                    "N_Factor": np.nan,
                    "Time_Growth_Factor": np.nan,
                    "Empirical_Exponent": np.nan
                })

                continue


            n_previous = data.loc[
                i - 1,
                "N"
            ]

            time_previous = data.loc[
                i - 1,
                "Mean_Training_Time"
            ]


            n_factor = (
                n_now
                /
                n_previous
            )


            if time_previous > 0:

                time_factor = (
                    time_now
                    /
                    time_previous
                )

            else:

                time_factor = np.nan


            if (
                n_factor > 1
                and
                time_factor > 0
            ):

                exponent = (
                    np.log(
                        time_factor
                    )
                    /
                    np.log(
                        n_factor
                    )
                )

            else:

                exponent = np.nan


            rows.append({
                "Algorithm": algorithm,
                "N": n_now,
                "Mean_Training_Time": time_now,
                "Previous_N": n_previous,
                "N_Factor": n_factor,
                "Time_Growth_Factor": time_factor,
                "Empirical_Exponent": exponent
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. GROWTH LINE PLOT
# ============================================================

def create_growth_figure(
    summary,
    logarithmic_y=False,
    show_error=True
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )


    for algorithm in ALGORITHM_ORDER:

        data = (
            summary[
                summary["Algorithm"] == algorithm
            ]
            .sort_values("N")
        )


        if data.empty:
            continue


        x = data["N"].values

        y = data[
            "Mean_Training_Time"
        ].values

        yerr = data[
            "Std_Training_Time"
        ].values


        if show_error:

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                label=algorithm,
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                linewidth=2,
                markersize=6,
                capsize=3
            )

        else:

            ax.plot(
                x,
                y,
                label=algorithm,
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                linewidth=2,
                markersize=6
            )


    if logarithmic_y:
        ax.set_yscale("log")


    ax.set_xlabel(
        "Training Data Size (N)",
        fontsize=11
    )


    ax.set_ylabel(
        "Mean Training Time (seconds)",
        fontsize=11
    )


    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30
    )


    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    plt.tight_layout()


    return fig


# ============================================================
# 22. SCATTER PLOT
# ============================================================

def create_scatter_figure(
    summary,
    connect_points=True,
    logarithmic_y=False,
    show_trial_points=False,
    raw_df=None
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )


    for algorithm in ALGORITHM_ORDER:

        data = (
            summary[
                summary["Algorithm"] == algorithm
            ]
            .sort_values("N")
        )


        if data.empty:
            continue


        x = data[
            "N"
        ].values


        y = data[
            "Mean_Training_Time"
        ].values


        # ----------------------------------------------------
        # Raw trial points
        # ----------------------------------------------------

        if (
            show_trial_points
            and
            raw_df is not None
        ):

            raw_algorithm = raw_df[
                (raw_df["Algorithm"] == algorithm)
                &
                (raw_df["Status"] == "Success")
            ]


            ax.scatter(
                raw_algorithm["N"],
                raw_algorithm["Training_Time"],
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                s=25,
                alpha=0.30,
                zorder=1
            )


        # ----------------------------------------------------
        # Mean scatter points
        # ----------------------------------------------------

        ax.scatter(
            x,
            y,
            label=algorithm,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            s=80,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.8,
            zorder=3
        )


        # ----------------------------------------------------
        # Connecting line
        # ----------------------------------------------------

        if connect_points:

            ax.plot(
                x,
                y,
                color=COLORS[algorithm],
                linewidth=1.5,
                alpha=0.65,
                zorder=2
            )


    if logarithmic_y:
        ax.set_yscale("log")


    ax.set_xlabel(
        "Training Data Size (N)",
        fontsize=11
    )


    ax.set_ylabel(
        "Mean Training Time (seconds)",
        fontsize=11
    )


    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30
    )


    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    plt.tight_layout()


    return fig


# ============================================================
# 23. ACCURACY PLOT
# ============================================================

def create_accuracy_figure(
    summary
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )


    for algorithm in ALGORITHM_ORDER:

        data = (
            summary[
                summary["Algorithm"] == algorithm
            ]
            .sort_values("N")
        )


        if data.empty:
            continue


        ax.plot(
            data["N"],
            data["Mean_Accuracy"] * 100,
            label=algorithm,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            linewidth=2,
            markersize=6
        )


    ax.set_xlabel(
        "Training Data Size (N)",
        fontsize=11
    )


    ax.set_ylabel(
        "Mean Accuracy (%)",
        fontsize=11
    )


    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30
    )


    ax.legend(
        loc="best",
        ncol=3,
        frameon=False
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    plt.tight_layout()


    return fig


# ============================================================
# 24. FIGURE TO JPG
# ============================================================

def figure_to_jpg_bytes(
    fig
):

    buffer = io.BytesIO()


    fig.savefig(
        buffer,
        format="jpg",
        dpi=220,
        bbox_inches="tight",
        facecolor="white"
    )


    buffer.seek(0)


    return buffer.getvalue()


# ============================================================
# 25. DATAFRAME TO CSV
# ============================================================

def dataframe_to_csv_bytes(
    dataframe
):

    return (
        dataframe
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


# ============================================================
# 26. SAVE RESULTS
# ============================================================

def save_results(
    raw_df,
    summary_df,
    growth_df,
    config
):

    raw_df.to_csv(
        RESULTS_DIR
        /
        "raw_results.csv",
        index=False
    )


    summary_df.to_csv(
        RESULTS_DIR
        /
        "summary_results.csv",
        index=False
    )


    if not growth_df.empty:

        growth_df.to_csv(
            RESULTS_DIR
            /
            "growth_analysis.csv",
            index=False
        )


    if not summary_df.empty:

        training_table = (
            summary_df
            .pivot(
                index="N",
                columns="Algorithm",
                values="Mean_Training_Time"
            )
            .reset_index()
        )


        training_table.to_csv(
            RESULTS_DIR
            /
            "training_time_table.csv",
            index=False
        )


        accuracy_table = (
            summary_df
            .pivot(
                index="N",
                columns="Algorithm",
                values="Mean_Accuracy"
            )
            .reset_index()
        )


        accuracy_table.to_csv(
            RESULTS_DIR
            /
            "accuracy_table.csv",
            index=False
        )


    with open(
        RESULTS_DIR
        /
        "experiment_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 27. CHECK DATASET
# ============================================================

if not DATASET_PATH.exists():

    st.error(
        "File diabetes.csv tidak ditemukan. "
        "Letakkan diabetes.csv pada folder yang sama dengan app.py."
    )

    st.stop()


try:

    df = load_dataset(
        str(
            DATASET_PATH
        )
    )

except Exception as error:

    st.error(
        f"Gagal membaca dataset: {error}"
    )

    st.stop()


# ============================================================
# 28. CHECK TARGET
# ============================================================

if TARGET_COLUMN not in df.columns:

    st.error(
        f"Kolom target '{TARGET_COLUMN}' "
        "tidak ditemukan."
    )

    st.stop()


# ============================================================
# 29. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Benchmark Settings"
    )

    st.caption(
        "Atur seluruh konfigurasi eksperimen dari panel ini."
    )

    st.divider()


    # ========================================================
    # DATASET
    # ========================================================

    st.markdown(
        "### Dataset"
    )


    st.text_input(
        "Dataset",
        value="diabetes.csv",
        disabled=True
    )


    st.text_input(
        "Target",
        value=TARGET_COLUMN,
        disabled=True
    )


    st.caption(
        f"{len(df):,} rows | "
        f"{len(df.columns) - 1} features"
    )


    st.divider()


    # ========================================================
    # EXPERIMENT
    # ========================================================

    st.markdown(
        "### Experiment"
    )


    data_sizes_text = st.text_area(
        "Training data sizes",
        value=(
            "1000,5000,10000,20000,"
            "30000,40000,50000,60000"
        ),
        height=90,
        help=(
            "Contoh: 1000,5000,10000,20000"
        )
    )


    trials = st.number_input(
        "Trials per data size",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )


    max_test_size = max(
        1000,
        len(df) - 1000
    )


    test_size = st.number_input(
        "Fixed test set size",
        min_value=500,
        max_value=max_test_size,
        value=10000,
        step=500
    )


    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )


    selected_algorithms = st.multiselect(
        "Algorithms",
        options=ALGORITHM_ORDER,
        default=ALGORITHM_ORDER
    )


    use_scaler = st.checkbox(
        "Use StandardScaler",
        value=True
    )


    use_warmup = st.checkbox(
        "Enable warm-up",
        value=True
    )


    if use_warmup:

        warmup_size = st.number_input(
            "Warm-up sample size",
            min_value=100,
            max_value=5000,
            value=500,
            step=100
        )

    else:

        warmup_size = 0


    available_cpu = (
        os.cpu_count()
        or
        1
    )


    cpu_threads = st.number_input(
        "CPU threads for RF and XGBoost",
        min_value=1,
        max_value=available_cpu,
        value=1,
        step=1
    )


    st.divider()


    # ========================================================
    # MODEL PARAMETER
    # ========================================================

    st.markdown(
        "### Model Parameters"
    )


    # SVM
    with st.expander(
        "SVM"
    ):

        svm_kernel = st.selectbox(
            "Kernel",
            [
                "rbf",
                "linear",
                "poly",
                "sigmoid"
            ]
        )


        svm_c = st.number_input(
            "C",
            min_value=0.001,
            max_value=1000.0,
            value=1.0,
            step=0.1
        )


        svm_gamma = st.selectbox(
            "Gamma",
            [
                "scale",
                "auto"
            ]
        )


        svm_cache = st.number_input(
            "Cache size (MB)",
            min_value=100,
            max_value=8000,
            value=1000,
            step=100
        )


    # Random Forest
    with st.expander(
        "Random Forest"
    ):

        rf_estimators = st.number_input(
            "Number of trees",
            min_value=10,
            max_value=2000,
            value=100,
            step=10
        )


        rf_max_depth = st.number_input(
            "Max depth (0 = unlimited)",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )


    # XGBoost
    with st.expander(
        "XGBoost"
    ):

        xgb_estimators = st.number_input(
            "Number of estimators",
            min_value=10,
            max_value=2000,
            value=100,
            step=10
        )


        xgb_max_depth = st.number_input(
            "Max depth",
            min_value=1,
            max_value=30,
            value=6,
            step=1
        )


        xgb_learning_rate = st.number_input(
            "Learning rate",
            min_value=0.001,
            max_value=1.0,
            value=0.1,
            step=0.01
        )


        xgb_subsample = st.slider(
            "Subsample",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1
        )


        xgb_colsample = st.slider(
            "Column sample by tree",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1
        )


    # Naive Bayes
    with st.expander(
        "Naive Bayes"
    ):

        nb_var_smoothing = st.number_input(
            "Var smoothing",
            min_value=1e-12,
            max_value=1e-3,
            value=1e-9,
            format="%.2e"
        )


    # CNN
    with st.expander(
        "CNN"
    ):

        cnn_epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=200,
            value=10,
            step=1
        )


        cnn_batch_size = st.selectbox(
            "Batch size",
            [
                16,
                32,
                64,
                128,
                256,
                512
            ],
            index=2
        )


        cnn_filters_1 = st.number_input(
            "Conv1D filters 1",
            min_value=4,
            max_value=512,
            value=32,
            step=4
        )


        cnn_filters_2 = st.number_input(
            "Conv1D filters 2",
            min_value=4,
            max_value=512,
            value=64,
            step=4
        )


        cnn_dense_units = st.number_input(
            "Dense units",
            min_value=8,
            max_value=1024,
            value=64,
            step=8
        )


        cnn_dropout = st.slider(
            "Dropout",
            min_value=0.0,
            max_value=0.8,
            value=0.0,
            step=0.05
        )


        cnn_learning_rate = st.number_input(
            "Learning rate CNN",
            min_value=0.00001,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.5f"
        )


    st.divider()


    run_button = st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True
    )


    clear_button = st.button(
        "Clear Results",
        use_container_width=True
    )


# ============================================================
# 30. CLEAR
# ============================================================

if clear_button:

    st.session_state.raw_results = None

    st.session_state.summary_results = None

    st.session_state.config = None

    st.session_state.total_wall_time = None

    st.rerun()


# ============================================================
# 31. MAIN HEADER
# ============================================================

st.markdown(
    "# Machine Learning Runtime Growth Analysis"
)


st.write(
    "Analisis empiris pertumbuhan waktu pelatihan "
    "SVM, CNN, Random Forest, XGBoost, dan Naive Bayes "
    "berdasarkan perubahan ukuran data training."
)


st.divider()


# ============================================================
# 32. DATASET INFO
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Features",
        len(df.columns) - 1
    )


with col3:

    st.metric(
        "Target",
        TARGET_COLUMN
    )


with col4:

    st.metric(
        "Classes",
        df[
            TARGET_COLUMN
        ].nunique()
    )


target_distribution = (
    df[
        TARGET_COLUMN
    ]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


st.caption(
    "Target distribution: "
    +
    " | ".join(
        [
            f"Class {index}: {value:.2f}%"
            for index, value
            in target_distribution.items()
        ]
    )
)


# ============================================================
# 33. PARAMETER DICTIONARY
# ============================================================

params = {

    "cpu_threads": int(
        cpu_threads
    ),

    # SVM
    "svm_kernel": svm_kernel,
    "svm_c": float(
        svm_c
    ),
    "svm_gamma": svm_gamma,
    "svm_cache": float(
        svm_cache
    ),

    # RF
    "rf_estimators": int(
        rf_estimators
    ),
    "rf_max_depth": int(
        rf_max_depth
    ),

    # XGB
    "xgb_estimators": int(
        xgb_estimators
    ),
    "xgb_max_depth": int(
        xgb_max_depth
    ),
    "xgb_learning_rate": float(
        xgb_learning_rate
    ),
    "xgb_subsample": float(
        xgb_subsample
    ),
    "xgb_colsample": float(
        xgb_colsample
    ),

    # NB
    "nb_var_smoothing": float(
        nb_var_smoothing
    ),

    # CNN
    "cnn_epochs": int(
        cnn_epochs
    ),
    "cnn_batch_size": int(
        cnn_batch_size
    ),
    "cnn_filters_1": int(
        cnn_filters_1
    ),
    "cnn_filters_2": int(
        cnn_filters_2
    ),
    "cnn_dense_units": int(
        cnn_dense_units
    ),
    "cnn_dropout": float(
        cnn_dropout
    ),
    "cnn_learning_rate": float(
        cnn_learning_rate
    )
}


# ============================================================
# 34. RUN EXPERIMENT
# ============================================================

if run_button:

    # ========================================================
    # VALIDASI
    # ========================================================

    if len(
        selected_algorithms
    ) == 0:

        st.error(
            "Pilih minimal satu algoritma."
        )

        st.stop()


    if (
        "CNN"
        in
        selected_algorithms
        and
        not TF_AVAILABLE
    ):

        st.error(
            "CNN dipilih tetapi TensorFlow belum tersedia."
        )

        st.stop()


    if (
        "XGBoost"
        in
        selected_algorithms
        and
        not XGB_AVAILABLE
    ):

        st.error(
            "XGBoost dipilih tetapi library xgboost belum tersedia."
        )

        st.stop()


    try:

        data_sizes = parse_data_sizes(
            data_sizes_text
        )

    except ValueError as error:

        st.error(
            str(error)
        )

        st.stop()


    # ========================================================
    # X / Y
    # ========================================================

    X = df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )


    y = (
        df[
            TARGET_COLUMN
        ]
        .astype(int)
    )


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_pool, X_test, y_pool, y_test = (
        train_test_split(
            X,
            y,
            test_size=int(
                test_size
            ),
            random_state=int(
                random_seed
            ),
            stratify=y
        )
    )


    X_pool = (
        X_pool
        .reset_index(drop=True)
    )


    y_pool = (
        y_pool
        .reset_index(drop=True)
    )


    X_test = (
        X_test
        .reset_index(drop=True)
    )


    y_test = (
        y_test
        .reset_index(drop=True)
    )


    # ========================================================
    # BALANCED NESTED POOL
    # ========================================================

    X_pool_balanced, y_pool_balanced = (
        create_balanced_nested_pool(
            X_pool,
            y_pool,
            int(
                random_seed
            )
        )
    )


    available_training_size = len(
        X_pool_balanced
    )


    if max(
        data_sizes
    ) > available_training_size:

        st.error(
            f"N maksimum = {max(data_sizes):,}, "
            f"tetapi training pool hanya "
            f"{available_training_size:,} data."
        )

        st.stop()


    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "dataset": "diabetes.csv",

        "target": TARGET_COLUMN,

        "rows": int(
            len(df)
        ),

        "features": int(
            X.shape[1]
        ),

        "data_sizes": [
            int(value)
            for value
            in data_sizes
        ],

        "trials": int(
            trials
        ),

        "test_size": int(
            test_size
        ),

        "random_seed": int(
            random_seed
        ),

        "algorithms": (
            selected_algorithms
        ),

        "standard_scaler": bool(
            use_scaler
        ),

        "warmup": bool(
            use_warmup
        ),

        "warmup_size": int(
            warmup_size
        ),

        "parameters": params,

        "timestamp": (
            datetime.now()
            .isoformat()
        )
    }


    # ========================================================
    # PROGRESS UI
    # ========================================================

    st.divider()

    st.markdown(
        "## Experiment Progress"
    )


    status_box = st.empty()

    detail_box = st.empty()

    progress_bar = st.progress(
        0.0
    )


    results = []


    # ========================================================
    # WARM-UP
    # ========================================================

    if use_warmup:

        status_box.info(
            "Running warm-up before benchmark."
        )


        actual_warmup_size = min(
            int(
                warmup_size
            ),
            len(
                X_pool_balanced
            )
        )


        X_warm = (
            X_pool_balanced
            .iloc[
                :actual_warmup_size
            ]
            .copy()
        )


        y_warm = (
            y_pool_balanced
            .iloc[
                :actual_warmup_size
            ]
            .copy()
        )


        if use_scaler:

            warm_scaler = (
                StandardScaler()
            )


            X_warm_processed = (
                warm_scaler
                .fit_transform(
                    X_warm
                )
            )

        else:

            X_warm_processed = (
                X_warm
                .to_numpy(
                    dtype=np.float32
                )
            )


        X_warm_cnn = (
            X_warm_processed
            .reshape(
                X_warm_processed.shape[0],
                X_warm_processed.shape[1],
                1
            )
        )


        for algorithm in selected_algorithms:

            detail_box.write(
                f"Warm-up: {algorithm}"
            )


            try:

                warm_up_algorithm(
                    algorithm=algorithm,
                    X_train=X_warm_processed,
                    y_train=y_warm,
                    X_train_cnn=X_warm_cnn,
                    params=params,
                    seed=int(
                        random_seed
                    )
                )

            except Exception as error:

                st.warning(
                    f"Warm-up {algorithm} gagal: {error}"
                )


        detail_box.empty()


    # ========================================================
    # JOB COUNT
    # ========================================================

    total_jobs = (
        len(
            data_sizes
        )
        *
        int(
            trials
        )
        *
        len(
            selected_algorithms
        )
    )


    completed_jobs = 0


    experiment_start = (
        time.perf_counter()
    )


    # ========================================================
    # LOOP N
    # ========================================================

    for n in data_sizes:

        # Nested subset
        X_n = (
            X_pool_balanced
            .iloc[:n]
            .copy()
        )


        y_n = (
            y_pool_balanced
            .iloc[:n]
            .copy()
        )


        # ====================================================
        # PREPROCESSING DI LUAR TIMER
        # ====================================================

        if use_scaler:

            scaler = (
                StandardScaler()
            )


            X_train_processed = (
                scaler
                .fit_transform(
                    X_n
                )
            )


            X_test_processed = (
                scaler
                .transform(
                    X_test
                )
            )

        else:

            X_train_processed = (
                X_n
                .to_numpy(
                    dtype=np.float32
                )
            )


            X_test_processed = (
                X_test
                .to_numpy(
                    dtype=np.float32
                )
            )


        # CNN format
        X_train_cnn = (
            X_train_processed
            .reshape(
                X_train_processed.shape[0],
                X_train_processed.shape[1],
                1
            )
        )


        X_test_cnn = (
            X_test_processed
            .reshape(
                X_test_processed.shape[0],
                X_test_processed.shape[1],
                1
            )
        )


        # ====================================================
        # LOOP TRIAL
        # ====================================================

        for trial in range(
            1,
            int(
                trials
            ) + 1
        ):


            # ================================================
            # LOOP ALGORITHM
            # ================================================

            for algorithm in selected_algorithms:

                status_box.info(
                    f"N = {n:,} | "
                    f"Trial = {trial}/{trials} | "
                    f"Algorithm = {algorithm}"
                )


                detail_box.write(
                    f"Completed {completed_jobs} "
                    f"of {total_jobs} jobs."
                )


                trial_seed = (
                    int(
                        random_seed
                    )
                    +
                    trial
                    -
                    1
                )


                try:

                    training_time, accuracy = (
                        train_and_measure(
                            algorithm=algorithm,
                            X_train=X_train_processed,
                            y_train=y_n,
                            X_test=X_test_processed,
                            y_test=y_test,
                            X_train_cnn=X_train_cnn,
                            X_test_cnn=X_test_cnn,
                            params=params,
                            seed=trial_seed
                        )
                    )


                    results.append({

                        "N": int(
                            n
                        ),

                        "Algorithm": algorithm,

                        "Trial": int(
                            trial
                        ),

                        "Training_Time": float(
                            training_time
                        ),

                        "Accuracy": float(
                            accuracy
                        ),

                        "Status": "Success",

                        "Error": ""
                    })


                except Exception as error:

                    results.append({

                        "N": int(
                            n
                        ),

                        "Algorithm": algorithm,

                        "Trial": int(
                            trial
                        ),

                        "Training_Time": np.nan,

                        "Accuracy": np.nan,

                        "Status": "Failed",

                        "Error": str(
                            error
                        )
                    })


                completed_jobs += 1


                progress_bar.progress(
                    min(
                        completed_jobs
                        /
                        total_jobs,
                        1.0
                    )
                )


                # Partial save
                pd.DataFrame(
                    results
                ).to_csv(
                    RESULTS_DIR
                    /
                    "raw_results_partial.csv",
                    index=False
                )


    # ========================================================
    # FINISH
    # ========================================================

    experiment_end = (
        time.perf_counter()
    )


    wall_time = (
        experiment_end
        -
        experiment_start
    )


    raw_df = pd.DataFrame(
        results
    )


    summary_df = create_summary(
        raw_df
    )


    growth_df = create_growth_analysis(
        summary_df
    )


    save_results(
        raw_df,
        summary_df,
        growth_df,
        config
    )


    st.session_state.raw_results = (
        raw_df
    )


    st.session_state.summary_results = (
        summary_df
    )


    st.session_state.config = (
        config
    )


    st.session_state.total_wall_time = (
        wall_time
    )


    progress_bar.progress(
        1.0
    )


    status_box.success(
        "Experiment completed."
    )


    detail_box.write(
        f"Total wall time: "
        f"{wall_time:.2f} seconds."
    )


# ============================================================
# 35. DISPLAY RESULTS
# ============================================================

if (
    st.session_state.raw_results
    is not None
):

    raw_df = (
        st.session_state
        .raw_results
        .copy()
    )


    summary_df = (
        st.session_state
        .summary_results
        .copy()
    )


    config = (
        st.session_state
        .config
    )


    growth_df = (
        create_growth_analysis(
            summary_df
        )
    )


    st.divider()


    st.markdown(
        "## Benchmark Results"
    )


    successful_runs = raw_df[
        raw_df["Status"]
        ==
        "Success"
    ]


    failed_runs = raw_df[
        raw_df["Status"]
        ==
        "Failed"
    ]


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Successful Runs",
            len(
                successful_runs
            )
        )


    with metric2:

        st.metric(
            "Failed Runs",
            len(
                failed_runs
            )
        )


    with metric3:

        if not summary_df.empty:

            fastest = (
                summary_df
                .loc[
                    summary_df[
                        "Mean_Training_Time"
                    ]
                    .idxmin()
                ]
            )


            st.metric(
                "Fastest Observed",
                fastest[
                    "Algorithm"
                ]
            )

        else:

            st.metric(
                "Fastest Observed",
                "-"
            )


    with metric4:

        if (
            st.session_state
            .total_wall_time
            is not None
        ):

            st.metric(
                "Total Wall Time",
                f"{st.session_state.total_wall_time:.2f} s"
            )

        else:

            st.metric(
                "Total Wall Time",
                "-"
            )


    # ========================================================
    # TABS
    # ========================================================

    (
        tab_time,
        tab_growth,
        tab_scatter,
        tab_accuracy,
        tab_raw,
        tab_info
    ) = st.tabs(
        [
            "Training Time",
            "Growth Plot",
            "Scatter Plot",
            "Accuracy",
            "Raw Results",
            "Experiment Info"
        ]
    )


    # ========================================================
    # TRAINING TIME TAB
    # ========================================================

    with tab_time:

        st.markdown(
            "### Mean Training Time"
        )


        if summary_df.empty:

            st.warning(
                "Tidak ada hasil yang berhasil."
            )

        else:

            time_table = (
                summary_df
                .pivot(
                    index="N",
                    columns="Algorithm",
                    values="Mean_Training_Time"
                )
                .reset_index()
            )


            format_dict = {

                column: "{:.6f}"

                for column
                in time_table.columns

                if column != "N"
            }


            st.dataframe(
                time_table.style.format(
                    format_dict
                ),
                use_container_width=True,
                hide_index=True
            )


            st.caption(
                "Training time menggunakan satuan detik. "
                "Jika trial lebih dari satu, nilai pada tabel "
                "merupakan rata-rata."
            )


            st.markdown(
                "### Detailed Summary"
            )


            detailed = (
                summary_df
                .copy()
            )


            detailed[
                "Mean_Accuracy"
            ] = (
                detailed[
                    "Mean_Accuracy"
                ]
                *
                100
            )


            st.dataframe(
                detailed.style.format(
                    {
                        "Mean_Training_Time": "{:.6f}",
                        "Median_Training_Time": "{:.6f}",
                        "Std_Training_Time": "{:.6f}",
                        "Min_Training_Time": "{:.6f}",
                        "Max_Training_Time": "{:.6f}",
                        "Mean_Accuracy": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # GROWTH ANALYSIS
            # =================================================

            st.markdown(
                "### Growth Analysis"
            )


            if not growth_df.empty:

                st.dataframe(
                    growth_df.style.format(
                        {
                            "Mean_Training_Time": "{:.6f}",
                            "Previous_N": "{:.0f}",
                            "N_Factor": "{:.3f}",
                            "Time_Growth_Factor": "{:.3f}",
                            "Empirical_Exponent": "{:.3f}"
                        },
                        na_rep="-"
                    ),
                    use_container_width=True,
                    hide_index=True
                )


                st.caption(
                    "Empirical Exponent hanya digunakan "
                    "sebagai indikator pola pertumbuhan waktu "
                    "pada hasil eksperimen, bukan sebagai klaim "
                    "kompleksitas teoretis."
                )


            # =================================================
            # DOWNLOAD
            # =================================================

            st.markdown(
                "### Download"
            )


            down1, down2, down3 = (
                st.columns(3)
            )


            with down1:

                st.download_button(
                    "Download Summary CSV",
                    dataframe_to_csv_bytes(
                        summary_df
                    ),
                    file_name="summary_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )


            with down2:

                st.download_button(
                    "Download Training Time CSV",
                    dataframe_to_csv_bytes(
                        time_table
                    ),
                    file_name="training_time_table.csv",
                    mime="text/csv",
                    use_container_width=True
                )


            with down3:

                if not growth_df.empty:

                    st.download_button(
                        "Download Growth CSV",
                        dataframe_to_csv_bytes(
                            growth_df
                        ),
                        file_name="growth_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


    # ========================================================
    # GROWTH PLOT TAB
    # ========================================================

    with tab_growth:

        st.markdown(
            "### Training Time Growth Plot"
        )


        col_plot1, col_plot2 = (
            st.columns(2)
        )


        with col_plot1:

            growth_log = st.checkbox(
                "Use logarithmic Y-axis",
                value=False,
                key="growth_log"
            )


        with col_plot2:

            growth_std = st.checkbox(
                "Show standard deviation",
                value=True,
                key="growth_std"
            )


        if not summary_df.empty:

            growth_fig = (
                create_growth_figure(
                    summary_df,
                    logarithmic_y=growth_log,
                    show_error=growth_std
                )
            )


            st.pyplot(
                growth_fig,
                use_container_width=True
            )


            growth_jpg = (
                figure_to_jpg_bytes(
                    growth_fig
                )
            )


            st.download_button(
                "Download Growth Plot JPG",
                growth_jpg,
                file_name="training_time_growth.jpg",
                mime="image/jpeg"
            )


            plt.close(
                growth_fig
            )


            st.caption(
                "Sumbu X menunjukkan jumlah data training. "
                "Sumbu Y menunjukkan rata-rata training time."
            )


    # ========================================================
    # SCATTER PLOT TAB
    # ========================================================

    with tab_scatter:

        st.markdown(
            "### Scatter Plot Training Time"
        )


        scatter_col1, scatter_col2, scatter_col3 = (
            st.columns(3)
        )


        with scatter_col1:

            scatter_connect = st.checkbox(
                "Connect data points",
                value=True,
                key="scatter_connect"
            )


        with scatter_col2:

            scatter_log = st.checkbox(
                "Use logarithmic Y-axis",
                value=False,
                key="scatter_log"
            )


        with scatter_col3:

            scatter_trials = st.checkbox(
                "Show individual trial points",
                value=False,
                key="scatter_trials"
            )


        if not summary_df.empty:

            scatter_fig = (
                create_scatter_figure(
                    summary=summary_df,
                    connect_points=scatter_connect,
                    logarithmic_y=scatter_log,
                    show_trial_points=scatter_trials,
                    raw_df=raw_df
                )
            )


            st.pyplot(
                scatter_fig,
                use_container_width=True
            )


            scatter_jpg = (
                figure_to_jpg_bytes(
                    scatter_fig
                )
            )


            st.download_button(
                "Download Scatter Plot JPG",
                scatter_jpg,
                file_name="training_time_scatter.jpg",
                mime="image/jpeg"
            )


            plt.close(
                scatter_fig
            )


            st.caption(
                "Titik utama menunjukkan rata-rata training time "
                "untuk setiap ukuran N. Jika individual trial points "
                "diaktifkan, titik transparan menunjukkan hasil "
                "masing-masing trial."
            )


    # ========================================================
    # ACCURACY TAB
    # ========================================================

    with tab_accuracy:

        st.markdown(
            "### Mean Accuracy"
        )


        if not summary_df.empty:

            accuracy_table = (
                summary_df
                .pivot(
                    index="N",
                    columns="Algorithm",
                    values="Mean_Accuracy"
                )
                .reset_index()
            )


            accuracy_percent = (
                accuracy_table
                .copy()
            )


            for column in accuracy_percent.columns:

                if column != "N":

                    accuracy_percent[
                        column
                    ] = (
                        accuracy_percent[
                            column
                        ]
                        *
                        100
                    )


            acc_format = {

                column: "{:.2f}%"

                for column
                in accuracy_percent.columns

                if column != "N"
            }


            st.dataframe(
                accuracy_percent.style.format(
                    acc_format
                ),
                use_container_width=True,
                hide_index=True
            )


            accuracy_fig = (
                create_accuracy_figure(
                    summary_df
                )
            )


            st.pyplot(
                accuracy_fig,
                use_container_width=True
            )


            accuracy_jpg = (
                figure_to_jpg_bytes(
                    accuracy_fig
                )
            )


            st.download_button(
                "Download Accuracy Plot JPG",
                accuracy_jpg,
                file_name="accuracy_plot.jpg",
                mime="image/jpeg"
            )


            plt.close(
                accuracy_fig
            )


            st.caption(
                "Accuracy digunakan sebagai informasi pendukung. "
                "Fokus utama eksperimen adalah pertumbuhan training time."
            )


    # ========================================================
    # RAW RESULT TAB
    # ========================================================

    with tab_raw:

        st.markdown(
            "### Raw Trial Results"
        )


        raw_display = (
            raw_df
            .copy()
        )


        raw_display[
            "Accuracy"
        ] = (
            raw_display[
                "Accuracy"
            ]
            *
            100
        )


        st.dataframe(
            raw_display.style.format(
                {
                    "Training_Time": "{:.6f}",
                    "Accuracy": "{:.2f}%"
                },
                na_rep="-"
            ),
            use_container_width=True,
            hide_index=True
        )


        st.download_button(
            "Download Raw Results CSV",
            dataframe_to_csv_bytes(
                raw_df
            ),
            file_name="raw_results.csv",
            mime="text/csv"
        )


        if len(
            failed_runs
        ) > 0:

            st.markdown(
                "### Failed Runs"
            )


            st.dataframe(
                failed_runs[
                    [
                        "N",
                        "Algorithm",
                        "Trial",
                        "Error"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # INFO TAB
    # ========================================================

    with tab_info:

        st.markdown(
            "### Experimental Design"
        )


        if config is not None:

            info1, info2 = (
                st.columns(2)
            )


            with info1:

                st.write(
                    f"Dataset: {config['dataset']}"
                )

                st.write(
                    f"Target: {config['target']}"
                )

                st.write(
                    f"Rows: {config['rows']:,}"
                )

                st.write(
                    f"Features: {config['features']}"
                )

                st.write(
                    f"Test set: {config['test_size']:,}"
                )

                st.write(
                    f"Random seed: {config['random_seed']}"
                )


            with info2:

                st.write(
                    "Training sizes: "
                    +
                    ", ".join(
                        [
                            f"{n:,}"
                            for n
                            in config[
                                "data_sizes"
                            ]
                        ]
                    )
                )


                st.write(
                    f"Trials: {config['trials']}"
                )


                st.write(
                    "Algorithms: "
                    +
                    ", ".join(
                        config[
                            "algorithms"
                        ]
                    )
                )


                st.write(
                    "StandardScaler: "
                    +
                    (
                        "Enabled"
                        if config[
                            "standard_scaler"
                        ]
                        else
                        "Disabled"
                    )
                )


                st.write(
                    "Warm-up: "
                    +
                    (
                        "Enabled"
                        if config[
                            "warmup"
                        ]
                        else
                        "Disabled"
                    )
                )


            st.markdown(
                "### Timing Method"
            )


            st.write(
                "Training time diukur menggunakan "
                "`time.perf_counter()`."
            )


            st.write(
                "Timer dimulai tepat sebelum `model.fit()` "
                "dan berhenti setelah training selesai."
            )


            st.write(
                "Load dataset, sampling, preprocessing, "
                "StandardScaler, prediction, dan evaluasi accuracy "
                "tidak dihitung sebagai training time."
            )


            st.write(
                "Training subset dibuat nested sehingga subset "
                "N kecil merupakan bagian dari subset N yang lebih besar."
            )


            st.write(
                "Training pool dibuat balanced antara kelas 0 dan kelas 1."
            )


            st.write(
                "Warm-up dilakukan sebelum benchmark dan tidak "
                "dimasukkan ke dalam hasil pengukuran."
            )


            st.write(
                "CNN dijalankan menggunakan CPU agar kondisi hardware "
                "lebih konsisten dengan algoritma lainnya."
            )


            # =================================================
            # PARAMETERS
            # =================================================

            st.markdown(
                "### Model Parameters"
            )


            parameter_rows = []


            for key, value in (
                config[
                    "parameters"
                ]
                .items()
            ):

                parameter_rows.append(
                    {
                        "Parameter": key,
                        "Value": value
                    }
                )


            parameter_df = pd.DataFrame(
                parameter_rows
            )


            st.dataframe(
                parameter_df,
                use_container_width=True,
                hide_index=True
            )


            config_json = json.dumps(
                config,
                indent=4,
                ensure_ascii=False
            )


            st.download_button(
                "Download Experiment Configuration",
                config_json,
                file_name="experiment_config.json",
                mime="application/json"
            )


# ============================================================
# 36. READY STATE
# ============================================================

else:

    st.divider()


    st.markdown(
        "## Ready to Run"
    )


    st.write(
        "Atur konfigurasi eksperimen melalui sidebar, "
        "kemudian pilih Run Experiment."
    )


    st.write(
        "Hasil eksperimen akan disimpan ke folder results "
        "dan tetap tampil pada halaman sampai eksperimen baru "
        "dijalankan atau Clear Results dipilih."
    )