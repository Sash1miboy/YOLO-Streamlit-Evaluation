from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from ultralytics import YOLO


@st.cache_data
def load_main_data(dataset: str):
    base_dir = Path(__file__).resolve().parent.parent

    overall_path = base_dir / "data" / dataset / "Testing-Results-LAST_ASLI.csv"
    all_class_path = base_dir / "data" / dataset / "yolo_metrics_detailed.csv"

    try:
        df_overall = pd.read_csv(overall_path)
        df_all_class = pd.read_csv(all_class_path)

        return df_overall, df_all_class

    except Exception as err:
        st.error(f"File data tidak ditemukan: {err}")
        return None, None


@st.cache_data
def load_logo():
    base_dir = Path(__file__).resolve().parent.parent

    logo_path = base_dir / "assets" / "binus.png"

    return logo_path


@st.cache_data
def load_training_logs(dataset: str) -> Optional[pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / dataset

    csv_files = list(data_path.glob("*_results.csv"))

    combined_data = []
    try:
        if not csv_files:
            return None

        for file_path in csv_files:
            model_name_clean = file_path.name.replace("_results.csv", "")
            curr_variant = get_variant_group(model_name_clean)

            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()

            data["Model"] = model_name_clean
            data["Variant_Group"] = curr_variant

            combined_data.append(data)

        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return None

    except Exception as err:
        st.error(f"Gagal memuat log training: {err}")
        return None


def get_variant_group(model_name: str):
    if not model_name:
        return "Lainnya"

    variant_char = model_name[-1].lower()

    if variant_char in ["n", "t"]:
        return "Nano & Tiny (n/t)"
    elif variant_char == "s":
        return "Small (s)"
    elif variant_char == "m":
        return "Medium (m)"
    elif variant_char in ["l", "c"]:
        return "Large & Compact (l/c)"
    else:
        return "Lainnya"


@st.cache_resource
def load_yolo_model(model_name: str):
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / f"{model_name}.onnx"

    model = YOLO(model_path)

    return model
