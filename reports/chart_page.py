import pandas as pd
import streamlit as st

from charts.yolo_charts import (
    latency_stacked_chart,
    map_comparison_chart,
    robustness_heatmap,
    tradeoff_scatter_chart,
    training_curve,
    training_time_chart,
)
from utils.filter import (
    chart_filter_options,
    filter_and_sort_heatmap_data,
    filter_and_sort_training_data,
)
from utils.load_data import load_training_logs
from utils.time_converter import hhmmss_to_minutes


def chart_sections(overall_data: pd.DataFrame, class_data: pd.DataFrame, dataset: str):
    st.subheader("Chart Perbandingan Akurasi Arsitekur Model YOLO: mAP50 & mAP50-95")
    mAP_data = overall_data.melt(
        id_vars=["Model"],
        value_vars=["mAP50", "mAP50-95"],
        var_name="Metrics",
        value_name="Score",
    )

    mAP_data = chart_filter_options(mAP_data, "mAP")
    map_comparison_chart(mAP_data)

    st.divider()

    st.subheader("Chart Perbandingan Latency Arsitekur Model YOLO")
    latency_data = overall_data.melt(
        id_vars="Model",
        value_vars=["Preprocessing (ms)", "Inference (ms)", "Postprocessing (ms)"],
        var_name="Metrics",
        value_name="Score",
    )

    latency_data = chart_filter_options(latency_data, "latency")

    latency_stacked_chart(latency_data)

    st.divider()

    st.subheader("Trade-off Kecepatan vs Akurasi (Scatter Plot)")
    s1, s2, s3 = st.columns(3)
    with s1:
        selected_models = st.multiselect(
            "Pilih Model yang Ditampilkan:",
            overall_data["Model"].unique(),
            key="scatter_filter",
        )

        if selected_models:
            overall_data = overall_data[overall_data["Model"].isin(selected_models)]

    tradeoff_scatter_chart(overall_data)

    st.divider()
    training_data = overall_data.copy()
    training_data["Training Time (menit)"] = overall_data["Training Time"].apply(
        hhmmss_to_minutes
    )

    st.subheader("Waktu Pelatihan Model YOLO (Menit)")
    training_data = filter_and_sort_training_data(training_data)
    training_time_chart(training_data)

    st.divider()

    st.subheader("Heat Map Ketangguhan (Robustness) Model YOLO")
    class_data, metrics = filter_and_sort_heatmap_data(class_data)
    robustness_heatmap(class_data, metrics)

    st.divider()

    training_logs = load_training_logs(dataset)
    if training_logs is None:
        st.write("Error: no training logs data!")
        st.stop()

    ignore_cols = [
        "epoch",
        "Model",
        "Variant_Group",
        "time",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    metric_options = []
    for i in training_logs.columns:
        if i not in ignore_cols:
            metric_options.append(i)

    st.subheader("Chart Kurva Pelatihan Model YOLO")

    variant_options = [
        "Nano & Tiny (n/t)",
        "Small (s)",
        "Medium (m)",
        "Large & Compact (l/c)",
    ]

    best_ver = []
    if dataset == "human-face-emotion-computer-vision-model":
        best_ver = [
            "yolov8l",
            "yolov9t",
            "yolov10n",
            "yolov11s",
            "yolov12n",
        ]
    elif dataset == "fer-2013":
        best_ver = [
            "yolov8m",
            "yolov9s",
            "yolov10s",
            "yolov11s",
            "yolov12m",
        ]
    elif dataset == "affectNet":
        best_ver = [
            "yolov8s",
            "yolov9c",
            "yolov10m",
            "yolov11m",
            "yolov12s",
        ]

    best_ver_per_family = [
        "All",
    ]

    for i in best_ver:
        best_ver_per_family.append(i)

    k1, k2, k3 = st.columns(3)

    with k1:
        filter_mode = st.selectbox(
            "Filter berdasarkan:",
            ["Variant Group", "Model Family", "Most Optimal from each Family"],
            key="filter_mode",
        )

    active_column = "Variant_Group"

    with k2:
        if filter_mode == "Variant Group":
            selected_filter = st.selectbox(
                "Pilih Variant Group:",
                variant_options,
                key="variant_group_filter",
            )
            active_column = "Variant_Group"

        elif filter_mode == "Model Family":
            training_logs["Family"] = training_logs["Model"].str.extract(
                r"(yolov\d+)", expand=False
            )

            family_options = sorted(training_logs["Family"].unique())

            selected_filter = st.selectbox(
                "Pilih Model Family:",
                family_options,
                key="family_filter",
            )
            active_column = "Family"

        else:
            selected_filter = st.selectbox(
                "Pilih Model", best_ver_per_family, key="best_ver_per_family"
            )
            active_column = "Model"

    with k3:
        selected_metric = st.selectbox(
            "Pilih Metrik:", metric_options, key="metric_selection"
        )

    filtered_data = training_logs.copy()

    if filter_mode == "Most Optimal from each Family" and selected_filter == "All":
        if dataset == "human-face-emotion-computer-vision-model":
            filtered_data = training_logs[
                training_logs[active_column].isin(
                    [
                        "yolov8l",
                        "yolov9t",
                        "yolov10n",
                        "yolov11s",
                        "yolov12n",
                    ]
                )
            ]
        elif dataset == "fer-2013":
            filtered_data = training_logs[
                training_logs[active_column].isin(
                    [
                        "yolov8m",
                        "yolov9s",
                        "yolov10s",
                        "yolov11s",
                        "yolov12m",
                    ]
                )
            ]
        elif dataset == "affectNet":
            filtered_data = training_logs[
                training_logs[active_column].isin(
                    [
                        "yolov8s",
                        "yolov9c",
                        "yolov10m",
                        "yolov11m",
                        "yolov12s",
                    ]
                )
            ]

    else:
        filtered_data = training_logs[training_logs[active_column] == selected_filter]

    st.subheader(f"Kurva Pelatihan untuk {selected_filter}")
    training_curve(filtered_data, selected_metric)
