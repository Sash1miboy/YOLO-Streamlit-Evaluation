import streamlit as st
import pandas as pd
from charts.yolo_charts import (
    map_comparison_chart,
    latency_stacked_chart,
    tradeoff_scatter_chart,
    training_time_chart,
    robustness_heatmap,
)
from utils.filter import (
    chart_filter_options,
    filter_and_sort_training_data,
    filter_and_sort_heatmap_data,
)
from utils.time_converter import hhmmss_to_minutes


def chart_sections(overall_data: pd.DataFrame, class_data: pd.DataFrame):
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
            key=f"scatter_filter",
        )

        if selected_models:
            overall_data = overall_data[overall_data["Model"].isin(selected_models)]

    tradeoff_scatter_chart(overall_data)

    st.divider()
    training_data = overall_data.copy()
    training_data["Training Time (menit)"] = overall_data["Training Time"].apply(
        hhmmss_to_minutes
    )

    st.subheader("Waktu Pelatihan Model YOLO")
    training_data = filter_and_sort_training_data(training_data)
    training_time_chart(training_data)

    st.divider()

    st.subheader("Heat Map Ketangguhan (Robustness) Model YOLO")
    class_data, metrics = filter_and_sort_heatmap_data(class_data)
    robustness_heatmap(class_data, metrics)
