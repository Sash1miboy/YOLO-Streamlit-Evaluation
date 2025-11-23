import streamlit as st
import pandas as pd
import numpy
import os
import plotly.express as px
import plotly.graph_objects as go

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "data", "Testing Results-Final-Verdict.csv")

st.set_page_config(page_title="Analisis Performa YOLO", layout="wide")

df = pd.read_csv(csv_path)

st.subheader("Tabel Mentah Performa Arsitektur Model")
st.write(df)

st.subheader("Tabel Akurasi Model")

accuracy_data = df[
    ["Model", "Version", "Size", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]
].copy()

selected_column = st.selectbox("Pilih kolom untuk sorting:", accuracy_data.columns)

sort_type = st.selectbox(
    "Mode Sort:", ["Terkecil → Terbesar", "Terbesar → Terkecil", "A → Z", "Z → A"]
)

match sort_type:
    case "Terkecil → Terbesar":
        sorted_df = accuracy_data.sort_values(by=selected_column, ascending=True)

    case "Terbesar → Terkecil":
        sorted_df = accuracy_data.sort_values(by=selected_column, ascending=False)

    case "A → Z":
        sorted_df = accuracy_data.sort_values(
            by=selected_column, ascending=True, key=lambda x: x.astype(str)
        )

    case "Z → A":
        sorted_df = accuracy_data.sort_values(
            by=selected_column, ascending=False, key=lambda x: x.astype(str)
        )

numerical_cols = ["Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]

df_display = sorted_df.copy()

for col in numerical_cols:
    max_val = df_display[col].max()

    df_display[col] = df_display[col].apply(
        lambda x: f"<b>{x}</b>" if x == max_val else x
    )

fig_table = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=list(df_display.columns),
                fill_color="lightgray",
                align="center",
                font=dict(size=15, color="black"),
                height=40,
            ),
            cells=dict(
                values=[df_display[col] for col in df_display.columns],
                fill_color="white",
                font=dict(size=12, color="black"),
                height=35,
                format=["html"] * len(df_display.columns),
            ),
        )
    ]
)

fig_table.update_layout(height=800)

st.plotly_chart(fig_table, width="stretch")

st.subheader("Tabel Efficiency Model")

efficiency_data = df[
    [
        "Model",
        "Version",
        "Size",
        "Preprocessing (ms)",
        "Inference (ms)",
        "Postprocessing (ms)",
        "FPS",
        "Training Time",
    ]
].copy()

selected_column = st.selectbox(
    "Pilih kolom untuk sorting:", efficiency_data.columns, key="sort_col_efficiency"
)

sort_type = st.selectbox(
    "Mode Sort:",
    ["Terkecil → Terbesar", "Terbesar → Terkecil", "A → Z", "Z → A"],
    key="select_efficiency",
)

match sort_type:
    case "Terkecil → Terbesar":
        sorted_df = efficiency_data.sort_values(by=selected_column, ascending=True)

    case "Terbesar → Terkecil":
        sorted_df = efficiency_data.sort_values(by=selected_column, ascending=False)

    case "A → Z":
        sorted_df = efficiency_data.sort_values(
            by=selected_column, ascending=True, key=lambda x: x.astype(str)
        )

    case "Z → A":
        sorted_df = efficiency_data.sort_values(
            by=selected_column, ascending=False, key=lambda x: x.astype(str)
        )


df_display = sorted_df.copy()

max_numerical_cols = ["FPS"]

for col in max_numerical_cols:
    max_val = df_display[col].max()

    df_display[col] = df_display[col].apply(
        lambda x: f"<b>{x}</b>" if x == max_val else x
    )

min_numerical_cols = ["Preprocessing (ms)", "Inference (ms)", "Postprocessing (ms)"]

for col in min_numerical_cols:
    min_val = df_display[col].min()

    df_display[col] = df_display[col].apply(
        lambda x: f"<b>{x}</b>" if x == min_val else x
    )

fig_table = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=list(df_display.columns),
                fill_color="lightgray",
                align="center",
                font=dict(size=15, color="black"),
                height=40,
            ),
            cells=dict(
                values=[df_display[col] for col in df_display.columns],
                fill_color="white",
                font=dict(size=12, color="black"),
                height=35,
                format=["html"] * len(df_display.columns),
            ),
        )
    ]
)

fig_table.update_layout(height=800)

st.plotly_chart(fig_table, width="stretch")

st.subheader("Chart Perbandingan Akurasi Arsitekur Model: mAP50 & mAP50-95")

df_melted = df.melt(
    id_vars=["Model"],
    value_vars=["mAP50", "mAP50-95"],
    var_name="Metrik",
    value_name="Nilai Score",
)

map_comparison_chart = px.bar(
    df_melted,
    x="Model",
    y="Nilai Score",
    color="Metrik",
    barmode="group",
    text_auto=".3f",
    color_discrete_map={"mAP50": "red", "mAP50-95": "blue"},
    height=400,
)

map_comparison_chart.update_layout(xaxis_title="Model", yaxis_title="Nilai mAP")
st.plotly_chart(map_comparison_chart, width="stretch")
