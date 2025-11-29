import streamlit as st
import plotly.graph_objects as go


def accuracy_table(data):
    numerical_cols = ["Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]

    display = data.copy()

    for col in numerical_cols:
        max_val = display[col].max()
        display[col] = display[col].apply(
            lambda x: f"<b>{x}</b>" if x == max_val else x
        )

    row_colors = []
    for i in range(len(display)):
        row_colors.append("#FFFFFF" if i % 2 == 0 else "#F7F7F7")

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{col}</b>" for col in display.columns],
                    fill_color="#E5E5E5",
                    align="center",
                    font=dict(size=14, color="#333"),
                    height=40,
                    line_color="#CCCCCC",
                ),
                cells=dict(
                    values=[display[col] for col in display.columns],
                    fill_color=[row_colors],
                    align="center",
                    font=dict(size=13, color="#333"),
                    height=35,
                    line_color="#DDDDDD",
                    format=["html"] * len(display.columns),
                ),
            )
        ]
    )

    fig_table.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        height=60 + (len(display) * 35),
    )

    st.plotly_chart(fig_table, width="stretch")


def efficiency_table(data):
    numerical_cols = [
        "Preprocessing (ms)",
        "Inference (ms)",
        "Postprocessing (ms)",
        "Total Time (ms)",
        "FPS",
        "Training Time",
    ]

    display = data.copy()

    for col in numerical_cols:
        target_val = display[col].min() if col != "FPS" else display[col].max()
        display[col] = display[col].apply(
            lambda x: f"<b>{x}</b>" if x == target_val else x
        )

    row_colors = ["#FFFFFF" if i % 2 == 0 else "#F7F7F7" for i in range(len(display))]

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{col}</b>" for col in display.columns],
                    fill_color="#E5E5E5",
                    align="center",
                    font=dict(size=14, color="#333"),
                    height=40,
                    line_color="#CCCCCC",
                ),
                cells=dict(
                    values=[display[col] for col in display.columns],
                    fill_color=[row_colors],
                    align="center",
                    font=dict(size=13, color="#333"),
                    height=35,
                    line_color="#DDDDDD",
                    format=["html"] * len(display.columns),
                ),
            )
        ]
    )

    fig_table.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        height=60 + (len(display) * 35),
    )

    st.plotly_chart(fig_table, width="stretch")


def class_emotions_table(data):
    cols = [
        "anger",
        "content",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    display = data.copy()

    for col in cols:
        max_val = display[col].max()
        display[col] = display[col].apply(
            lambda x: f"<b>{x}</b>" if x == max_val else x
        )

    row_colors = ["#FFFFFF" if i % 2 == 0 else "#F7F7F7" for i in range(len(display))]

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{col}</b>" for col in display.columns],
                    fill_color="#E5E5E5",
                    align="center",
                    font=dict(size=14, color="#333"),
                    height=40,
                    line_color="#CCCCCC",
                ),
                cells=dict(
                    values=[display[col] for col in display.columns],
                    fill_color=[row_colors],
                    align="center",
                    font=dict(size=13, color="#333"),
                    height=35,
                    line_color="#DDDDDD",
                    format=["html"] * len(display.columns),
                ),
            )
        ]
    )

    fig_table.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        height=60 + (len(display) * 35),
    )

    st.plotly_chart(fig_table, width="stretch")
