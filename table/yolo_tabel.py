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

    row_height = 35
    header_height = 35
    dynamic_height = header_height + (len(display) * row_height)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display.columns),
                    fill_color="lightgray",
                    align="center",
                    font=dict(size=14, color="black"),
                    height=header_height,
                ),
                cells=dict(
                    values=[display[col] for col in display.columns],
                    fill_color="white",
                    font=dict(size=13, color="black"),
                    height=row_height,
                    format=["html"] * len(display.columns),
                ),
            )
        ]
    ).update_layout(height=dynamic_height + 50, margin=dict(l=0, r=0, t=5, b=5))

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

    row_height = 35
    header_height = 35
    dynamic_height = header_height + (len(display) * row_height)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display.columns),
                    fill_color="lightgray",
                    align="center",
                    font=dict(size=14, color="black"),
                    height=header_height,
                ),
                cells=dict(
                    values=[display[col] for col in display.columns],
                    fill_color="white",
                    font=dict(size=13, color="black"),
                    height=row_height,
                    format=["html"] * len(display.columns),
                ),
            )
        ]
    ).update_layout(height=dynamic_height + 50, margin=dict(l=0, r=0, t=5, b=5))

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

    row_height = 35
    header_height = 35
    dynamic_height = header_height + (len(display) * row_height)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display.columns),
                    fill_color="lightgray",
                    align="center",
                    font=dict(size=14, color="black"),
                    height=header_height,
                ),
                cells=dict(
                    values=[display[col] for col in display.columns],
                    fill_color="white",
                    font=dict(size=13, color="black"),
                    height=row_height,
                    format=["html"] * len(display.columns),
                ),
            )
        ]
    ).update_layout(height=dynamic_height + 50, margin=dict(l=0, r=0, t=5, b=5))

    st.plotly_chart(fig_table, width="stretch")
