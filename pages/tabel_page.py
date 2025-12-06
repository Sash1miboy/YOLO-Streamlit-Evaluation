import streamlit as st
from utils.filter import table_filter
from table.yolo_tabel import accuracy_table, efficiency_table, class_emotions_table
from utils.download_data import download_to_excel


def table_sections(overall_data, all_class_data):
    accuracy_data = overall_data[
        [
            "Index",
            "Model",
            "Size",
            "Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP50-95",
        ]
    ].copy()
    sorted_accuracy_data = table_filter(accuracy_data, "accuracy")

    st.subheader("Tabel Akurasi Model")
    accuracy_table(sorted_accuracy_data)
    
    excel_data = download_to_excel(sorted_accuracy_data)
    
    st.download_button(
        label="📥 Download Excel",
        data=excel_data,
        file_name="accuracy_yolo_comparison_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    efficiency_data = overall_data[
        [
            "Index",
            "Model",
            "Size",
            "Preprocessing (ms)",
            "Inference (ms)",
            "Postprocessing (ms)",
            "Total Time (ms)",
            "FPS",
            "Training Time",
        ]
    ]
    sorted_efficiency_data = table_filter(efficiency_data, "efficiency")

    st.subheader("Tabel Efisiensi Model")
    efficiency_table(sorted_efficiency_data)
    
    excel_data = download_to_excel(sorted_efficiency_data)
    
    st.download_button(
        label="📥 Download Excel",
        data=excel_data,
        file_name="efficiency_yolo_comparison_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    model_order = {
        1: "YOLOv8n",
        2: "YOLOv8s",
        3: "YOLOv8m",
        4: "YOLOv8l",
        5: "YOLOv9t",
        6: "YOLOv9s",
        7: "YOLOv9m",
        8: "YOLOv9c",
        9: "YOLOv10n",
        10: "YOLOv10s",
        11: "YOLOv10m",
        12: "YOLOv10l",
        13: "YOLOv11n",
        14: "YOLOv11s",
        15: "YOLOv11m",
        16: "YOLOv11l",
        17: "YOLOv12n",
        18: "YOLOv12s",
        19: "YOLOv12m",
        20: "YOLOv12l",
    }

    pivot_all_class_data = all_class_data.pivot_table(
        index="Model", columns="Class", values="mAP50"
    ).reset_index()

    reverse_map = {}
    for idx, name in model_order.items():
        reverse_map[name] = idx

    pivot_all_class_data["Index"] = pivot_all_class_data["Model"].map(reverse_map)
    pivot_all_class_data = pivot_all_class_data.sort_values("Index")

    cols = []
    cols.append("Index")
    for col in pivot_all_class_data.columns:
        if col != "Index":
            cols.append(col)

    pivot_all_class_data = pivot_all_class_data[cols].reset_index(drop=True)

    sorted_all_class_data = table_filter(pivot_all_class_data, "class")

    st.subheader("Tabel mAP50 per Kelas Emosi")
    class_emotions_table(sorted_all_class_data)
    
    excel_data = download_to_excel(sorted_all_class_data)
    
    st.download_button(
        label="📥 Download Excel",
        data=excel_data,
        file_name="class_yolo_comparison_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
