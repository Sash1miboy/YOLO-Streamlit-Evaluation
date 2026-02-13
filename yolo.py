import streamlit as st

from reports.chart_page import chart_sections
from reports.demo_page import demo_page
from reports.tabel_page import table_sections
from utils.download_data import download_to_excel
from utils.load_data import load_logo, load_main_data

st.set_page_config(page_title="Analisis Performa YOLO", layout="wide")

st.set_page_config(
    page_title="Analisis Performa YOLO",
    layout="wide",
    initial_sidebar_state="expanded",
)

logo_path = load_logo()

st.sidebar.image(str(logo_path), width=250)

dataset = st.sidebar.selectbox(
    "Pilih Dataset:", ["Human Face Emotions Computer Vision Model", "FER2013"]
)

# page = st.sidebar.selectbox("Pilih Halaman:", ["📊 Tabel", "📈 Chart", "🧪 Demo"])
page = st.sidebar.selectbox("Pilih Halaman:", ["📊 Tabel", "📈 Chart", "🧪 Demo"])

if dataset == "Human Face Emotions Computer Vision Model":
    dataset = "human-face-emotion-computer-vision-model"
    bestpt = "best-YOLOv11s-Roboflow.pt"
elif dataset == "FER2013":
    dataset = "fer-2013"
    bestpt = "Unavaible"

overall_data, all_class_data = load_main_data(dataset)

if overall_data is None or all_class_data is None:
    st.error("Gagal memuat data. Pastikan file CSV tersedia.")
    st.stop()

st.subheader("Tabel Raw Data Performa Arsitektur Model")
st.dataframe(overall_data)

excel_data = download_to_excel(overall_data)

st.download_button(
    label="📥 Download Excel",
    data=excel_data,
    file_name="raw_yolo_comparison_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


if page == "📊 Tabel":
    table_sections(overall_data, all_class_data, dataset)

elif page == "📈 Chart":
    chart_sections(overall_data, all_class_data, dataset)
elif page == "🧪 Demo":
    demo_page(bestpt)

# elif page == "🧪 Demo":
#     st.write("Ini halaman Demo")
