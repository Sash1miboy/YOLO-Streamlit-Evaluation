import streamlit as st
import pandas as pd
from pathlib import Path
from pages.tabel_page import table_sections
from pages.chart_page import chart_sections

base_dir = Path(__file__).resolve().parent
overall_path = base_dir / "data" / "Testing-Results-Last-Final-Verdict.csv"
all_class_path = base_dir / "data" / "yolo_metrics_detailed.csv"
logo_path = base_dir / "assets" / "binus.png"

st.set_page_config(page_title="Analisis Performa YOLO", layout="wide")

overall_data = pd.read_csv(overall_path)
all_class_data = pd.read_csv(all_class_path)

st.subheader("Tabel Raw Data Performa Arsitektur Model")
st.dataframe(overall_data)

st.set_page_config(
    page_title="Analisis Performa YOLO",
    layout="wide",
    initial_sidebar_state="expanded",  
    menu_items=None  
)

st.sidebar.image(str(logo_path), width=250)

page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["📊 Tabel", "📈 Chart", "🧪 Demo"]
)

if page == "📊 Tabel":
    table_sections(overall_data, all_class_data)

elif page == "📈 Chart":
    chart_sections(overall_data, all_class_data)

elif page == "🧪 Demo":
    st.write("Ini halaman Demo")
