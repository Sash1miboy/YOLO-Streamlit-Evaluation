import pandas as pd
from io import BytesIO


def download_to_excel(download):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        download.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()
