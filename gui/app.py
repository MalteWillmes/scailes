"""sc[ai]les: classify salmon scale images as wild or farmed.

Run from the repository root:  streamlit run gui/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import inference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "scale_classifier.onnx"

st.set_page_config(page_title="sc[ai]les", layout="wide")
st.title("sc[ai]les: wild or farmed salmon scales")


@st.cache_resource
def get_session(path: str):
    return inference.load_model(Path(path))


def browse_folder() -> None:
    """Native folder dialog (the app runs on the user's own computer)."""
    import tkinter as tk
    from tkinter import filedialog

    tk_root = tk.Tk()
    tk_root.withdraw()
    tk_root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(title="Folder with scale images")
    tk_root.destroy()
    if chosen:
        st.session_state["folder"] = chosen


# --- sidebar: model and threshold --------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model file (.onnx)", str(DEFAULT_MODEL))
    threshold = st.slider(
        "Farmed threshold",
        0.0,
        1.0,
        0.5,
        0.01,
        help="A scale is called Farmed when P(farmed) is at or above this value. "
        "Changing it relabels the table without re-running the model.",
    )

if not Path(model_path).is_file():
    st.error(f"Model file not found: {model_path}")
    st.stop()

# --- input: folder of images -------------------------------------------------------
col_path, col_btn = st.columns([5, 1], vertical_alignment="bottom")
folder = col_path.text_input("Folder with scale images", key="folder")
col_btn.button("Browse…", on_click=browse_folder, use_container_width=True)

images: list[Path] = []
if folder:
    if Path(folder).is_dir():
        images = inference.list_images(Path(folder))
        st.caption(f"{len(images)} image(s) found")
    else:
        st.warning("That folder does not exist.")

if st.button("Classify", type="primary", disabled=not images):
    bar = st.progress(0.0, text="Classifying…")
    st.session_state["result"] = {
        "folder": folder,
        "df": inference.classify(
            get_session(model_path), images, progress=lambda f: bar.progress(f)
        ),
    }
    bar.empty()

# --- results -----------------------------------------------------------------------
result = st.session_state.get("result")
if result:
    df: pd.DataFrame = result["df"].copy()
    df["label"] = inference.label(df["p_farmed"], threshold)
    counts = df["label"].value_counts()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images", len(df))
    m2.metric("Wild", int(counts.get("Wild", 0)))
    m3.metric("Farmed", int(counts.get("Farmed", 0)))
    m4.metric("Errors", int(counts.get("Error", 0)))

    left, right = st.columns([3, 2])
    with left:
        event = st.dataframe(
            df[["file", "label", "p_farmed", "error"]],
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "p_farmed": st.column_config.ProgressColumn(
                    "P(farmed)", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )
        st.download_button(
            "Download CSV",
            df.assign(threshold=threshold).to_csv(index=False),
            file_name="scailes_predictions.csv",
            mime="text/csv",
        )
    with right:
        rows = event.selection.rows
        if rows:
            name = df.iloc[rows[0]]["file"]
            st.image(
                inference.load_fitted(Path(result["folder"]) / name), caption=name
            )
        else:
            st.caption("Select a row to see the image.")
