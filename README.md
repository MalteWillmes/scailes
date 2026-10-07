# sc[ai]les
Model architecture and development for Willmes et al., in review: "Identifying escaped farmed salmon from fish scales using deep learning". This deep learning image classifier can distinguish farmed from wild Atlantic salmon using scale images. The model was trained on ~90,000 fish scales from hundreds of rivers across Norway.

## Installation
Python version 3.11 was used for the development in this project.

Dependencies are managed with [poetry](https://python-poetry.org/docs/).
For detailed dependencies, refer to `pyproject.toml`.

```bash
# install poetry, if not installed yet
curl -sSL https://install.python-poetry.org | python -
# install project
poetry install
```

## Classify scales with the GUI
A small web app classifies a folder of scale images as wild or farmed. It only needs the
exported ONNX model, not the training dependencies (no PyTorch).

1. Put `scale_classifier.onnx` in the repository root (or set its path in the app's sidebar).
2. Start the app from the repository root:
```bash
uv run --no-project --with-requirements gui/requirements.txt python -m streamlit run gui/app.py
```
3. Choose a folder of images (`.tif`, `.jpg`, `.png`, `.bmp`; subfolders are included) and click **Classify**.

The app shows one row per image with its folder, `P(wild)`, `P(farmed)` and a label. The model
was trained with a sigmoid focal loss, so each output is an independent score and the two
probabilities do not have to add up to 1. A scale is labelled *Farmed* when `P(farmed)` is at or
above the threshold set in the sidebar (default 0.5); changing it relabels the table without
re-running the model. The table can be downloaded as CSV.

Images are shrunk to fit within 720x480 px, then resized to 384x512 and normalized, as in the
paper. The inference code is in `gui/inference.py`.

## Model training
The model is trained using `train.py`.

The model configuration used in the paper is found at `config/config_final.yml`. Models can be tested using `test.py`.

## Exported onnx model
The final model weights have been exported as an onnx model file. For the conversion process, please refer to:
```
notebooks/load_and_export_model.ipynb
```

The notebook also provides an example of loading of the onnx model and running model inference on images.

However, note that example expects that the initial image resizing to fit within [720, 480] pixels has been carried out (refer to the paper).
