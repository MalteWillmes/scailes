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
