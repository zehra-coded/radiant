# Synthetic sample X-rays

**These are NOT clinical chest X-rays. They are placeholders.**

Every file in this folder is a synthetic placeholder created for:

- UI testing (does the upload widget accept it?)
- File validation (does preprocessing reject corrupt files?)
- Pipeline wiring (does the vision module run end-to-end without errors?)
- Error handling (does the app behave correctly on invalid input?)

They are **not**:
- Real chest X-rays
- NIH ChestX-ray14 images
- Medical data of any kind
- Evidence that DenseNet-121 analyzed a real radiograph

Any pipeline output produced from these files is a **wiring test**,
not a clinical result. The UI and any logs must label results from
these files as `PIPELINE SMOKE TEST — NOT CLINICALLY MEANINGFUL`.

For meaningful inference, use an NIH ChestX-ray14 image once the
dataset is downloaded and licensed for use (see `docs/datasets.md`).