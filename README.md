This is modified to be used in ImageTrans.

Installation:

1. Install Python (version >= 3.8)
2. Download the project: https://github.com/xulihang/manga-image-translator/archive/refs/heads/main.zip
3. Install dependencies: `pip install -r requirements.txt`
4. Download the model files into the project's folder: [detect.ckpt
](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/detect.ckpt), [ocr.ckpt](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/ocr.ckpt), [inpainting.ckpt](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/inpainting.ckpt)
5. Download and unzip the OCR CTC model into the same folder: [ocr-ctc.zip](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip). This OCR model supports Korean while the OCR model in the previous step only supports English, Chinese and Japanese. Its speed is also higher.
6. Download the [OCR 48px model](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt) and its [dictionary](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt) into the same folder. This OCR model has a higher accuracy.
7. Run the server: `python server.py`


Test pages:

* <http://127.0.0.1:8080/ocr.html>
* <http://127.0.0.1:8080/getmask.html>
* <http://127.0.0.1:8080/remove.html>

## How to Enable GPU

To use GPU for inference, you need to do the following:

1. Install a Pytorch version with GPU support according to [this](https://pytorch.org/get-started/locally/#start-locally).
2. Create a new file named `use_cuda` in the root of the project to enable CUDA or `use_mps` to enable the GPU for Mac.

## Use with ImageTrans

You can find the additional guide here: https://github.com/xulihang/ImageTrans_plugins/tree/master/mangaTranslatorOCR
