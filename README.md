This is modified to be used in ImageTrans.

Installation:

1. Install Python (version >= 3.8)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python server.py`


Test pages:

* <http://127.0.0.1:8080/ocr.html>
* <http://127.0.0.1:8080/getmask.html>
* <http://127.0.0.1:8080/remove.html>

## How to Enable GPU

To use GPU for inference, you need to do the following:

1. Install a Pytorch version with GPU support according to [this](https://pytorch.org/get-started/locally/#start-locally).
2. Create a new file named `use_cuda` in the root of the project to enable CUDA.