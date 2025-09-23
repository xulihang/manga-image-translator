import onnx
from onnxconverter_common import float16

onnx_model = onnx.load("ocr_ctc.onnx")
onnx_model_fp16 = float16.convert_float_to_float16(
    onnx_model,
    keep_io_types=True,        # 保持输入输出 FP32
    op_block_list=['Cast']     # 跳过 Cast 节点，不转换
)
onnx.save(onnx_model_fp16, "ocr_ctc_fp16.onnx")

print("已导出混合精度 FP16 模型：ocr_ctc_fp16.onnx")