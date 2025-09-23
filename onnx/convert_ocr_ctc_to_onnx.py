import torch
import numpy as np
from model_ocr_ctc import OCR
import os

# 设置设备
device = torch.device('cpu')  # 使用CPU转换以确保更好的兼容性

# 加载字典
with open('alphabet-all-v5.txt', 'r', encoding='utf-8') as fp:
    dictionary = [s[:-1] for s in fp.readlines()]

# 创建模型实例
model = OCR(dictionary, 768).to(device)
model.eval()

# 加载模型权重
try:
    checkpoint = torch.load('ocr-ctc.ckpt', map_location=device)
    # 检查checkpoint结构
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    
    # 处理位置编码的问题
    if 'encoders.layers.0.pe.pe' in checkpoint:
        del checkpoint['encoders.layers.0.pe.pe']
    if 'encoders.layers.1.pe.pe' in checkpoint:
        del checkpoint['encoders.layers.1.pe.pe']
    if 'encoders.layers.2.pe.pe' in checkpoint:
        del checkpoint['encoders.layers.2.pe.pe']
    
    # 加载权重，strict=False允许忽略不匹配的键
    model.load_state_dict(checkpoint, strict=False)
    print("成功加载模型权重")
except Exception as e:
    print(f"加载模型权重时出错: {e}")
    exit(1)

# 创建示例输入（高度固定为48，宽度可以是任意合理值）
batch_size = 1
channels = 3
height = 48
width = 1536  # 使用一个典型的最大宽度

dummy_input = torch.randn(batch_size, channels, height, width).to(device)

# 导出为ONNX格式
onnx_output_path = 'ocr_ctc.onnx'
print(f"正在将模型转换为ONNX格式，保存到: {onnx_output_path}")

# 定义输入输出名称
export_input_names = ['input']
export_output_names = ['char_logits', 'color_values']

# 导出模型
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=14,  # 选择与ONNX Runtime兼容的opset版本
    do_constant_folding=True,
    input_names=export_input_names,
    output_names=export_output_names,
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},  # 允许高度和宽度变化
        'char_logits': {1: 'seq_len'},       # 字符logits的序列长度可变
        'color_values': {1: 'seq_len'}       # 颜色值的序列长度可变
    },
    verbose=False
)

print(f"模型已成功转换为ONNX格式: {onnx_output_path}")

# 验证ONNX模型
print("正在验证ONNX模型...")
try:
    import onnx
    # 检查模型是否有效
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证成功")

except Exception as e:
    print(f"ONNX模型验证失败: {e}")