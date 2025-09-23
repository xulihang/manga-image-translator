import onnxruntime as ort
import numpy as np
import cv2
import math
import torch
from typing import List, Tuple

class ONNXOCRCTC:
    def __init__(self, onnx_model_path, dictionary_path, blank=0):
        # 加载字典
        with open(dictionary_path, 'r', encoding='utf-8') as fp:
            self.dictionary = [s[:-1] for s in fp.readlines()]
        
        self.blank = blank
        self.dict_size = len(self.dictionary)
        
        # 创建ONNX Runtime会话
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # 获取输入和输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"成功加载ONNX模型: {onnx_model_path}")
        print(f"字典大小: {self.dict_size}")
        print(f"ONNX输入名称: {self.input_name}")
        print(f"ONNX输出名称: {self.output_names}")
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 确保输入图像尺寸正确
        if image.shape[0] != 48:
            # 保持宽高比缩放
            h, w = image.shape[:2]
            ratio = 48 / h
            new_w = int(w * ratio)
            image = cv2.resize(image, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        
        # 转换为RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # 归一化处理
        image = (image.astype(np.float32) - 127.5) / 127.5
        
        # 调整维度顺序为 [C, H, W]
        image = np.transpose(image, (2, 0, 1))
        
        # 添加批次维度 [B, C, H, W]
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def infer(self, image, verbose=False):
        """进行OCR推理"""
        # 预处理图像
        preprocessed_img = self.preprocess(image)
        
        # 获取图像宽度
        img_width = preprocessed_img.shape[3]
        
        # 执行ONNX推理
        try:
            outputs = self.session.run(self.output_names, {self.input_name: preprocessed_img})
            char_logits = outputs[0]
            color_values = outputs[1]
            
            # 解码结果
            return self.decode_ctc_top1(char_logits, color_values, verbose=verbose)
        except Exception as e:
            print(f"推理过程中出错: {e}")
            return []
    
    def decode_ctc_top1(self, pred_char_logits, pred_color_values, verbose=False):
        """解码CTC输出"""
        pred_chars = []
        
        # 计算log probabilities
        logprobs = pred_char_logits - np.log(np.sum(np.exp(pred_char_logits), axis=2, keepdims=True))
        
        # 获取预测索引
        preds_index = np.argmax(logprobs, axis=2)
        
        # 处理批次中的每个样本
        for b in range(pred_char_logits.shape[0]):
            current_chars = []
            last_ch = self.blank
            
            for t in range(pred_char_logits.shape[1]):
                pred_ch = preds_index[b, t]
                
                # CTC解码：跳过空白符和连续重复字符
                if pred_ch != last_ch and pred_ch != self.blank:
                    lp = logprobs[b, t, pred_ch]
                    
                    # 提取颜色值
                    color_values = pred_color_values[b, t].tolist()
                    
                    # 将索引转换为字符
                    current_chars.append((
                        pred_ch,
                        lp,
                        color_values[0],
                        color_values[1],
                        color_values[2],
                        color_values[3],
                        color_values[4],
                        color_values[5]
                    ))
                
                last_ch = pred_ch
            
            pred_chars.append(current_chars)
        
        # 将预测结果转换为文本
        result_text = []
        for chars in pred_chars:
            text = ''
            total_logprob = 0
            
            for (ch_idx, logprob, fr, fg, fb, br, bg, bb) in chars:
                ch = self.dictionary[ch_idx]
                if ch == '<SP>':
                    ch = ' '
                text += ch
                print(fr)
                total_logprob += logprob
            
            # 计算概率
            prob = math.exp(total_logprob / len(chars)) if chars else 0
            result_text.append((text, prob))
        
        if verbose:
            print(color_values)
            for i, (text, prob) in enumerate(result_text):
                print(f"样本 {i+1}: 文本='{text}', 概率={prob:.4f}")
        
        return result_text

# 示例用法
def main():
    # 初始化ONNX OCR模型
    onnx_ocr = ONNXOCRCTC(
        onnx_model_path='ocr_ctc.onnx',
        dictionary_path='alphabet-all-v5.txt'
    )
    
    # 加载测试图像
    try:
        # 假设test.jpg是一个包含文本的图像
        img = cv2.imread('test.jpg')
        if img is None:
            raise FileNotFoundError("测试图像未找到")
        
        # 进行OCR推理
        results = onnx_ocr.infer(img, verbose=True)
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        print("使用随机生成的图像进行测试...")
        
        # 生成随机测试图像
        test_img = np.random.randint(0, 256, (48, 320, 3), dtype=np.uint8)
        results = onnx_ocr.infer(test_img, verbose=True)

if __name__ == '__main__':
    main()