import torch
import torch.nn as nn
import torch.nn.functional as F
from rfdetr import RFDETRNano

# ==========================================
# 🧙‍♂️ 唯一需要的魔法补丁：精准摘除 antialias
# 解决 aten::_upsample_bicubic2d_aa 报错
# ==========================================
original_interpolate = F.interpolate

def patched_interpolate(*args, **kwargs):
    if 'antialias' in kwargs:
        kwargs['antialias'] = False # 强制关闭导致 ONNX 不兼容的抗锯齿
    return original_interpolate(*args, **kwargs)

# 替换官方插值函数
F.interpolate = patched_interpolate
# ==========================================


def find_real_model(obj):
    """自动探测并返回纯正的 PyTorch nn.Module"""
    if isinstance(obj, nn.Module): return obj
    if hasattr(obj, 'model') and isinstance(obj.model, nn.Module): return obj.model
    if hasattr(obj, 'net') and isinstance(obj.net, nn.Module): return obj.net
    if hasattr(obj, 'network') and isinstance(obj.network, nn.Module): return obj.network
    if hasattr(obj, 'model') and hasattr(obj.model, 'model') and isinstance(obj.model.model, nn.Module): return obj.model.model
    return None

print("正在加载 300 轮最强 Nano 权重...")
# 加载外壳 (传 num_classes=10 消除类别警告)
wrapper = RFDETRNano(pretrain_weights='runs/rf_detr_nano/checkpoint_best_ema.pth', num_classes=10)

real_model = find_real_model(wrapper)
if real_model is None and hasattr(wrapper, 'model'):
    real_model = find_real_model(wrapper.model)

if real_model is not None:
    print("🔍 成功拿到纯正的底层模型！准备转换 ONNX...")
    real_model.eval() 
    
    # RDK X5 Nano 原生输入分辨率 384x384
    dummy_input = torch.randn(1, 3, 384, 384)
    
    print("正在以 Opset 16 导出 ONNX (已自动免疫抗锯齿算子报错)...")
    torch.onnx.export(
        real_model, 
        dummy_input, 
        "rf_detr_nano.onnx", 
        opset_version=11,      # 保持 16，释放 RDK X5 的全部潜力
        input_names=['images'], 
        output_names=['output']
    )
    print("🎉 历经千辛万苦，ONNX 终于导出成功了！")
else:
    print("😭 模型藏得太深了！")