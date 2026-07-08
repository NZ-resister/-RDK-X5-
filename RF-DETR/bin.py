import cv2
import numpy as np
import time
# 👇 唤醒 BPU 的终极咒语
from hobot_dnn import pyeasy_dnn as dnn 

def letterbox(img, new_shape=(384, 384), color=(114, 114, 114)):
    """和校准时一模一样的预处理函数，保证精度不折损"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def main():
    # ==========================================
    # 1. 加载 BPU 引擎
    # ==========================================
    print("🔥 正在加载 BPU 模型: rf_detr.bin ...")
    models = dnn.load("rf_detr.bin")
    model = models[0] # 一个 bin 文件里可能打包了多个模型，我们取第一个
    print("✅ BPU 引擎加载完毕！")

    # ==========================================
    # 2. 图片预处理 (极其严谨)
    # ==========================================
    # 请把这里的图片路径换成你板子上真实存在的测试图
    img_path = "test_image.jpg" 
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 找不到图片: {img_path}")
        return

    # 必须和 Docker 校准时的操作完全一致：BGR转RGB -> Letterbox -> 转NCHW排布
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_padded = letterbox(img_rgb, (384, 384))
    input_data = np.ascontiguousarray(img_padded.transpose(2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0) # 加上 Batch 维度变成 (1, 3, 384, 384)

    # ==========================================
    # 3. BPU 狂暴执行
    # ==========================================
    print("⚡ BPU 推理中...")
    start_time = time.time()
    
    # 核心推理代码，就这一句！
    outputs = model.forward(input_data)
    
    end_time = time.time()
    print(f"⏱️ 纯 BPU 推理耗时: {(end_time - start_time) * 1000:.2f} ms")

    # ==========================================
    # 4. 解析 BPU 输出
    # ==========================================
    print(f"📦 BPU 输出了 {len(outputs)} 个特征图节点：")
    for i, out in enumerate(outputs):
        # ⚠️ 极其关键：hobot_dnn 的输出是一个封装对象，必须用 .buffer 才能拿到纯 Numpy 数组
        tensor_data = out.buffer 
        print(f"  👉 节点 {i} 形状: {tensor_data.shape}")

    # ==========================================
    # 5. 后处理与画框 (解码 BPU 吐出的数字)
    # ==========================================
    print("🎨 正在解析边界框并绘制图片...")
    
    # DETR 通常输出多个解码层的预测，我们取最后一层（节点 4 是类别，节点 5 是框）
    # 注意：这里的 buffer 提取出来就是纯 Numpy 数组了
    logits = outputs[4].buffer.reshape(300, 11) 
    boxes = outputs[5].buffer.reshape(300, 4)   

    # 恢复原图用于画框 (假设你一开始读取的图叫 img)
    draw_img = img.copy()
    h_orig, w_orig = draw_img.shape[:2]

    # 设置置信度阈值，过滤掉垃圾框
    CONF_THRESH = 0.5 
    
    # 将 logits 转换为概率 (如果是 Focal Loss 可能是 sigmoid，这里给个通用的处理)
    # 假设最后/第一个维度是背景类，我们取所有前景类别的最大概率
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True) # Softmax
    scores = np.max(probs[:, :-1], axis=-1) # 抛弃最后一个背景类的概率
    class_ids = np.argmax(probs[:, :-1], axis=-1)

    for i in range(300):
        score = scores[i]
        if score > CONF_THRESH:
            # 获取归一化的中心点和宽高
            cx, cy, w, h = boxes[i]
            
            # 反归一化到原图尺寸 (DETR 输出通常是 0~1 的比例值)
            # 注意：如果你的模型输出没有归一化，请把这里的 * w_orig 去掉
            x1 = int((cx - w / 2) * w_orig)
            y1 = int((cy - h / 2) * h_orig)
            x2 = int((cx + w / 2) * w_orig)
            y2 = int((cy + h / 2) * h_orig)

            # 画框 (绿色，线宽 2)
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 打标签
            label = f"Class {class_ids[i]}: {score:.2f}"
            cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 6. 保存最终结果！
    save_path = "result_bpu.jpg"
    cv2.imwrite(save_path, draw_img)
    print(f"🎉 大功告成！结果已保存至: {save_path}")

# 注意：把这块代码放到 main() 函数的最下面。

if __name__ == '__main__':
    main()
