from rfdetr import RFDETRBase
import os

# 确保代理已开启，或者模型已在本地
os.environ['TRANSFORMERS_OFFLINE'] = '1' 

if __name__ == '__main__':
    model = RFDETRBase(pretrain_weights='/root/rf-detr-base-coco.pth')

    model.train(
        dataset_dir='/root/rf_detr_data', 
        epochs=100,              
        batch_size=16,           
        # --- 核心修改处 ---
        resolution=616,          # 必须是 56 的倍数，这里改成了 616
        # ----------------
        lr=1e-4,                 
        output_dir='runs/rf_detr_v1', 
        device='cuda',           
        use_ema=True,            
    )