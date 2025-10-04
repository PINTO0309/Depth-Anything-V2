import torch

from depth_anything_v2.dpt import DepthAnythingV2
import onnx
from onnxsim import simplify

DEVICE = 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' #'vitl', 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# ----------------------------------------------------
# ダミー入力
# ----------------------------------------------------
batch_size = 1
height = 518  # 好きなサイズに変更可（例: 518, 512, 384）
width = 518
dummy_input = torch.randn(batch_size, 3, height, width, device=DEVICE)

# ----------------------------------------------------
# ONNX エクスポート
# ----------------------------------------------------
onnx_path = f'depth_anything_v2_small.onnx'

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=['pixel_values'],
    output_names=['predicted_depth'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
        'predicted_depth': {0: 'batch_size', 2: 'height', 3: 'width'},
    }
)

model_simp, check = simplify(onnx_path)
onnx.save(model_simp, onnx_path)

print(f"✅ ONNX model exported successfully → {onnx_path}")