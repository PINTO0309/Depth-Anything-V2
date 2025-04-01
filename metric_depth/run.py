import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import onnx
from onnxsim import simplify

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vits.pth')

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    parser.add_argument('--export-onnx', dest='export_onnx', action='store_true')
    parser.add_argument('--opset-version', dest='opset_version', default=13)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if args.export_onnx:
        DEVICE = 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    inout = ""
    max_depth = 20
    if "hypersim" in args.load_from:
        max_depth = 20
        inout = f"indoor_maxdepth{max_depth}"
    elif "vkitti" in args.load_from:
        max_depth = 80
        inout = f"outdoor_maxdepth{max_depth}"

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if not args.export_onnx:
        if os.path.isfile(args.img_path):
            if args.img_path.endswith('txt'):
                with open(args.img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [args.img_path]
        else:
            filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

        os.makedirs(args.outdir, exist_ok=True)

        cmap = matplotlib.colormaps.get_cmap('Spectral')

        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')

            raw_image = cv2.imread(filename)

            depth = depth_anything.infer_image(
                raw_image=raw_image,
                input_size=args.input_size,
            )

            if args.save_numpy:
                output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
                np.save(output_path, depth)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
            if args.pred_only:
                cv2.imwrite(output_path, depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])

                cv2.imwrite(output_path, combined_result)

    else:
        basename_without_ext = os.path.splitext(os.path.basename(args.load_from))[0]
        MODEL = f'{basename_without_ext}_{inout}'

        onnx_file = f"{MODEL}_1x3x{args.input_size}x{args.input_size}.onnx"
        x = torch.randn(1, 3, args.input_size, args.input_size).cpu()
        torch.onnx.export(
            depth_anything,
            args=(x),
            f=onnx_file,
            opset_version=args.opset_version,
            input_names=['pixel_values'],
            output_names=['predicted_depth'],
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

        # onnx_file = f"{MODEL}_Nx3xHxW.onnx"
        # x = torch.randn(1, 3, args.input_size, args.input_size).cpu()
        # torch.onnx.export(
        #     depth_anything,
        #     args=(x),
        #     f=onnx_file,
        #     opset_version=args.opset_version,
        #     input_names=['pixel_values'],
        #     output_names=['predicted_depth'],
        #     dynamic_axes={
        #         'pixel_values' : {0:'batch_size', 2: 'height', 3: 'width'},
        #         'predicted_depth' : {0:'batch_size', 1: 'height', 2: 'width'},
        #     }
        # )
        # model_onnx1 = onnx.load(onnx_file)
        # model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        # onnx.save(model_onnx1, onnx_file)