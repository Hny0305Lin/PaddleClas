import paddle2onnx
import os

INFER_DIR = r'deploy/models/PPLCNet_x1_0_traffic_sign_100ep_infer'
MODEL = os.path.join(INFER_DIR, 'inference.pdmodel')
PARAMS = os.path.join(INFER_DIR, 'inference.pdiparams')
OUT = os.path.join(INFER_DIR, 'traffic_sign_100ep.onnx')

paddle2onnx.export(
    model_filename=MODEL,
    params_filename=PARAMS,
    save_file=OUT,
    opset_version=14,
    auto_upgrade_opset=True,
    enable_onnx_checker=True,
    enable_optimize=True,
    deploy_backend='onnxruntime',
)
print('saved:', OUT, '| size:', os.path.getsize(OUT), 'bytes')
