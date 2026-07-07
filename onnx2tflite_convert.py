import os
from onnx2tf.onnx2tf import convert

ONNX = r'deploy/models/PPLCNet_x1_0_traffic_sign_100ep_infer/traffic_sign_100ep.onnx'
OUT_DIR = r'deploy/models/PPLCNet_x1_0_traffic_sign_100ep_tflite'

os.makedirs(OUT_DIR, exist_ok=True)
convert(
    input_onnx_file_path=ONNX,
    output_folder_path=OUT_DIR,
    output_signaturedefs=True,
    copy_onnx_input_output_names_to_tflite=True,
    verbosity='info',
)
print('done. files in', OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, f)
    if os.path.isfile(p):
        print('  ', f, os.path.getsize(p), 'bytes')
