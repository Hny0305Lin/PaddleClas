import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

ONNX = r'deploy/models/PPLCNet_x1_0_traffic_sign_100ep_infer/traffic_sign_100ep.onnx'
TFLITE = r'deploy/models/PPLCNet_x1_0_traffic_sign_100ep_tflite/traffic_sign_100ep_float32.tflite'

np.random.seed(0)
x_nchw = np.random.randn(2, 3, 224, 224).astype(np.float32)

# ONNX (NCHW)
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
onnx_out = sess.run(None, {in_name: x_nchw})[0]
print('ONNX  out shape:', onnx_out.shape, '| first5:', np.round(onnx_out[0,:5],4))

# TFLite (NHWC)
itp = Interpreter(model_path=TFLITE)
itp.allocate_tensors()
id_det = itp.get_input_details()[0]
od_det = itp.get_output_details()[0]
print('TFLite IN :', id_det['name'], id_det['shape'], id_det['dtype'])
print('TFLite OUT:', od_det['name'], od_det['shape'], od_det['dtype'])
x_nhwc = np.transpose(x_nchw, (0,2,3,1))
if id_det['shape'][0] == 1:
    x_nhwc = x_nhwc[:1]
itp.set_tensor(id_det['index'], x_nhwc.astype(id_det['dtype']))
itp.invoke()
tfl_out = itp.get_tensor(od_det['index'])
if tfl_out.shape[0] == 1 and onnx_out.shape[0] == 2:
    onnx_cmp = onnx_out[:1]
else:
    onnx_cmp = onnx_out
print('TFLite out shape:', tfl_out.shape, '| first5:', np.round(tfl_out[0,:5],4))
diff = np.abs(onnx_cmp - tfl_out).max()
print('MAX abs diff (ONNX vs TFLite):', float(diff))
print('argmax ONNX :', np.argmax(onnx_cmp,1))
print('argmax TFLite:', np.argmax(tfl_out,1))
