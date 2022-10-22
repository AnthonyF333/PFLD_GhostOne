# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from torch.autograd import Variable
import torch

from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostOne import PFLD_GhostOne
import copy

import onnx
from onnxsim import simplify
import onnxoptimizer


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.
    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--model_type', default='PFLD_GhostOne', type=str)
parser.add_argument('--input_size', default=112, type=int)
parser.add_argument('--width_factor', default=1, type=float)
parser.add_argument('--landmark_number', default=98, type=int)
parser.add_argument('--model_path', default="./pfld_ghostone_best.pth")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
MODEL_DICT = {'PFLD': PFLD,
              'PFLD_GhostNet': PFLD_GhostNet,
              'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
              'PFLD_GhostOne': PFLD_GhostOne,
              }
MODEL_TYPE = args.model_type
WIDTH_FACTOR = args.width_factor
INPUT_SIZE = args.input_size
LANDMARK_NUMBER = args.landmark_number
model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE, LANDMARK_NUMBER)
model.load_state_dict(checkpoint)

if 'ghostone' in MODEL_TYPE.lower():
    model = reparameterize_model(model)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
input_names = ["input"]
output_names = ["output"]
onnx_save_name = "{}_{}_{}.onnx".format(MODEL_TYPE, INPUT_SIZE, WIDTH_FACTOR)
torch.onnx.export(model, dummy_input, onnx_save_name, verbose=False, input_names=input_names, output_names=output_names)

model = onnx.load(onnx_save_name)
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"

passes = onnxoptimizer.get_fuse_and_elimination_passes()
opt_model = onnxoptimizer.optimize(model=model, passes=passes)
final_save_name = "{}_opt.onnx".format(onnx_save_name.split('.')[0])
onnx.save(opt_model, final_save_name)
print("=====> ONNX Model save in {}".format(final_save_name))
