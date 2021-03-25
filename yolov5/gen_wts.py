import torch
import struct
from utils.torch_utils import select_device
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input',type=str,required=True,help='The path of input model.')
    args.add_argument('--output',type=str,required=True,help='The path you want to out put the model.')
    opt = args.parse_args()

    # Initialize
    device = select_device('cpu')
    # Load model
    model = torch.load(opt.input, map_location=device)
    model = model['model'].float()  # load to FP32
    model.to(device).eval()

    f = open(opt.output, 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
