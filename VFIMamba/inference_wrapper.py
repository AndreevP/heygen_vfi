import math
import torch
import torch.nn.functional as F
from .config import MODEL_CONFIG
from .padder import InputPadder   

############################
# Convert function
############################
def convert(param):
    """
    Strips "module." from keys in the checkpoint state_dict, if present.
    """
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }

############################
# Minimal Model class
############################
class Model:
    """
    Minimal model for inference usage only (no DDP, no training).
    """
    def __init__(self, local_rank=-1):
        # Read architecture config from MODEL_CONFIG
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']   # from config.py
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']     # from config.py

        # Create the network
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.device()  # Move to CUDA if available

    def device(self):
        """Moves model to GPU."""
        self.net.to(torch.device("cuda"))

    def eval(self):
        """Sets model to eval mode."""
        self.net.eval()

    def load_model(self, name=None):
        """
        Load the checkpoint into the net. If `name` is None,
        we fall back to MODEL_CONFIG['LOGNAME'].
        """
        if name is None:
            name = MODEL_CONFIG['LOGNAME']
        print(f"Loading checkpoint ckpt/{name}.pkl ...")
        checkpoint = torch.load(f'ckpt/{name}.pkl')
        self.net.load_state_dict(convert(checkpoint), strict=True)

    @torch.no_grad()
    def inference(self, img0, img1, local=True, TTA=False, fast_TTA=False,
                  timestep=0.5, scale=0.0):
        """
        Inference method that returns the middle frame between img0 and img1.
        `img0` and `img1`: Tensors with shape [B, 3, H, W].
        """
        # Concatenate frames along channel dimension: [B, 6, H, W]
        imgs = torch.cat((img0, img1), 1)

        if fast_TTA:
            # Example "fast_TTA" logic: flip, run again, average
            imgs_flip = imgs.flip(2).flip(3)
            input_ = torch.cat([imgs, imgs_flip], dim=0)
            # If your net returns multiple outputs, e.g.: _, _, _, preds = self.net(...)
            _, _, _, preds = self.net(input_, local=local, timestep=timestep, scale=scale)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        # Normal single-pass forward
        _, _, _, pred = self.net(imgs, local=local, timestep=timestep, scale=scale)
        if not TTA:
            return pred
        else:
            # Flip-based TTA
            _, _, _, pred_flip = self.net(imgs.flip(2).flip(3), local=local,
                                          timestep=timestep, scale=scale)
            return (pred + pred_flip.flip(2).flip(3)) / 2.


############################
# Recursion helper
############################
@torch.no_grad()
def _recursive_generator(model, frame1, frame2, num_recursions, index, TTA, scale):
    """
    Recursively generate intermediate frames between frame1 and frame2.
    Yields tuples of (frame, index).
    """
    if num_recursions == 0:
        yield (frame1, index)
    else:
        mid_frame = model.inference(frame1, frame2, local=True, TTA=TTA,
                                    fast_TTA=TTA, timestep=0.5, scale=scale)
        gap = 2 ** (num_recursions - 1)
        yield from _recursive_generator(model, frame1, mid_frame, num_recursions - 1,
                                        index - gap, TTA, scale)
        yield from _recursive_generator(model, mid_frame, frame2, num_recursions - 1,
                                        index + gap, TTA, scale)

############################
# Public "vfi_infer" function
############################
@torch.no_grad()
def vfi_infer(I0_np, I1_np, n=4, model_name='VFIMamba', scale=0.0):
    """
    1) Sets up MODEL_CONFIG for VFIMamba,
    2) Builds & loads the Model,
    3) Pads the input frames,
    4) Recursively generates n intermediate frames,
    5) Returns the list of frames as NumPy uint8.

    I0_np, I1_np: input frames as NumPy arrays (HxWxC, uint8, BGR or RGB)
    n: e.g. 4 => log2(4)=2 recursive calls
    model_name: defaults to 'VFIMamba'
    scale: defualt 0, recommend setting the scale to 0.5 for 2K frames and 0.25 for 4K frames
    """
    from . import config as cfg

    TTA = False
    if model_name == 'VFIMamba':
        TTA = True
        cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
        # EXACT init_model_config that matches your checkpoint
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])

    # Build model
    model = Model()
    model.load_model()  # uses MODEL_CONFIG['LOGNAME']
    model.eval()
    model.device()

    # Convert np->tensor->GPU
    I0_tensor = torch.tensor(I0_np.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.
    I1_tensor = torch.tensor(I1_np.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.

    # Pad frames to be divisible by 32
    padder = InputPadder(I0_tensor.shape, divisor=32)
    I0_tensor, I1_tensor = padder.pad(I0_tensor, I1_tensor)

    # Recursive generation
    num_recursions = int(math.log2(n))
    frames_list = list(_recursive_generator(model, I0_tensor, I1_tensor,
                                           num_recursions, n//2, TTA, scale))

    # Sort frames by index to get them in chronological order
    frames_list = sorted(frames_list, key=lambda x: x[1])

    # Unpad and convert to NumPy
    results = []
    for (f, idx) in frames_list:
        f = padder.unpad(f)[0]  # shape [3, H, W]
        f_np = (f.detach().cpu().numpy().transpose(1, 2, 0) * 255.).astype("uint8")
        results.append(f_np)

    return results
