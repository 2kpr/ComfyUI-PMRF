# Code from https://huggingface.co/spaces/ohayonguy/PMRF/blob/main/app.py
# Some of the implementations below are adopted from
# https://huggingface.co/spaces/sczhou/CodeFormer and https://huggingface.co/spaces/wzhouxiff/RestoreFormerPlusPlus
import cv2
from tqdm import tqdm
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan.utils import RealESRGANer
from .lightning_models.mmse_rectified_flow import MMSERectifiedFlow
import torchvision
import numpy as np
from PIL import Image
import os
import folder_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_realesrgan():
    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        no_half_gpu_list = ["1650", "1660"]  # set False for GPUs that don't support f16
        if not True in [
            gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list
        ]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2,
    )
    upscale_models_path = os.path.join(folder_paths.models_dir, "upscale_models")
    realesrgan_path = os.path.join(upscale_models_path, "RealESRGAN_x2plus.pth")
    upsampler = RealESRGANer(
        scale=2,
        model_path=realesrgan_path,
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )
    return upsampler

pmrf_path = os.path.join(folder_paths.models_dir, "pmrf")
upsampler = set_realesrgan()
pmrf = MMSERectifiedFlow.from_pretrained(pmrf_path).to(device=device)

def generate_reconstructions(pmrf_model, x, y, non_noisy_z0, num_flow_steps, device):
    source_dist_samples = pmrf_model.create_source_distribution_samples(
        x, y, non_noisy_z0
    )
    dt = (1.0 / num_flow_steps) * (1.0 - pmrf_model.hparams.eps)
    x_t_next = source_dist_samples.clone()
    t_one = torch.ones(x.shape[0], device=device)
    for i in tqdm(range(num_flow_steps)):
        num_t = (i / num_flow_steps) * (
            1.0 - pmrf_model.hparams.eps
        ) + pmrf_model.hparams.eps
        v_t_next = pmrf_model(x_t=x_t_next, t=t_one * num_t, y=y).to(x_t_next.dtype)
        x_t_next = x_t_next.clone() + v_t_next * dt
    return x_t_next.clip(0, 1)

def resize(img, size, interpolation):
    # From https://github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
    h, w = img.shape[0:2]
    scale = float(size) / float(min(h, w))
    h, w = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (w, h), interpolation=interpolation)

@torch.inference_mode()
def enhance_face(img, face_helper, num_flow_steps, scale=2, interpolation=cv2.INTER_LANCZOS4):
    face_helper.clean_all()
    face_helper.read_image(img)
    face_helper.input_img = resize(face_helper.input_img, 640, interpolation)
    face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    face_helper.align_warp_face()

    # face restoration
    for i, cropped_face in tqdm(enumerate(face_helper.cropped_faces)):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        output = generate_reconstructions(
            pmrf,
            torch.zeros_like(cropped_face_t),
            cropped_face_t,
            None,
            num_flow_steps,
            device,
        )
        restored_face = tensor2img(
            output.to(torch.float32).squeeze(0), rgb2bgr=True, min_max=(0, 1)
        )
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    # upsample the background
    # Now only support RealESRGAN for upsampling background
    bg_img = upsampler.enhance(img, outscale=scale)[0]
    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
    return face_helper.cropped_faces, face_helper.restored_faces, restored_img

@torch.inference_mode()
def inference(
    imgs,
    scale,
    num_flow_steps,
    seed,
    interpolation,
):
    torch.manual_seed(seed)
    if interpolation == "lanczos4":
        interpolation = cv2.INTER_LANCZOS4
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif interpolation == "linear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "cubic":
        interpolation = cv2.INTER_CUBIC
    elif interpolation == "area":
        interpolation = cv2.INTER_AREA
    elif interpolation == "linear_exact":
        interpolation = cv2.INTER_LINEAR_EXACT
    elif interpolation == "nearest_exact":
        interpolation = cv2.INTER_NEAREST_EXACT
    imgs_output = []
    for img in imgs:
        img = img.permute(2, 0, 1)
        img = torchvision.transforms.functional.to_pil_image(img.clamp(0, 1)).convert("RGB")
        img = np.array(img)
        img = img[:, :, ::-1].copy()
        h, w = img.shape[0:2]
        size = min(h, w)
        
        face_scale = scale*(size/640)
        face_scale = face_scale if face_scale < scale else scale
        face_scale = face_scale if face_scale > 1.0 else 1.0

        face_helper = FaceRestoreHelper(
            face_scale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=None,
        )

        cropped_face, restored_faces, restored_img = enhance_face(
            img,
            face_helper,
            num_flow_steps=num_flow_steps,
            scale=face_scale,
            interpolation=interpolation,
        )

        output = restored_img
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = resize(output, size*scale, interpolation)

        torch.cuda.empty_cache()
        output = torchvision.transforms.functional.pil_to_tensor(Image.fromarray(output)).to(torch.float32) / 255.0
        output = output.permute(1, 2, 0)
        imgs_output.append(output[None,])
    return (torch.cat(tuple(imgs_output), dim=0),)

class PMRF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 40.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 400, "step": 1}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 2**32, "step": 1}),
                "interpolation": (["lanczos4", "nearest", "linear", "cubic", "area", "linear_exact", "nearest_exact"],)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images", )
    FUNCTION = "pmrf"
    CATEGORY = "PMRF"

    def pmrf(self, images, scale, num_steps, seed, interpolation):
        return inference(images, scale, num_steps, seed, interpolation)
    
NODE_CLASS_MAPPINGS = {
    "PMRF": PMRF,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PMRF": "PMRF",
}
