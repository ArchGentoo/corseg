
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Callable
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from .soft_skeleton import SoftSkeletonize

def compute_skeleton(mask):
    """
    Compute the skeleton of a binary mask.
    """
    # Convert PyTorch tensor to NumPy array
    mask_np = mask.cpu().numpy()
    # Compute skeleton
    skeleton_np = skeletonize(mask_np).astype(mask_np.dtype)
    # Convert back to PyTorch tensor
    skeleton_tensor = torch.from_numpy(skeleton_np).to(mask.device)
    return skeleton_tensor

def compute_skeletons(V_P, V_L):
    """
    Compute skeletons from predicted and ground truth masks.
    """
    batch_size = V_P.shape[0]
    S_P_list = []
    S_L_list = []
    for i in range(batch_size):
        # Binarize masks
        vp_bin = (V_P[i, 0] > 0.5).float()
        vl_bin = (V_L[i, 0] > 0.5).float()
        # Compute skeletons
        sp = compute_skeleton(vp_bin)
        sl = compute_skeleton(vl_bin)
        # Append to lists
        S_P_list.append(sp.unsqueeze(0))
        S_L_list.append(sl.unsqueeze(0))
    # Stack tensors
    S_P = torch.stack(S_P_list, dim=0)
    S_L = torch.stack(S_L_list, dim=0)
    return S_P, S_L

def compute_nsdt(skeleton, mask):
    """
    Compute the Normalized Skeleton Distance Transform (NSDT).
    """
    batch_size = skeleton.shape[0]
    NSDT_list = []
    for i in range(batch_size):
        skeleton_np = skeleton[i, 0].cpu().numpy()
        mask_np = mask[i, 0].cpu().numpy()
        # Compute distance transform from the skeleton
        distance = distance_transform_edt(1 - skeleton_np)
        # Initialize NSDT array
        nsdt_np = np.zeros_like(distance)
        # Compute NSDT for foreground pixels
        foreground_indices = mask_np > 0
        nsdt_np[foreground_indices] = 1 / (distance[foreground_indices] + 1)
        # Convert to PyTorch tensor
        nsdt_tensor = torch.from_numpy(nsdt_np).unsqueeze(0).to(mask.device)
        NSDT_list.append(nsdt_tensor)
    NSDT = torch.stack(NSDT_list, dim=0)
    return NSDT

def compute_T_prec(S_P, NSDTP, V_L):
    """
    Compute topology-aware precision T_prec*.
    """
    numerator = (S_P * NSDTP * V_L).sum(dim=[1, 2, 3])
    denominator = (S_P * NSDTP).sum(dim=[1, 2, 3]) + 1e-6  # Avoid division by zero
    T_prec = numerator / denominator
    return T_prec

def compute_T_sens(S_L, NSDTL, V_P):
    """
    Compute topology-aware sensitivity T_sens*.
    """
    numerator = (S_L * NSDTL * V_P).sum(dim=[1, 2, 3])
    denominator = (S_L * NSDTL).sum(dim=[1, 2, 3]) + 1e-6  # Avoid division by zero
    T_sens = numerator / denominator
    return T_sens

def compute_L_dscl(T_prec, T_sens):
    """
    Compute NSDT Soft-ClDice loss.
    """
    numerator = 2 * T_prec * T_sens
    denominator = T_prec + T_sens + 1e-6  # Avoid division by zero
    L_dscl = 1 - numerator / denominator
    return L_dscl

def compute_L_dc(V_P, V_L):
    """
    Compute Soft-Dice loss.
    """
    smooth = 1e-6  # Smoothing factor to avoid division by zero
    intersection = (V_P * V_L).sum(dim=[1, 2, 3])
    union = V_P.sum(dim=[1, 2, 3]) + V_L.sum(dim=[1, 2, 3])
    dice = (2 * intersection + smooth) / (union + smooth)
    L_dc = 1 - dice
    return L_dc

def compute_L_dc_dscl(L_dc, L_dscl, gamma=0.5):
    """
    Compute the combined loss L_dc&dscl.
    """
    L_dc_dscl = (1 - gamma) * L_dc + gamma * L_dscl
    return L_dc_dscl

def NSDT_Soft_ClDice_Loss(V_P, V_L, gamma=0.5):
    """
    Compute the final NSDT Soft-ClDice loss.
    """
    # Step 1: Compute skeletons
    S_P, S_L = compute_skeletons(V_P, V_L)
    # Step 2: Compute NSDTs
    NSDTP = compute_nsdt(S_P, V_P)
    NSDTL = compute_nsdt(S_L, V_L)
    # Step 3: Compute topology-aware precision
    T_prec = compute_T_prec(S_P, NSDTP, V_L)
    # Step 4: Compute topology-aware sensitivity
    T_sens = compute_T_sens(S_L, NSDTL, V_P)
    # Step 5: Compute NSDT Soft-ClDice loss
    L_dscl = compute_L_dscl(T_prec, T_sens)
    # Step 6: Compute Soft-Dice loss
    L_dc = compute_L_dc(V_P, V_L)
    # Combine losses
    L_dc_dscl = compute_L_dc_dscl(L_dc, L_dscl, gamma)
    # Return the mean loss over the batch
    loss = L_dc_dscl.mean()
    return loss

if __name__ == '__main__':
    # Example inputs (batch_size=1, channels=1, height=256, width=256)
    V_P = torch.rand(1, 1, 256, 256)  # Predicted mask (values between 0 and 1)
    V_L = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Ground truth mask (values 0 or 1)

    # Compute loss
    loss = NSDT_Soft_ClDice_Loss(V_P, V_L, gamma=0.5)
    print("Loss:", loss.item())
