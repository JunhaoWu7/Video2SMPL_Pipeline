import torch
import numpy as np

def get_frame_id_list_from_mask(mask):

    """
    Vectorized approach to get frame id list from a boolean mask.

    Args:
        mask (F,), bool tensor: Mask array where `True` indicates a frame to be processed.

    Returns:
        frame_id_list: List of torch.Tensors, each tensor containing continuous indices where mask is True.
    """
    # Find the indices where the mask changes from False to True and vice versa
    padded_mask = torch.cat(
        [torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)]
    )
    diffs = torch.diff(padded_mask.int())
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze()
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze()
    if starts.numel() == 0:
        return []
    if starts.numel() == 1:
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)

    # Create list of ranges
    frame_id_list = [torch.arange(start, end) for start, end in zip(starts, ends)]
    return frame_id_list


def get_frame_id_list_from_frame_id(frame_id):
    mask = torch.zeros(frame_id[-1] + 1, dtype=torch.bool)
    mask[frame_id] = True
    frame_id_list = get_frame_id_list_from_mask(mask)
    return frame_id_list

def rearrange_by_mask(x, mask):
    """
    x (L, *)
    mask (M,), M >= L
    """
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    assert M > L
    assert mask.sum() == L
    x_rearranged = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    x_rearranged[mask] = x
    return x_rearranged


def frame_id_to_mask(frame_id, max_len):
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[frame_id] = True
    return mask


def mask_to_frame_id(mask):
    frame_id = torch.where(mask)[0]
    return frame_id

def linear_interpolate_frame_ids(data, frame_id_list):
    """
    A more robust version of the interpolation function.
    """
    data = data.clone()
    total_len = len(data)

    for invalid_frame_ids in frame_id_list:
        if len(invalid_frame_ids) == 0:
            continue

        start_id = invalid_frame_ids[0]
        end_id = invalid_frame_ids[-1]

        # Case 1: Missing frames at the beginning
        if start_id == 0:
            next_valid_id = end_id + 1
            # Check if the whole sequence is missing
            if next_valid_id >= total_len:
                # Handle the case where all frames are invalid
                # Option 1: Raise an error
                # raise ValueError("All frames are invalid, cannot interpolate.")
                # Option 2: Do nothing and return the original data (or a zero tensor)
                print(f"Warning: All {total_len} frames are invalid. Returning original data.")
                return data # Or return torch.zeros_like(data)
            
            # Fill with the next valid frame's data
            fill_data = data[next_valid_id].clone()
            for frame_id in invalid_frame_ids:
                data[frame_id] = fill_data.clone()
        
        # Case 2: Missing frames at the end
        elif end_id == total_len - 1:
            prev_valid_id = start_id - 1
            # Fill with the previous valid frame's data
            fill_data = data[prev_valid_id].clone()
            for frame_id in invalid_frame_ids:
                data[frame_id] = fill_data.clone()

        # Case 3: Missing frames in the middle
        else:
            prev_valid_id = start_id - 1
            next_valid_id = end_id + 1
            
            prev_data = data[prev_valid_id]
            next_data = data[next_valid_id]
            
            # Linear interpolation logic (your original logic is fine here)
            if data.dim() == 1:
                # For 1D tensor
                interpolation_points = torch.linspace(
                    0, 1, len(invalid_frame_ids) + 2, device=data.device
                )[1:-1]
                interpolated_values = prev_data + interpolation_points * (next_data - prev_data)
                data[invalid_frame_ids] = interpolated_values
            else:
                # For multi-dimensional tensor
                interpolation_points = torch.linspace(
                    0, 1, len(invalid_frame_ids) + 2, device=data.device
                )[1:-1]
                # Unsqueeze to allow broadcasting
                interpolated_values = prev_data + interpolation_points.view(-1, *([1] * (data.dim() - 1))) * (next_data - prev_data)
                data[invalid_frame_ids] = interpolated_values
    
    return data


def linear_interpolate(data, N_middle_frames):
    """
    Args:
        data: (2, C)
    Returns:
        data_interpolated: (1+N+1, C)
    """
    prev = data[0]
    next = data[1]
    middle = torch.linspace(0, 1, N_middle_frames + 2)[1:-1][:, None] * (next - prev)[None] + prev[None]  # (N, C)
    data_interpolated = torch.cat([data[0][None], middle, data[1][None]], dim=0)  # (1+N+1, C)
    return data_interpolated


def find_top_k_span(mask, k=3):
    """
    Args:
        mask: (L,)
    Return:
        topk_span: List of tuple, usage: [start, end)
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.sum() == 0:
        return []
    mask = mask.clone().float()
    mask = torch.cat([mask.new([0]), mask, mask.new([0])])
    diff = mask[1:] - mask[:-1]
    start = torch.where(diff == 1)[0]
    end = torch.where(diff == -1)[0]
    assert len(start) == len(end)
    span_lengths = end - start
    span_lengths, idx = span_lengths.sort(descending=True)
    start = start[idx]
    end = end[idx]
    return list(zip(start.tolist(), end.tolist()))[:k]
