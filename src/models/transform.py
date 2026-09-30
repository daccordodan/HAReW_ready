import numpy as np

def dopplerTraceTransformation(window):
    """
    Applies sequential data augmentation transformations to a Doppler trace.

    Args:
        window: input Doppler trace array

    Returns:
        augmented Doppler trace array
    """
    p = 0.6
    augmented = doppler_shift(p, window, shift_bin=5)
    augmented = gaussian_noise(p, augmented, noise_std=0.04)
    augmented = temporal_mask(p, augmented, mask_width=(3, 10), n_masks=4)
    augmented = frequency_mask(p, augmented, mask_width=(2, 5), n_masks=3)
    return augmented


def doppler_shift(p, trace, shift_bin):
    """
    Randomly shifts the Doppler trace along the temporal axis.

    Args:
        p: probability of applying shift
        trace: input Doppler trace array
        shift_bin: maximum bin shift range

    Returns:
        shifted Doppler trace array
    """
    if np.random.rand() > p:
        return trace
    shift = np.random.randint(-shift_bin, shift_bin + 1)
    shifted = np.roll(trace, shift, axis=1)

    if shift > 0:
        shifted[:, :shift, :] = trace.min()
    elif shift < 0:
        shifted[:, shift:, :] = trace.min()
    return shifted


def gaussian_noise(p, trace, noise_std=0.05):
    """
    Adds zero-mean Gaussian noise to the Doppler trace.

    Args:
        p: probability of applying noise
        trace: input Doppler trace array
        noise_std: standard deviation of Gaussian noise

    Returns:
        noisy Doppler trace array clipped to [0, 1]
    """
    if np.random.rand() > p:
        return trace

    noise = np.random.randn(*trace.shape) * noise_std
    return np.clip(trace + noise, 0, 1)


def temporal_mask(p, trace, mask_width, n_masks):
    """
    Applies random zero-value temporal masks across time steps.

    Args:
        p: probability of applying temporal masking
        trace: input Doppler trace array
        mask_width: tuple specifying min and max mask width
        n_masks: upper bound for number of temporal masks

    Returns:
        temporally masked Doppler trace array
    """
    if np.random.rand() > p:
        return trace

    masked = trace.copy()
    ts = trace.shape[1]
    n_masks = np.random.randint(1, n_masks)
    for _ in range(n_masks):
        width = np.random.randint(mask_width[0], mask_width[1])
        start_time = np.random.randint(0, max(1, ts - width))
        masked[:, start_time:start_time + width, :] = trace.min()
    return masked


def frequency_mask(p, trace, mask_width, n_masks):
    """
    Applies random zero-value frequency masks across frequency bins.

    Args:
        p: probability of applying frequency masking
        trace: input Doppler trace array
        mask_width: tuple specifying min and max mask width
        n_masks: upper bound for number of frequency masks

    Returns:
        frequency-masked Doppler trace array
    """
    if np.random.rand() > p:
        return trace

    masked = trace.copy()
    fb = trace.shape[2]
    n_masks = np.random.randint(1, n_masks)
    for _ in range(n_masks):
        width = np.random.randint(mask_width[0], mask_width[1])
        start_freq = np.random.randint(0, max(1, fb - width))
        masked[:, :, start_freq:start_freq + width] = trace.min()
    return masked