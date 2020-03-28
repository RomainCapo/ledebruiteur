from skimage import metrics


def compare_images(orignal_img, transformed_img):
    """Compares original image to transformed image

    Arguments:
        orignal_img {Array} -- Numpy like array of image
        transformed_img {Array} -- Numpy like array of image

    Returns:
        dict -- {"MSE", "NRMSE", "PSNR", "SSIM"}
    """
    mse = metrics.mean_squared_error(orignal_img, transformed_img)
    nrmse = metrics.normalized_root_mse(orignal_img, transformed_img)
    psnr = metrics.peak_signal_noise_ratio(orignal_img, transformed_img)
    ssim = metrics.structural_similarity(
        orignal_img, transformed_img, multichannel=True)

    return {"MSE": mse, "NRMSE": nrmse, "PSNR": psnr, "SSIM": ssim}
