def apply_filters(roi):
    # Apply the Canny edge detection filter
    edges = cv2.Canny(roi, 100, 200)
    
    # Blur the edges
    blurred_edges = cv2.GaussianBlur(edges, (5, 5), 0)
    
    # Add the blurred edges to the original image to create edge glow
    roi_with_edges = cv2.addWeighted(roi, 1, blurred_edges, 1.75, 0)
    
    # Gamma correction and exposure
    gamma = 0.48
    exposure = 0.87
    corrected_roi = np.power(roi_with_edges / 255, gamma) * (255 + 255 * exposure)
    
    # Enhance the details
    detail_enhanced_roi = cv2.detailEnhance(corrected_roi.astype(np.uint8), sigma_s=10, sigma_r=0.15)

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(detail_enhanced_roi, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    L, A, B = cv2.split(lab_image)

    # Adjust shadows and highlights
    shadows = 0
    highlights = 85
    L = cv2.normalize(L, None, shadows, highlights + 100, cv2.NORM_MINMAX)

    # Merge the LAB channels back together
    updated_lab_image = cv2.merge([L, A, B])

    # Convert the LAB image back to BGR
    updated_bgr_image = cv2.cvtColor(updated_lab_image, cv2.COLOR_LAB2BGR)

    # Adjust vibrance and saturation
    hsv_image = cv2.cvtColor(updated_bgr_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)
    S = cv2.add(S, 34)  # Saturation
    V = cv2.add(V, 100)  # Vibrance
    updated_hsv_image = cv2.merge([H, S, V])

    # Convert the HSV image back to BGR
    final_roi = cv2.cvtColor(updated_hsv_image, cv2.COLOR_HSV2BGR)

    return final_roi
