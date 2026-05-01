import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import os

def apply_clahe_bgr(img, clip=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)




# ============================================================
# 1. ALIGN METHODS (ORB / SIFT / SURF)
# ============================================================

def orb_align_images(ref_img, test_img, n_features=2000, draw_matches=True):
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(test, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    aligned_color = cv2.warpPerspective(test_img, H, (ref_img.shape[1], ref_img.shape[0]))

    if draw_matches:
        vis = cv2.drawMatches(ref, kp1, test, kp2, matches[:50], None, flags=2)
        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1); plt.imshow(vis, cmap='gray')
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(aligned_color, cv2.COLOR_BGR2RGB))
        plt.show()

    return aligned_color, H


def sift_align_images(ref_img, test_img, n_features=4000, draw_matches=True):
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(n_features)
    kp1, des1 = sift.detectAndCompute(ref, None)
    kp2, des2 = sift.detectAndCompute(test, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, m = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    aligned = cv2.warpPerspective(test_img, H, (ref_img.shape[1], ref_img.shape[0]))

    if draw_matches:
        vis = cv2.drawMatches(ref, kp1, test, kp2, good[:50], None, flags=2)
        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1); plt.imshow(vis)
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        plt.show()

    return aligned, H


def surf_align_images(ref_img, test_img, hessian=400, draw_matches=True):
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(hessian)
    kp1, des1 = surf.detectAndCompute(ref, None)
    kp2, des2 = surf.detectAndCompute(test, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    aligned = cv2.warpPerspective(test_img, H, (ref_img.shape[1], ref_img.shape[0]))

    if draw_matches:
        vis = cv2.drawMatches(ref, kp1, test, kp2, good[:50], None, flags=2)
        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1); plt.imshow(vis)
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        plt.show()

    return aligned, H


# ============================================================
# 2. PCB CROP / CANVAS / RESIZE
# ============================================================

def crop_by_points(img, x1, y1, x2, y2):
    H, W = img.shape[:2]

    # đảm bảo thứ tự đúng
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # clamp biên ảnh
    x1 = max(0, min(x1, W-1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H-1))
    y2 = max(0, min(y2, H))

    return img[y1:y2, x1:x2]

def crop_pcb_from_color(aligned_img, lower_hsv=(90, 50, 40), upper_hsv=(140, 255, 255), debug=True):
    hsv = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return aligned_img

    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cropped = aligned_img[y:y+h, x:x+w]

    if debug:
        debug_img = aligned_img.copy()
        cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,0,255), 3)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return cropped


def paste_crop_to_canvas(cropped_img, canvas_size=(2590,2590), debug=False):
    H, W = canvas_size
    canvas = np.ones((H,W,3), np.uint8)*255

    ch, cw = cropped_img.shape[:2]

    if ch>H or cw>W:
        scale = min(H/ch, W/cw)
        cropped_img = cv2.resize(cropped_img, (int(cw*scale), int(ch*scale)))
        ch, cw = cropped_img.shape[:2]

    y_off = (H-ch)//2
    x_off = (W-cw)//2

    canvas[y_off:y_off+ch, x_off:x_off+cw] = cropped_img

    if debug:
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.show()

    return canvas


def resize_with_black_padding(img, target=1024):
    h,w = img.shape[:2]
    scale = target/max(h,w)

    new_w = int(w*scale)
    new_h = int(h*scale)

    resized = cv2.resize(img, (new_w,new_h))
    canvas = np.zeros((target,target,3), np.uint8)

    top = (target-new_h)//2
    left = (target-new_w)//2

    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas


# ============================================================
# 3. YOLO INFERENCE
# ============================================================

def yolo_infer(model_path, img_path, save_folder="infer_results", debug=False):
    os.makedirs(save_folder, exist_ok=True)

    model = YOLO(model_path)
    results = model(img_path, show=debug)

    img = cv2.imread(img_path)
    for b in results[0].boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        label = f"{results[0].names[int(b.cls[0])]} {float(b.conf[0]):.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    out = os.path.join(save_folder, os.path.basename(img_path))
    cv2.imwrite(out, img)
    return results, out


def yolo_color(model_path, img_path, save_folder="infer_results", debug=True):
    os.makedirs(save_folder, exist_ok=True)

    model = YOLO(model_path)
    results = model(img_path)

    annotated = results[0].plot()
    out = os.path.join(save_folder, os.path.basename(img_path))
    cv2.imwrite(out, annotated)

    if debug:
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.show()

    return results, out


# ============================================================
# 4. SSIM / EDGE SSIM / DIFF VISUALIZATION
# ============================================================

def ssim_compare_and_highlight(results, golden_img, test_img, pad=-20, threshold=0.5):
    g_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    annotated = test_img.copy()
    report = []
    clahe = cv2.createCLAHE(2.0,(8,8))

    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        name = results[0].names[int(box.cls[0])]

        x1p = max(0, x1-pad)
        y1p = max(0, y1-pad)
        x2p = min(t_gray.shape[1], x2+pad)
        y2p = min(t_gray.shape[0], y2+pad)

        roi_t = t_gray[y1p:y2p, x1p:x2p]
        roi_g = g_gray[y1p:y2p, x1p:x2p]

        if roi_t.size == 0 or roi_g.size == 0:
            score = None
            status = "SKIP"
            color = (255,255,0)
        else:
            roi_t = clahe.apply(cv2.GaussianBlur(roi_t,(5,5),0))
            roi_g = clahe.apply(cv2.GaussianBlur(roi_g,(5,5),0))

            ssim_score = ssim(roi_t, roi_g)
            e1 = cv2.Canny(roi_t,20,60)
            e2 = cv2.Canny(roi_g,20,60)
            edge_score = ssim(e1, e2)

            score = 0.7*ssim_score + 0.3*edge_score

            status = "OK" if score>=threshold else "NG"
            color = (0,255,0) if status=="OK" else (0,0,255)

        cv2.rectangle(annotated,(x1p,y1p),(x2p,y2p),color,2)
        text = f"{name} {score:.2f}" if score else f"{name} ?"
        cv2.putText(annotated,text,(x1p,y1p-3),0,0.6,color,2)

        report.append({
            "class":name,
            "bbox":(x1p,y1p,x2p,y2p),
            "ssim":score,
            "status":status
        })

    return annotated, report


def diff_visualize(golden_img, test_img, bbox=None, save_prefix="debug", show=False):
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)

    g = golden_img.copy()
    t = test_img.copy()
    if bbox:
        x1,y1,x2,y2 = bbox
        g = g[y1:y2, x1:x2]
        t = t[y1:y2, x1:x2]

    g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

    ssim_score, ssim_map = ssim(g_gray, t_gray, full=True)
    diff_map = (1.0-ssim_map)*255
    diff_8u = diff_map.astype(np.uint8)
    ssim_heat = cv2.applyColorMap(diff_8u, cv2.COLORMAP_JET)
    ssim_overlay = cv2.addWeighted(t,0.6,ssim_heat,0.4,0)

    abs_heat = cv2.applyColorMap(cv2.absdiff(g_gray,t_gray), cv2.COLORMAP_MAGMA)
    abs_overlay = cv2.addWeighted(t,0.6,abs_heat,0.4,0)

    e1 = cv2.Canny(g_gray,50,150)
    e2 = cv2.Canny(t_gray,50,150)
    edge_heat = cv2.applyColorMap(cv2.absdiff(e1,e2), cv2.COLORMAP_TURBO)
    edge_overlay = cv2.addWeighted(t,0.6,edge_heat,0.4,0)

    paths = {
        "ssim_score": float(ssim_score),
        "ssim_overlay": f"{save_prefix}_ssim_overlay.png",
        "abs_overlay": f"{save_prefix}_abs_overlay.png",
        "edge_overlay": f"{save_prefix}_edge_overlay.png"
    }

    cv2.imwrite(paths["ssim_overlay"], ssim_overlay)
    cv2.imwrite(paths["abs_overlay"], abs_overlay)
    cv2.imwrite(paths["edge_overlay"], edge_overlay)

    if show:
        for p in paths.values():
            if p.endswith(".png"):
                img = cv2.imread(p)
                cv2.imshow(p, img)
        cv2.waitKey(0)

    return paths


# ============================================================
# 5. ALIGN SHIFT CHECK (ORB / SIFT)
# ============================================================

def check_shift_orb(golden_img, test_img, bbox, px_tol=6, deg_tol=5):
    x1,y1,x2,y2 = bbox
    roi_g = golden_img[y1:y2, x1:x2]
    roi_t = test_img[y1:y2, x1:x2]

    g = cv2.GaussianBlur(cv2.cvtColor(roi_g,0),(3,3),0)
    t = cv2.GaussianBlur(cv2.cvtColor(roi_t,0),(3,3),0)

    orb = cv2.ORB_create(800)
    kp1, des1 = orb.detectAndCompute(g,None)
    kp2, des2 = orb.detectAndCompute(t,None)

    if des1 is None or des2 is None:
        return {"status":"NG","reason":"no_features"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good = sorted(bf.match(des1,des2), key=lambda m:m.distance)[:40]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H,_ = cv2.estimateAffinePartial2D(pts1,pts2)
    if H is None:
        return {"status":"NG","reason":"no_H"}

    dx = float(H[0,2]); dy = float(H[1,2])
    theta = float(np.degrees(np.arctan2(H[1,0],H[0,0])))

    ok = abs(dx)<=px_tol and abs(dy)<=px_tol and abs(theta)<=deg_tol

    return {
        "status":"OK" if ok else "NG",
        "dx": round(dx,2),
        "dy": round(dy,2),
        "theta": round(theta,2),
        "reason": None if ok else "misaligned"
    }


def check_shift_sift(golden_img, test_img, bbox, px_tol=20, deg_tol=20):
    x1,y1,x2,y2=bbox
    roi_g = golden_img[y1:y2, x1:x2]
    roi_t = test_img[y1:y2, x1:x2]

    g = cv2.GaussianBlur(cv2.cvtColor(roi_g,0),(3,3),0)
    t = cv2.GaussianBlur(cv2.cvtColor(roi_t,0),(3,3),0)

    sift = cv2.SIFT_create(3000)
    kp1,des1 = sift.detectAndCompute(g,None)
    kp2,des2 = sift.detectAndCompute(t,None)
    if des1 is None or des2 is None:
        return {"status":"NG","reason":"no_features"}

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1,des2,k=2)
    good = [m for m,n in matches if m.distance<0.75*n.distance]

    if len(good)<8:
        return {"status":"NG","reason":"few_matches"}

    pts1=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H,_=cv2.estimateAffinePartial2D(pts1,pts2,method=cv2.RANSAC)
    if H is None:
        return {"status":"NG","reason":"no_H"}

    dx=float(H[0,2]); dy=float(H[1,2])
    theta=float(np.degrees(np.arctan2(H[1,0],H[0,0])))

    ok = abs(dx)<=px_tol and abs(dy)<=px_tol and abs(theta)<=deg_tol

    return {
        "status":"OK" if ok else "NG",
        "dx":round(dx,2),
        "dy":round(dy,2),
        "theta":round(theta,2),
        "reason":None if ok else "misaligned"
    }


def crop_pcb_from_blue(aligned_img, debug=False):
    b, g, r = cv2.split(aligned_img)

    # Mask theo màu xanh đậm
    mask = (b > 110).astype(np.uint8) * 255
    mask &= ((b.astype(int) - r.astype(int)) > 40).astype(np.uint8) * 255

    # Morphology làm sạch mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Không tìm thấy PCB")
        return aligned_img

    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cropped = aligned_img[y:y+h, x:x+w]

    if debug:
        debug_img = aligned_img.copy()
        cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,0,255), 4)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.show()

    return cropped


