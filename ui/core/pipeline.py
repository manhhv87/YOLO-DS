import cv2
import os

from core.algorithm import (
    orb_align_images,
    apply_clahe_bgr,
    sift_align_images,
    crop_pcb_from_color, crop_pcb_from_blue,
    paste_crop_to_canvas,
    yolo_infer,
    check_shift_sift,
    crop_by_points, orb_align_images,

)

def main_pipeline(image_path):
    import os
    import cv2
    import json
    from datetime import datetime

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # Danh sách class YOLO
    CLASS_NAMES = [
        "cap_NG", "cap_OK",
        "cd4017_NG", "cd4017_OK",
        "ne555_NG", "ne555_OK",
        "res_NG", "res_OK"
    ]

    # ==========================
    # 1) LOAD RAW IMAGE
    # ==========================
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise Exception(f"Không đọc được ảnh: {image_path}")

    # ==========================
    # 2) LOAD GOLDEN + ALIGN
    # ==========================
    golden_path = os.path.join(BASE_DIR, "stored_images/golden/golden2.bmp")
    golden_img = cv2.imread(golden_path)
    if golden_img is None:
        raise Exception("Không đọc được golden2.bmp!")

    # CLAHE trước align
    raw_img = apply_clahe_bgr(raw_img, clip=2.0, grid=(8, 8))
    golden_img = apply_clahe_bgr(golden_img, clip=2.0, grid=(8, 8))

    print("Align ảnh…")
    aligned_img, H = orb_align_images(golden_img, raw_img,2000, draw_matches=False)

    # ==========================
    # 3) CROP → CANVAS
    # ==========================
    #cropped_img = crop_pcb_from_blue(aligned_img, debug=False)
    cropped_img = crop_by_points(aligned_img, 158, 862, 2427, 1730)
    #final_img = paste_crop_to_canvas(cropped_img, debug=False)
    print("Paste canvas…")
    final_img = paste_crop_to_canvas(cropped_img, debug=False)

    # Lưu ảnh final
    fname = os.path.basename(image_path)
    final_path = os.path.join(BASE_DIR, "stored_images/final_images", fname)
    cv2.imwrite(final_path, final_img)

    # ==========================
    # 4) YOLO INFER
    # ==========================
    print("YOLO infer…")

    model_path = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")
    infer_folder = os.path.join(BASE_DIR, "stored_images/infer_images")

    results, _ = yolo_infer(
        model_path=model_path,
        img_path=final_path,
        save_folder=infer_folder,
        debug=False
    )

    det_raw = results[0].boxes.data

    # ==========================
    # 5) FILTER BOXES + CLASS
    # ==========================
    CONF_TH = 0.75
    det_boxes = []

    for b in det_raw:
        x1, y1, x2, y2, conf, cls = b.tolist()
        if conf < CONF_TH:
            continue

        cls_id = int(cls)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        det_boxes.append([
            int(x1), int(y1), int(x2), int(y2),
            float(conf), cls_id, cls_name
        ])

    print("Filtered det:", det_boxes)

    # ==========================
    # 6) CHECK MISSING + NG CLASS + SHIFT
    # ==========================
    golden_json = os.path.join(BASE_DIR, "golden_points.json")
    with open(golden_json, "r") as f:
        GOLDEN_POINTS = json.load(f)

    missing_list = []
    ng_components = []
    shift_results = []
    shift_ng = None

    for g in GOLDEN_POINTS:
        cx, cy = g["cx"], g["cy"]

        matched_boxes = []
        has_ok = False
        ng_cls_item = None
        ok_family = None
        ok_bbox = None

        for (x1, y1, x2, y2, conf, cls_id, cls_name) in det_boxes:
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue

            matched_boxes.append([x1, y1, x2, y2, cls_name])

            if cls_name.endswith("_OK"):
                has_ok = True
                ok_family = cls_id // 2     # classes are (NG, OK) pairs per family
                ok_bbox = [x1, y1, x2, y2]

            elif cls_name.endswith("_NG"):
                ng_cls_item = {
                    "cx": cx, "cy": cy,
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name
                }

        # Missing
        if not matched_boxes:
            missing_list.append(g)
            continue

        # NG class — any NG detection fails the component, even if an OK box of
        # the same component also overlaps (NG is dominant, so a defect cannot be
        # masked by a co-located OK detection).
        if ng_cls_item:
            ng_components.append(ng_cls_item)
            continue

        # Wrong component — an OK box whose component family differs from the one
        # the golden expects (g["cls"]//2) means the wrong part was placed.
        expected_family = g.get("cls")
        if expected_family is not None and ok_family is not None \
                and ok_family != expected_family // 2:
            ng_components.append({
                "cx": cx, "cy": cy,
                "bbox": ok_bbox,
                "class": "WRONG_COMPONENT",
            })
            continue

        # Shift check
        if has_ok:
            for (x1, y1, x2, y2, cls_name) in matched_boxes:
                if cls_name.endswith("_OK"):
                    bbox = [x1, y1, x2, y2]
                    out = check_shift_sift(golden_img, final_img, bbox)

                    shift_results.append({
                        "cx": cx, "cy": cy,
                        "bbox": bbox,
                        "dx": out["dx"],
                        "dy": out["dy"],
                        "theta": out["theta"],
                        "status": out["status"]
                    })

                    if out["status"] == "NG":
                        shift_ng = out

    # ==========================
    # 7) FINAL RESULT
    # ==========================
    if missing_list:
        final_status = "NG"
        reason = "MISSING"
    elif ng_components:
        final_status = "NG"
        reason = "CLASS_NG"
    elif shift_ng:
        final_status = "NG"
        reason = "SHIFT_NG"
    else:
        final_status = "OK"
        reason = "OK"

    result_info = {
        "status": final_status,
        "reason": reason,
        "missing": missing_list,
        "ng_components": ng_components,
        "shift_results": shift_results
    }

    # ==========================
    # 8) DRAW RESULT IMAGE
    # ==========================
    result_img = final_img.copy()

    for (x1, y1, x2, y2, conf, cls_id, cls_name) in det_boxes:
        color = (0, 255, 0)
        text = "OK"
        if cls_name.endswith("_NG"):
            color = (0, 0, 255)
            text = "NG"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for m in missing_list:
        cx, cy = m["cx"], m["cy"]
        cv2.circle(result_img, (cx, cy), 20, (0, 0, 255), 3)
        cv2.putText(result_img, "MISSING", (cx - 40, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for s in shift_results:
        if s["status"] == "NG":
            x1, y1, x2, y2 = s["bbox"]
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(result_img, "SHIFT_NG", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ==========================
    # 9) SAVE RESULT
    # ==========================
    result_folder = os.path.join(BASE_DIR, "stored_images/result")
    os.makedirs(result_folder, exist_ok=True)

    # Lấy tên gốc của ảnh đầu vào
    input_name = os.path.basename(image_path)  # vd: 20251206_205355.bmp
    name_no_ext = os.path.splitext(input_name)[0]  # vd: 20251206_205355

    # Tạo tên file kết quả
    save_path = os.path.join(result_folder, f"result_{name_no_ext}.bmp")
    cv2.imwrite(save_path, result_img)

    print("✔ RESULT SAVED:", save_path)
    print("✔ FINAL:", final_status, "| Reason:", reason)

    return result_img, result_info

def main_pipeline_no_shift(image_path):
    import os
    import cv2
    import json
    from datetime import datetime

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # YOLO class names
    CLASS_NAMES = [
        "cap_NG", "cap_OK",
        "cd4017_NG", "cd4017_OK",
        "ne555_NG", "ne555_OK",
        "res_NG", "res_OK"
    ]

    # ==========================
    # 1) LOAD RAW IMAGE
    # ==========================
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise Exception(f"Không đọc được ảnh: {image_path}")

    # ==========================
    # 2) LOAD GOLDEN + ALIGN
    # ==========================
    golden_path = os.path.join(BASE_DIR, "stored_images/golden/golden2.bmp")
    golden_img = cv2.imread(golden_path)
    if golden_img is None:
        raise Exception("Không đọc được golden2.bmp!")

    # CLAHE before align — match main_pipeline so the live (no-shift) path feeds
    # the detector the same pixel distribution it was tuned on.
    raw_img = apply_clahe_bgr(raw_img, clip=2.0, grid=(8, 8))
    golden_img = apply_clahe_bgr(golden_img, clip=2.0, grid=(8, 8))

    aligned_img, H = orb_align_images(golden_img, raw_img,2000, draw_matches=False)

    # ==========================
    # 3) CROP → CANVAS
    # ==========================
    #cropped_img = crop_pcb_from_blue(aligned_img, debug=False)
    cropped_img = crop_by_points(aligned_img, 158, 862, 2427, 1730)
    final_img = paste_crop_to_canvas(cropped_img, debug=False)

    # Save intermediate result
    fname = os.path.basename(image_path)
    final_path = os.path.join(BASE_DIR, "stored_images/final_images", fname)
    cv2.imwrite(final_path, final_img)

    # ==========================
    # 4) YOLO INFER
    # ==========================
    model_path = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")
    infer_folder = os.path.join(BASE_DIR, "stored_images/infer_images")

    results, _ = yolo_infer(
        model_path=model_path,
        img_path=final_path,
        save_folder=infer_folder,
        debug=False
    )

    det_raw = results[0].boxes.data

    # ==========================
    # 5) FILTER BOXES
    # ==========================
    CONF_TH = 0.75
    det_boxes = []

    for b in det_raw:
        x1, y1, x2, y2, conf, cls = b.tolist()
        if conf < CONF_TH:
            continue

        cls_id = int(cls)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        det_boxes.append([
            int(x1), int(y1), int(x2), int(y2),
            float(conf), cls_id, cls_name
        ])

    # ==========================
    # 6) MISSING + CLASS_NG
    # ==========================
    golden_json = os.path.join(BASE_DIR, "golden_points.json")
    with open(golden_json, "r") as f:
        GOLDEN_POINTS = json.load(f)

    missing_list = []
    ng_components = []

    for g in GOLDEN_POINTS:
        cx, cy = g["cx"], g["cy"]

        matched = []
        is_ok = False
        ng_item = None
        ok_family = None
        ok_bbox = None

        for (x1, y1, x2, y2, conf, cls_id, cls_name) in det_boxes:
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue

            matched.append(cls_name)

            if cls_name.endswith("_OK"):
                is_ok = True
                ok_family = cls_id // 2     # classes are (NG, OK) pairs per family
                ok_bbox = [x1, y1, x2, y2]
            elif cls_name.endswith("_NG"):
                ng_item = {
                    "cx": cx, "cy": cy,
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name
                }

        # Missing
        if not matched:
            missing_list.append(g)
            continue

        # NG class — dominant over a co-located OK box (a defect cannot be masked
        # by an overlapping OK detection).
        if ng_item:
            ng_components.append(ng_item)
            continue

        # Wrong component — an OK box whose family differs from the one the golden
        # expects (g["cls"]//2) means the wrong part was placed there.
        expected_family = g.get("cls")
        if expected_family is not None and ok_family is not None \
                and ok_family != expected_family // 2:
            ng_components.append({
                "cx": cx, "cy": cy,
                "bbox": ok_bbox,
                "class": "WRONG_COMPONENT",
            })
            continue

    # ==========================
    # 7) FINAL RESULT
    # ==========================
    if missing_list:
        final_status = "NG"
        reason = "MISSING"
    elif ng_components:
        final_status = "NG"
        reason = "CLASS_NG"
    else:
        final_status = "OK"
        reason = "OK"

    result_info = {
        "status": final_status,
        "reason": reason,
        "missing": missing_list,
        "ng_components": ng_components
    }

    # ==========================
    # 8) DRAW RESULT IMAGE
    # ==========================
    result_img = final_img.copy()

    for (x1, y1, x2, y2, conf, cls_id, cls_name) in det_boxes:
        color = (0, 255, 0)
        text = "OK"
        if cls_name.endswith("_NG"):
            color = (0, 0, 255)
            text = "NG"

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for m in missing_list:
        cx, cy = m["cx"], m["cy"]
        cv2.circle(result_img, (cx, cy), 20, (0, 0, 255), 3)
        cv2.putText(result_img, "MISSING", (cx - 40, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ==========================
    # 9) SAVE RESULT IMAGE
    # ==========================
    result_folder = os.path.join(BASE_DIR, "stored_images/result")
    os.makedirs(result_folder, exist_ok=True)

    # Lấy tên gốc của ảnh đầu vào
    input_name = os.path.basename(image_path)  # vd: 20251206_205355.bmp
    name_no_ext = os.path.splitext(input_name)[0]  # vd: 20251206_205355

    # Tạo tên file kết quả
    save_path = os.path.join(result_folder, f"result_{name_no_ext}.bmp")

    cv2.imwrite(save_path, result_img)

    print("✔ RESULT SAVED:", save_path)
    print("✔ FINAL:", final_status, "| Reason:", reason)

    return result_img, result_info


if __name__ == "__main__":

    ## dùng test pipeline
    main_pipeline("C:/Users/vuong/PycharmProjects/smd-detect-python39/stored_images/raw_images/20251208_165953.bmp")
    #main_pipeline_no_shift("C:/Users/vuong/PycharmProjects/smd-detect-python39/stored_images/raw_images/20251208_172312.bmp")
