import json, math, zipfile

def write_submission(images, preds, size=512, out="submission.zip", boxes_are_original=False):
    """images: from images.json; preds[file_name] = (label,
    boxes [x1, y1, x2, y2] on the model input, classes, scores)"""
    ids = [int(im['id']) for im in images]
    if len(ids) != len(set(ids)):
        raise ValueError('images.json contains duplicate integer image IDs')
    rows, dets = ["image_id,label"], []
    for im in images:
        label, boxes, classes, scores = preds[im["file_name"]]
        if not 0 <= int(label) < 9:
            raise ValueError(f'Classification label for {im["file_name"]} must be in 0..8')
        rows.append(f"{im['id']},{int(label)}")  # the integer id
        for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
            if not all(math.isfinite(float(v)) for v in (x1, y1, x2, y2, s)):
                continue
            if boxes_are_original:
                x1, x2 = min(max(float(x1), 0.0), float(im['width'])), min(max(float(x2), 0.0), float(im['width']))
                y1, y2 = min(max(float(y1), 0.0), float(im['height'])), min(max(float(y2), 0.0), float(im['height']))
                box = [x1, y1, x2 - x1, y2 - y1]
            else:
                sx, sy = im["width"] / size, im["height"] / size
                box = [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]
            if box[2] <= 0 or box[3] <= 0 or not 0 <= int(c) < 9 or not 0 <= float(s) <= 1:
                continue
            dets.append({"image_id": im["id"],
                         "category_id": int(c) + 1,  # label + 1
                         "bbox": [float(v) for v in box],
                         "score": float(s)})
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("classification.csv", "\n".join(rows) + "\n")
        z.writestr("detection.json", json.dumps(dets))
