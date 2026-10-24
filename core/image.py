"""Frames for Gemma with no required third-party package. A frame is either a numpy BGR array (robots with OpenCV) or JPEG
bytes (anything else). With OpenCV installed both work the same; without it, JPEG bytes are resized and cropped by macOS
`sips` (the Mac mock). 480x336 is llama.cpp's floor for Gemma 4: 70 image tokens (docs/LATENCY_RESULTS.md)."""
import os, struct, subprocess, tempfile
try:
    import cv2, numpy as np
except ImportError:
    cv2 = np = None

SIZE = (480, 336)


def is_array(img): return np is not None and isinstance(img, np.ndarray)


def jpeg_size(data):
    """(width, height) of JPEG bytes, read from the SOF marker."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF: i += 1; continue
        m = data[i + 1]
        if m == 0xFF: i += 1; continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7: i += 2; continue
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9]); return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    raise ValueError("not a JPEG")


def size(img): return (img.shape[1], img.shape[0]) if is_array(img) else jpeg_size(img)


def decode(data): return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _sips(data, *args):
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.jpg"), os.path.join(d, "out.jpg")
        with open(src, "wb") as f: f.write(data)
        subprocess.run(["sips", *args, src, "--out", dst, "-s", "format", "jpeg"], check=True, capture_output=True)
        with open(dst, "rb") as f: return f.read()


def to_jpeg(img, size=SIZE, quality=85):
    """JPEG bytes of a frame, resized to `size` (None keeps the frame's size)."""
    if is_array(img):
        if size: img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    if not size or jpeg_size(img) == tuple(size): return img
    if cv2 is not None: return to_jpeg(decode(img), size, quality)
    return _sips(img, "-z", str(size[1]), str(size[0]))


def crop(img, box, margin=10):
    """The box (x1, y1, x2, y2 pixels) plus `margin` pixels, clipped to the frame; the G1 brain's crop (10 px). Same type as input."""
    W, H = size(img); x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, int(x1 - margin)), max(0, int(y1 - margin)), min(W, int(x2 + margin)), min(H, int(y2 + margin))
    if is_array(img): return img[y1:y2, x1:x2].copy()
    if cv2 is not None: return to_jpeg(decode(img)[y1:y2, x1:x2], None)
    return _sips(img, "-c", str(y2 - y1), str(x2 - x1), "--cropOffset", str(y1), str(x1))
