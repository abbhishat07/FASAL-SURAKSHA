"""
Generate sample BMP/JPEG image files in dataset/ using pure Python (no dependencies required).
"""
import os
import struct

def write_bmp(filename: str, width: int, height: int, pixels: list):
    # BMP 24-bit RGB
    row_padding = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + row_padding) * height
    file_size = 54 + image_size

    # BMP Header
    bmp_header = struct.pack(
        "<2sIHHI",
        b"BM",
        file_size,
        0, 0,
        54
    )
    # DIB Header (BITMAPINFOHEADER)
    dib_header = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        image_size,
        2835, 2835,
        0, 0
    )

    with open(filename, "wb") as f:
        f.write(bmp_header)
        f.write(dib_header)
        # BMP pixels are stored bottom-to-top, BGR order
        for y in range(height - 1, -1, -1):
            for x in range(width):
                r, g, b = pixels[y * width + x]
                f.write(struct.pack("BBB", b, g, r))
            f.write(b"\x00" * row_padding)

classes = [
    ("Tomato___Late_blight", (40, 150, 40), [(70, 70, 30, (80, 45, 25)), (110, 100, 20, (90, 50, 30))]),
    ("Tomato___healthy", (30, 160, 35), []),
    ("Potato___Early_blight", (45, 140, 45), [(75, 75, 25, (100, 60, 20))]),
    ("Corn_(maize)___Common_rust_", (60, 170, 70), [(60, 60, 15, (140, 70, 20)), (90, 100, 15, (140, 70, 20))]),
    ("Rice___Leaf_Blast", (45, 150, 50), [(80, 80, 20, (80, 55, 30))]),
    ("Mango___Anthracnose", (35, 125, 38), [(75, 75, 25, (25, 30, 35)), (110, 110, 20, (20, 25, 30))]),
    ("Mango___Bacterial_canker", (30, 120, 35), [(80, 80, 18, (220, 210, 35)), (80, 80, 8, (25, 20, 20))]),
    ("Mango___healthy", (32, 135, 36), []),
    ("Guava___Canker", (40, 135, 45), [(80, 75, 20, (140, 75, 30)), (115, 105, 15, (130, 70, 25))]),
    ("Guava___Anthracnose", (38, 130, 42), [(75, 80, 22, (75, 40, 30))]),
    ("Guava___healthy", (40, 145, 45), []),
    ("Lemon___Citrus_canker", (35, 140, 45), [(75, 75, 20, (230, 220, 35)), (75, 75, 10, (120, 75, 35))]),
    ("Lemon___Dieback", (30, 130, 40), [(80, 40, 30, (65, 35, 25))]),
    ("Lemon___healthy", (35, 150, 45), []),
]

width, height = 160, 160

for cls_name, bg_color, lesions in classes:
    cls_dir = os.path.join("dataset", cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    for idx in range(1, 4):
        pixels = []
        for y in range(height):
            for x in range(width):
                # Circular leaf shape
                dx = x - width // 2
                dy = y - height // 2
                dist_sq = dx * dx + dy * dy
                if dist_sq < 60 * 60:
                    r, g, b = bg_color
                    # Check lesions
                    for lx, ly, lr, lcol in lesions:
                        ldx = x - lx
                        ldy = y - ly
                        if ldx * ldx + ldy * ldy < lr * lr:
                            r, g, b = lcol
                            break
                    pixels.append((r, g, b))
                else:
                    pixels.append((255, 255, 255)) # White background
        filepath = os.path.join(cls_dir, f"sample_leaf_{idx}.bmp")
        write_bmp(filepath, width, height, pixels)
        print(f"Created: {filepath}")

print("Dataset successfully populated with sample micro-crop classes!")
