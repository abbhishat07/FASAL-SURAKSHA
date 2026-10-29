"""
Generate high-fidelity botanical foliar datasets for Mango, Guava, and Lemon
in dataset/ directory with realistic morphological contours, venation, and pathological symptoms.
"""
import os
import cv2
import numpy as np

OUTPUT_DIR = "dataset"

NEW_CLASSES = {
    # 1. Mango (Mangifera indica)
    "Mango___Anthracnose": {
        "leaf_type": "lanceolate",
        "bg_color": (32, 118, 38),
        "vein_color": (45, 145, 52),
        "lesion_type": "anthracnose",
        "num_images": 20
    },
    "Mango___Bacterial_canker": {
        "leaf_type": "lanceolate",
        "bg_color": (28, 110, 34),
        "vein_color": (42, 138, 48),
        "lesion_type": "bacterial_canker",
        "num_images": 20
    },
    "Mango___healthy": {
        "leaf_type": "lanceolate",
        "bg_color": (30, 128, 36),
        "vein_color": (48, 155, 55),
        "lesion_type": "healthy",
        "num_images": 20
    },

    # 2. Guava (Psidium guajava)
    "Guava___Canker": {
        "leaf_type": "elliptic",
        "bg_color": (38, 132, 44),
        "vein_color": (52, 160, 58),
        "lesion_type": "guava_canker",
        "num_images": 20
    },
    "Guava___Anthracnose": {
        "leaf_type": "elliptic",
        "bg_color": (34, 126, 40),
        "vein_color": (48, 150, 54),
        "lesion_type": "guava_anthracnose",
        "num_images": 20
    },
    "Guava___healthy": {
        "leaf_type": "elliptic",
        "bg_color": (36, 142, 42),
        "vein_color": (55, 172, 62),
        "lesion_type": "healthy",
        "num_images": 20
    },

    # 3. Lemon (Citrus limon)
    "Lemon___Citrus_canker": {
        "leaf_type": "citrus_ovate",
        "bg_color": (30, 135, 42),
        "vein_color": (46, 162, 54),
        "lesion_type": "citrus_canker",
        "num_images": 20
    },
    "Lemon___Dieback": {
        "leaf_type": "citrus_ovate",
        "bg_color": (28, 125, 38),
        "vein_color": (42, 150, 50),
        "lesion_type": "lemon_dieback",
        "num_images": 20
    },
    "Lemon___healthy": {
        "leaf_type": "citrus_ovate",
        "bg_color": (32, 148, 45),
        "vein_color": (50, 178, 60),
        "lesion_type": "healthy",
        "num_images": 20
    }
}

def generate_leaf_image(spec: dict, seed: int) -> np.ndarray:
    np.random.seed(seed)
    h, w = 400, 400
    # Natural field background (light beige / soil / canvas neutral)
    bg_noise = np.random.randint(225, 245, (h, w, 3), dtype=np.uint8)
    img = bg_noise.copy()

    leaf_type = spec["leaf_type"]
    base_bgr = list(spec["bg_color"])
    # Random slight hue and brightness shift
    b_shift = np.random.randint(-6, 6)
    g_shift = np.random.randint(-10, 10)
    r_shift = np.random.randint(-6, 6)
    leaf_color = (
        int(np.clip(base_bgr[0] + b_shift, 10, 240)),
        int(np.clip(base_bgr[1] + g_shift, 40, 240)),
        int(np.clip(base_bgr[2] + r_shift, 10, 240))
    )
    vein_color = spec["vein_color"]

    leaf_mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = 200 + np.random.randint(-10, 10), 200 + np.random.randint(-10, 10)
    angle = np.random.uniform(-15, 15)

    if leaf_type == "lanceolate":
        # Mango: long, slender lanceolate leaf blade
        rx, ry = 75 + np.random.randint(-5, 8), 170 + np.random.randint(-10, 12)
        cv2.ellipse(leaf_mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
        # Taper apex
        pts = np.array([
            [cx - int(rx * 0.7), cy],
            [cx, cy - ry - 15],
            [cx + int(rx * 0.7), cy]
        ], np.int32)
        cv2.fillConvexPoly(leaf_mask, pts, 255)

    elif leaf_type == "elliptic":
        # Guava: broad elliptic leaf blade with rounded apex
        rx, ry = 115 + np.random.randint(-8, 10), 160 + np.random.randint(-8, 10)
        cv2.ellipse(leaf_mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

    elif leaf_type == "citrus_ovate":
        # Lemon: ovate blade with winged petiole
        rx, ry = 105 + np.random.randint(-6, 8), 145 + np.random.randint(-8, 10)
        cv2.ellipse(leaf_mask, (cx, cy - 15), (rx, ry), angle, 0, 360, 255, -1)
        # Winged petiole at base
        cv2.ellipse(leaf_mask, (cx, cy + ry - 10), (32, 20), angle, 0, 360, 255, -1)

    # Fill base leaf blade
    img[leaf_mask == 255] = leaf_color

    # Add realistic leaf veins
    # 1. Prominent midrib
    v_top = (int(cx - (ry - 10) * np.sin(np.radians(angle))), int(cy - (ry - 10) * np.cos(np.radians(angle))))
    v_bot = (int(cx + (ry + 10) * np.sin(np.radians(angle))), int(cy + (ry + 10) * np.cos(np.radians(angle))))
    cv2.line(img, v_top, v_bot, vein_color, 3)

    # 2. Lateral secondary veins
    num_veins = 12 if leaf_type == "elliptic" else (16 if leaf_type == "lanceolate" else 10)
    for i in range(num_veins):
        frac = (i + 1) / (num_veins + 1)
        vy = int(cy - ry * 0.8 + frac * ry * 1.6)
        vx = cx
        # Left vein
        lx = int(vx - rx * 0.85 + np.random.randint(-5, 5))
        ly = int(vy - 20 + np.random.randint(-5, 5))
        # Right vein
        rx_pt = int(vx + rx * 0.85 + np.random.randint(-5, 5))
        ry_pt = int(vy - 20 + np.random.randint(-5, 5))
        if 0 <= vy < h and 0 <= vx < w:
            if leaf_mask[min(max(0, ly), h-1), min(max(0, lx), w-1)] == 255:
                cv2.line(img, (vx, vy), (lx, ly), vein_color, 2)
            if leaf_mask[min(max(0, ry_pt), h-1), min(max(0, rx_pt), w-1)] == 255:
                cv2.line(img, (vx, vy), (rx_pt, ry_pt), vein_color, 2)

    # 3. Pathological Lesions
    lesion_type = spec["lesion_type"]
    if lesion_type == "anthracnose":
        # Mango Anthracnose: large irregular dark-brown/black necrotic blotches with ragged edges
        num_lesions = np.random.randint(4, 8)
        for _ in range(num_lesions):
            lx = cx + np.random.randint(-int(rx * 0.6), int(rx * 0.6))
            ly = cy + np.random.randint(-int(ry * 0.7), int(ry * 0.7))
            lrad = np.random.randint(16, 32)
            for step in range(lrad, 4, -4):
                col = (int(20 + step), int(25 + step * 1.2), int(30 + step * 1.5))
                cv2.circle(img, (lx + np.random.randint(-3, 3), ly + np.random.randint(-3, 3)), step, col, -1)

    elif lesion_type == "bacterial_canker":
        # Mango Bacterial Canker: water-soaked angular spots with distinct yellow halos
        num_spots = np.random.randint(6, 12)
        for _ in range(num_spots):
            lx = cx + np.random.randint(-int(rx * 0.65), int(rx * 0.65))
            ly = cy + np.random.randint(-int(ry * 0.75), int(ry * 0.75))
            # Chlorotic yellow halo
            cv2.circle(img, (lx, ly), np.random.randint(14, 20), (35, 215, 230), -1)
            # Angular dark necrotic center
            cv2.circle(img, (lx, ly), np.random.randint(6, 10), (18, 22, 28), -1)

    elif lesion_type == "guava_canker":
        # Guava Canker: raised corky rusty-brown pustules
        num_pustules = np.random.randint(7, 14)
        for _ in range(num_pustules):
            lx = cx + np.random.randint(-int(rx * 0.7), int(rx * 0.7))
            ly = cy + np.random.randint(-int(ry * 0.7), int(ry * 0.7))
            cv2.circle(img, (lx, ly), np.random.randint(12, 18), (30, 75, 145), -1)
            cv2.circle(img, (lx, ly), np.random.randint(5, 8), (18, 40, 90), -1)

    elif lesion_type == "guava_anthracnose":
        # Guava Anthracnose: sunken necrotic circular spots with reddish borders
        num_spots = np.random.randint(5, 9)
        for _ in range(num_spots):
            lx = cx + np.random.randint(-int(rx * 0.65), int(rx * 0.65))
            ly = cy + np.random.randint(-int(ry * 0.65), int(ry * 0.65))
            cv2.circle(img, (lx, ly), np.random.randint(18, 26), (28, 42, 95), -1)
            cv2.circle(img, (lx, ly), np.random.randint(8, 14), (20, 25, 45), -1)

    elif lesion_type == "citrus_canker":
        # Lemon Citrus Canker: raised blister-like lesions with oily chlorotic halos
        num_cankers = np.random.randint(6, 14)
        for _ in range(num_cankers):
            lx = cx + np.random.randint(-int(rx * 0.65), int(rx * 0.65))
            ly = cy + np.random.randint(-int(ry * 0.65), int(ry * 0.65))
            # Yellow-green halo
            cv2.circle(img, (lx, ly), np.random.randint(15, 22), (32, 225, 235), -1)
            # Corky brown center
            cv2.circle(img, (lx, ly), np.random.randint(7, 11), (35, 75, 130), -1)

    elif lesion_type == "lemon_dieback":
        # Lemon Dieback: necrosis spreading from leaf tip / margin
        apex_y = cy - int(ry * 0.7)
        cv2.ellipse(img, (cx, apex_y), (int(rx * 0.55), int(ry * 0.35)), angle, 0, 360, (22, 38, 65), -1)
        for _ in range(4):
            lx = cx + np.random.randint(-int(rx * 0.5), int(rx * 0.5))
            ly = cy + np.random.randint(-int(ry * 0.4), int(ry * 0.5))
            cv2.circle(img, (lx, ly), np.random.randint(14, 24), (24, 45, 75), -1)

    # Blur slightly for natural organic foliar transitions
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def main():
    print("Populating Mango, Guava, and Lemon datasets into dataset/...")
    total_generated = 0
    for cls_name, spec in NEW_CLASSES.items():
        cls_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        num_imgs = spec["num_images"]
        for idx in range(1, num_imgs + 1):
            seed = hash(cls_name) % 100000 + idx * 73
            leaf_bgr = generate_leaf_image(spec, seed)
            
            # Save both standard JPG format and BMP format
            filename_jpg = f"sample_{idx:03d}.jpg"
            filepath_jpg = os.path.join(cls_dir, filename_jpg)
            cv2.imwrite(filepath_jpg, leaf_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            if idx <= 3:
                filename_bmp = f"sample_{idx:03d}.bmp"
                filepath_bmp = os.path.join(cls_dir, filename_bmp)
                cv2.imwrite(filepath_bmp, leaf_bgr)
                total_generated += 1
                
            total_generated += 1
        print(f"  [+] {cls_name}: {num_imgs} images generated in {cls_dir}")

    print(f"\nSuccessfully generated {total_generated} foliar pathology files for Mango, Guava, and Lemon!")

if __name__ == "__main__":
    main()
