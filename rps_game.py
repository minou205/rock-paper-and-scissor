import cv2
import time
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

# ═══════════════════════════════════════════════════════════════════════════
#  Settings
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PATH  = "rps_model_v2.keras"
CLASSES     = ["rock", "paper", "scissors"]
IMG_SIZE    = (224, 224)
CONF_MIN    = 0.60
PAD_RATIO   = 0.20
COUNTDOWN   = 3
RESULT_HOLD = 4.0
FLASH_DUR   = 0.5

# Skin detection thresholds (YCrCb color space — robust across lighting)
SKIN_LOWER = np.array([0,   133,  77], dtype=np.uint8)
SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)
MIN_HAND_AREA = 3000   # px² — ignore tiny blobs

COL = {
    "p1":     ( 30, 140, 255),
    "p2":     ( 50,  50, 230),
    "green":  ( 60, 200,  80),
    "yellow": (  0, 215, 240),
    "white":  (255, 255, 255),
    "dark":   ( 15,  15,  15),
    "silver": (190, 190, 190),
}
FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX

# ═══════════════════════════════════════════════════════════════════════════
#  Load model
# ═══════════════════════════════════════════════════════════════════════════
print("⏳  Loading model...")
model = load_model(MODEL_PATH)
print(f"✅  {MODEL_PATH} ready\n")

# Morphological kernel for skin mask cleanup
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


# ═══════════════════════════════════════════════════════════════════════════
#  Hand detection (OpenCV skin-color, no MediaPipe / no extra model)
# ═══════════════════════════════════════════════════════════════════════════

def detect_hand_bbox(half: np.ndarray):
    """
    Returns (x1, y1, x2, y2) bounding box of the largest skin blob,
    or None if no hand is found.
    """
    H, W = half.shape[:2]

    # --- skin mask in YCrCb ---
    ycrcb = cv2.cvtColor(half, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb, SKIN_LOWER, SKIN_UPPER)

    # --- clean up noise ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    mask = cv2.dilate(mask, _KERNEL, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_HAND_AREA:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    pw = int(bw * PAD_RATIO)
    ph = int(bh * PAD_RATIO)
    x1 = max(0, x - pw);      y1 = max(0, y - ph)
    x2 = min(W, x + bw + pw); y2 = min(H, y + bh + ph)
    return x1, y1, x2, y2


def hand_to_input(half: np.ndarray, bbox):
    """
    Crop bbox → square letter-box → 224×224 → preprocess_input (same as training).
    Returns (tensor, bbox) or (None, None).
    """
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    crop = half[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    ch, cw   = crop.shape[:2]
    side     = max(ch, cw)
    canvas   = np.zeros((side, side, 3), dtype=np.uint8)
    yo = (side - ch) // 2
    xo = (side - cw) // 2
    canvas[yo:yo+ch, xo:xo+cw] = crop

    rgb    = cv2.cvtColor(cv2.resize(canvas, IMG_SIZE), cv2.COLOR_BGR2RGB)
    tensor = preprocess_input(rgb.astype("float32"))
    return tensor, bbox


def predict(tensor):
    if tensor is None:
        return None, 0.0
    probs = model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    return (CLASSES[idx] if conf >= CONF_MIN else None), conf


def decide(g1, g2):
    if g1 is None or g2 is None:
        return -1
    if g1 == g2:
        return 0
    wins = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}
    return 1 if (g1, g2) in wins else 2


# ═══════════════════════════════════════════════════════════════════════════
#  HUD helpers
# ═══════════════════════════════════════════════════════════════════════════

def alpha_fill(frame, x1, y1, x2, y2, color, a=0.72):
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, a, frame, 1 - a, 0, frame)


def text_cx(frame, text, cy, fnt, sc, col, th, w):
    (tw, fh), _ = cv2.getTextSize(text, fnt, sc, th)
    cv2.putText(frame, text, (w // 2 - tw // 2, cy + fh // 2),
                fnt, sc, col, th, cv2.LINE_AA)


def draw_topbar(frame, s1, s2, w):
    alpha_fill(frame, 0, 0, w, 62, COL["dark"], 0.82)
    cv2.rectangle(frame, (0, 60), (w, 62), COL["silver"], -1)
    cv2.putText(frame, f"P1  {s1}", (18, 44),
                FONT, 1.15, COL["p1"], 2, cv2.LINE_AA)
    lbl = f"P2  {s2}"
    (tw, _), _ = cv2.getTextSize(lbl, FONT, 1.15, 2)
    cv2.putText(frame, lbl, (w - tw - 18, 44),
                FONT, 1.15, COL["p2"], 2, cv2.LINE_AA)
    text_cx(frame, "ROCK   PAPER   SCISSORS", 35, FONTS, 0.70, COL["white"], 1, w)


def draw_divider(frame, mid, h):
    cv2.line(frame, (mid, 65), (mid, h - 32), COL["silver"], 2)


def draw_player_tags(frame, mid):
    cv2.putText(frame, "PLAYER 1", (16, 95),
                FONT, 0.85, COL["p1"], 2, cv2.LINE_AA)
    cv2.putText(frame, "PLAYER 2", (mid + 16, 95),
                FONT, 0.85, COL["p2"], 2, cv2.LINE_AA)


def draw_live(frame, g, c, side, mid, h):
    x = 12 if side == "L" else mid + 12
    if g:
        lbl   = f"{g.upper()}  {c*100:.0f}%"
        color = COL["p1"] if side == "L" else COL["p2"]
    else:
        lbl, color = "No hand", (80, 80, 80)
    cv2.putText(frame, lbl, (x, h - 40), FONT, 0.80, color, 2, cv2.LINE_AA)


def draw_controls(frame, w, h):
    txt = "[SPACE] New Round    [R] Reset    [Q] Quit"
    (tw, _), _ = cv2.getTextSize(txt, FONTS, 0.50, 1)
    cv2.putText(frame, txt, (w // 2 - tw // 2, h - 8),
                FONTS, 0.50, (110, 110, 110), 1, cv2.LINE_AA)


def draw_bbox(half, bbox, color):
    if bbox:
        cv2.rectangle(half, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


def draw_countdown(frame, n, w, h):
    cx, cy, r = w // 2, h // 2, 70
    alpha_fill(frame, cx - r - 12, cy - r - 12,
               cx + r + 12, cy + r + 12, COL["dark"], 0.80)
    cv2.circle(frame, (cx, cy), r, COL["yellow"], 4, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(str(n), FONTS, 3.5, 6)
    cv2.putText(frame, str(n), (cx - tw // 2, cy + th // 2),
                FONTS, 3.5, COL["yellow"], 6, cv2.LINE_AA)
    text_cx(frame, "GET READY!", cy + r + 38, FONT, 0.85, COL["white"], 1, w)


def draw_shoot(frame, w, h):
    text_cx(frame, "SHOOT!", h // 2 + 10, FONTS, 3.0, COL["green"], 6, w)


def draw_result(frame, wcode, g1, g2, w, h):
    g1s = (g1 or "???").upper()
    g2s = (g2 or "???").upper()

    if wcode == 1:
        msg, color = f"P1 WINS !    {g1s}  beats  {g2s}", COL["p1"]
    elif wcode == 2:
        msg, color = f"P2 WINS !    {g1s}  loses  {g2s}", COL["p2"]
    elif wcode == 0:
        msg, color = f"DRAW         {g1s}  =  {g2s}", COL["yellow"]
    else:
        msg, color = "Hands not detected – try again!", (100, 100, 100)

    bw, bh = min(w - 40, 680), 96
    bx, by = w // 2 - bw // 2, h // 2 - bh // 2
    alpha_fill(frame, bx - 5, by - 5, bx + bw + 5, by + bh + 5, COL["dark"], 0.82)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 3, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(msg, FONT, 1.0, 2)
    cv2.putText(frame, msg, (w // 2 - tw // 2, by + bh // 2 + th // 2),
                FONT, 1.0, color, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  Main game loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    s1 = s2   = 0
    state     = "idle"
    cd_start  = 0.0
    g1_snap   = g2_snap = None
    res_code  = -1
    phase_end = 0.0

    print("=== Game ready ===   Press SPACE to start\n")
    print("TIP: Make sure both hands are clearly visible on each side.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        mid   = w // 2
        now   = time.time()

        lhalf = frame[:, :mid].copy()
        rhalf = frame[:, mid:].copy()

        # ── Player 1 (left) ───────────────────────────────────────────────
        bb_l               = detect_hand_bbox(lhalf)
        t_l, bb_l          = hand_to_input(lhalf, bb_l)
        g_l, c_l           = predict(t_l)
        draw_bbox(lhalf, bb_l, COL["p1"])

        # ── Player 2 (right) ──────────────────────────────────────────────
        bb_r               = detect_hand_bbox(rhalf)
        t_r, bb_r          = hand_to_input(rhalf, bb_r)
        g_r, c_r           = predict(t_r)
        draw_bbox(rhalf, bb_r, COL["p2"])

        # ── Merge halves ──────────────────────────────────────────────────
        frame[:, :mid] = lhalf
        frame[:, mid:] = rhalf

        # ── Static HUD ────────────────────────────────────────────────────
        draw_topbar(frame, s1, s2, w)
        draw_divider(frame, mid, h)
        draw_player_tags(frame, mid)
        draw_live(frame, g_l, c_l, "L", mid, h)
        draw_live(frame, g_r, c_r, "R", mid, h)
        draw_controls(frame, w, h)

        # ── Game state machine ────────────────────────────────────────────
        if state == "countdown":
            elapsed   = now - cd_start
            remaining = COUNTDOWN - int(elapsed)
            if remaining > 0:
                draw_countdown(frame, remaining, w, h)
            else:
                g1_snap  = g_l
                g2_snap  = g_r
                res_code = decide(g1_snap, g2_snap)
                if res_code == 1:
                    s1 += 1
                elif res_code == 2:
                    s2 += 1
                state     = "shoot_flash"
                phase_end = now + FLASH_DUR

        elif state == "shoot_flash":
            draw_shoot(frame, w, h)
            if now >= phase_end:
                state     = "result"
                phase_end = now + RESULT_HOLD

        elif state == "result":
            draw_result(frame, res_code, g1_snap, g2_snap, w, h)
            if now >= phase_end:
                state = "idle"

        cv2.imshow("Rock · Paper · Scissors  –  AI Edition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and state == "idle":
            state    = "countdown"
            cd_start = time.time()
        elif key in (ord("r"), ord("R")):
            s1 = s2 = 0
            state   = "idle"
            print("🔄  Score reset")
        elif key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🏆  Final Score  →  P1: {s1}  |  P2: {s2}")
    print("Thanks for playing! 👋\n")


if __name__ == "__main__":
    main()