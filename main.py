import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time

# ── 1. MediaPipe Setup ───────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ── 2. Camera Setup ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting... Pinch thumb & index finger to control volume. Press 'q' to quit.")

# ── 3. Volume State ──────────────────────────────────────────────────────────
vol_percent = 50          # track our estimated volume (0-100)
prev_distance = None
last_vol_time = time.time()
VOL_COOLDOWN = 0.08       # seconds between volume key presses

def set_volume_to(target_percent, current_percent):
    """Press volume up/down keys to reach target from current."""
    diff = target_percent - current_percent
    steps = int(abs(diff) / 2)   # each key press = ~2%
    if steps == 0:
        return current_percent
    if diff > 0:
        for _ in range(steps):
            pyautogui.press('volumeup')
    else:
        for _ in range(steps):
            pyautogui.press('volumedown')
    return target_percent

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # ── 4. Process Frame ─────────────────────────────────────────────────────
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB.flags.writeable = False
    results = hands.process(imgRGB)
    imgRGB.flags.writeable = True

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            lm = handLms.landmark

            # Thumb tip = 4, Index tip = 8
            thumb_x, thumb_y = int(lm[4].x * w), int(lm[4].y * h)
            index_x, index_y = int(lm[8].x * w), int(lm[8].y * h)
            mid_x,   mid_y   = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2

            # Draw circles and connecting line
            cv2.circle(img, (thumb_x, thumb_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 3)

            # ── 5. Distance → Volume ──────────────────────────────────────
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            target_vol = int(np.interp(distance, [20, 200], [0, 100]))

            now = time.time()
            if now - last_vol_time > VOL_COOLDOWN:
                vol_percent = set_volume_to(target_vol, vol_percent)
                last_vol_time = now

            # Pinch indicator
            if distance < 40:
                cv2.circle(img, (mid_x, mid_y), 12, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "MUTE", (mid_x - 25, mid_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.circle(img, (mid_x, mid_y), 8, (255, 0, 255), cv2.FILLED)

            # ── 6. Volume Bar ─────────────────────────────────────────────
            bar_x, bar_y, bar_w, bar_h = 30, 100, 30, 280

            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), cv2.FILLED)

            fill_y = int(np.interp(vol_percent, [0, 100], [bar_y + bar_h, bar_y]))
            cv2.rectangle(img, (bar_x, fill_y),
                          (bar_x + bar_w, bar_y + bar_h), (0, 215, 255), cv2.FILLED)

            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

            cv2.putText(img, f'{vol_percent}%',
                        (bar_x - 5, bar_y + bar_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

    # ── 7. UI Labels ─────────────────────────────────────────────────────────
    cv2.putText(img, "Gesture Volume Control", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "Pinch fingers to adjust volume", (130, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── 8. Cleanup ───────────────────────────────────────────────────────────────
hands.close()
cap.release()
cv2.destroyAllWindows()