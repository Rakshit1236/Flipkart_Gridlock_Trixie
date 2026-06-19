import cv2
import easyocr
import re

reader = easyocr.Reader(['en'], gpu=False)
cap = cv2.VideoCapture(r'C:\Users\raksh\Desktop\PROJECT\Flipkart\DEMo\test_video.mp4')

for target_frame in [100, 300, 500, 700]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    scale = 640 / w
    resized = cv2.resize(frame, (640, int(h * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    results = reader.readtext(enhanced, detail=1, paragraph=False,
                              text_threshold=0.3, low_text=0.3)

    print(f'\nFrame {target_frame}: {len(results)} text regions')
    for bbox, text, conf in results:
        cleaned = text.upper().strip()
        cleaned2 = re.sub(r'[\s\-\.\,]', '', cleaned)
        plate_match = bool(re.match(r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$', cleaned2))
        print(f'  "{text}" (conf={conf:.2f}) cleaned="{cleaned2}" plate_match={plate_match}')

cap.release()
