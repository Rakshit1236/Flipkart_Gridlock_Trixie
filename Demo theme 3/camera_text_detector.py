"""
License Plate Detection Pipeline
Detects vehicle license plates using contour-based localization + EasyOCR.
Filters text to only show valid license plate patterns.
"""

import cv2
import easyocr
import numpy as np
import os
import sys
import re
import time
import threading
from datetime import datetime
from collections import deque


# Indian license plate patterns
PLATE_PATTERNS = [
    # Standard: MH12AB1234, DL01AA1234
    re.compile(r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$'),
    # BH series (new): BH12AB1234
    re.compile(r'^BH\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$'),
    # With separators: MH-12-AB-1234, MH 12 AB 1234
    re.compile(r'^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{1,4}$'),
    # CD/TC/ARMO etc. military: AD12A1234
    re.compile(r'^[A-Z]{2,4}\s?\d{1,2}\s?[A-Z]?\s?\d{1,4}$'),
]

# Aspect ratio range for license plates (width / height)
PLATE_ASPECT_MIN = 1.8
PLATE_ASPECT_MAX = 7.0

# Area range as fraction of frame area
PLATE_AREA_MIN = 0.001
PLATE_AREA_MAX = 0.20


class ThreadedCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


class LicensePlateDetector:
    def __init__(self, gpu=True, ocr_width=640, skip_frames=2,
                 confidence_threshold=0.4, headless=False):
        self.ocr_width = ocr_width
        self.skip_frames = skip_frames
        self.confidence_threshold = confidence_threshold
        self.headless = headless

        print("[INFO] Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.cap = None
        self.frame_count = 0
        print("[INFO] Reader ready.\n")

    def _find_plate_regions(self, frame):
        """Find candidate license plate regions using contour detection."""
        h, w = frame.shape[:2]
        frame_area = h * w

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(blur, 30, 200)

        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # License plates are roughly rectangular (3-8 vertices)
            if len(approx) >= 3 and len(approx) <= 8:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect = cw / max(ch, 1)
                area_ratio = (cw * ch) / frame_area

                if (PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX and
                        PLATE_AREA_MIN <= area_ratio <= PLATE_AREA_MAX):
                    # Add padding around the detected region
                    pad_x, pad_y = int(cw * 0.1), int(ch * 0.3)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + cw + pad_x)
                    y2 = min(h, y + ch + pad_y)
                    candidates.append((x1, y1, x2, y2))

        # Merge overlapping boxes
        return self._merge_boxes(candidates)

    def _merge_boxes(self, boxes):
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0]]

        for x1, y1, x2, y2 in boxes[1:]:
            lx1, ly1, lx2, ly2 = merged[-1]
            # Check overlap
            if x1 <= lx2 and x2 >= lx1 and y1 <= ly2 and y2 >= ly1:
                merged[-1] = (min(lx1, x1), min(ly1, y1),
                              max(lx2, x2), max(ly2, y2))
            else:
                merged.append((x1, y1, x2, y2))

        return merged

    def _is_valid_plate(self, text):
        """Check if text matches a license plate pattern."""
        cleaned = text.upper().strip()
        cleaned = re.sub(r'[\s\-\.\,]', '', cleaned)
        for pattern in PLATE_PATTERNS:
            if pattern.match(cleaned):
                return True
        return False

    def _preprocess_plate(self, plate_img):
        """Enhance plate region for better OCR accuracy."""
        # Resize for consistent OCR
        h, w = plate_img.shape[:2]
        target_w = 300
        scale = target_w / w
        resized = cv2.resize(plate_img, (target_w, int(h * scale)))

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Adaptive threshold for binarization
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.medianBlur(binary, 3)

        return denoised

    def detect_plates(self, frame):
        """Detect and read license plates in frame."""
        plate_regions = self._find_plate_regions(frame)
        detections = []

        for (x1, y1, x2, y2) in plate_regions:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            processed = self._preprocess_plate(roi)

            results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                text_threshold=0.7,
                low_text=0.3,
                contrast_ths=0.2,
                adjust_contrast=0.5
            )

            for (bbox, text, conf) in results:
                if conf < self.confidence_threshold:
                    continue

                cleaned = text.upper().strip()
                cleaned = re.sub(r'[\s\-\.\,]', '', cleaned)

                if self._is_valid_plate(cleaned):
                    scale_x = (x2 - x1) / 300
                    scale_y = (y2 - y1) / max(int((y2 - y1) * 300 / max(x2 - x1, 1)), 1)
                    frame_bbox = [
                        [int(x1 + p[0] * scale_x), int(y1 + p[1] * scale_y)]
                        for p in bbox
                    ]
                    detections.append((frame_bbox, cleaned, conf))

        # Fallback: if no plates found via contours, scan full frame
        if not detections:
            h, w = frame.shape[:2]
            scale = self.ocr_width / w
            resized = cv2.resize(frame, (self.ocr_width, int(h * scale)))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            results = self.reader.readtext(
                enhanced,
                detail=1,
                paragraph=False,
                text_threshold=0.5,
                low_text=0.4,
                contrast_ths=0.1,
                adjust_contrast=0.5
            )

            for (bbox, text, conf) in results:
                if conf < self.confidence_threshold:
                    continue
                cleaned = text.upper().strip()
                cleaned = re.sub(r'[\s\-\.\,]', '', cleaned)
                if self._is_valid_plate(cleaned):
                    frame_bbox = [
                        [int(p[0] / scale), int(p[1] / scale)]
                        for p in bbox
                    ]
                    detections.append((frame_bbox, cleaned, conf))

        return detections

    def draw_plates(self, frame, detections):
        """Draw license plate bounding boxes with distinct styling."""
        for (bbox, text, confidence) in detections:
            pts = np.array(bbox, dtype=np.int32)

            # Draw plate outline (yellow for plates)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 3)

            # Label background
            x, y = int(bbox[0][0]), int(bbox[0][1]) - 12
            if y < 20:
                y = int(bbox[2][1]) + 25

            label = f"PLATE: {text} ({confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x - 2, y - th - 6), (x + tw + 6, y + 4),
                          (0, 0, 0), -1)
            cv2.rectangle(frame, (x - 2, y - th - 6), (x + tw + 6, y + 4),
                          (0, 255, 255), 2)
            cv2.putText(frame, label, (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def run_live(self, save_dir=None, camera_index=0, max_frames=0):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print("[INFO] Starting camera...")
        self.cap = ThreadedCapture(camera_index)

        print("[INFO] Warming up OCR...")
        dummy = np.zeros((50, 150), dtype=np.uint8)
        self.reader.readtext(dummy)
        print("[INFO] Ready.")
        if self.headless:
            print(f"[INFO] Headless mode - processing {'all frames' if max_frames == 0 else max_frames} frames.\n")
        else:
            print("[INFO] Press 'q' to quit, 's' to save.\n")

        fps_deque = deque(maxlen=30)
        last_ocr_time = 0
        cached_detections = []
        self.frame_count = 0

        while True:
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            self.frame_count += 1

            if self.frame_count % self.skip_frames == 0:
                ocr_start = time.time()
                cached_detections = self.detect_plates(frame)
                last_ocr_time = time.time() - ocr_start

            display_frame = self.draw_plates(frame, cached_detections)

            fps = 1.0 / max(time.time() - loop_start, 1e-6)
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)

            # Status bar
            cv2.rectangle(display_frame, (0, 0), (300, 110), (0, 0, 0), -1)
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"OCR: {last_ocr_time*1000:.0f}ms",
                f"Plates found: {len(cached_detections)}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(display_frame, line, (8, 25 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if cached_detections:
                cv2.putText(display_frame, "LICENSE PLATE DETECTED!",
                            (10, 30 + len(info_lines) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not self.headless:
                cv2.imshow("License Plate Detector", display_frame)

            key = cv2.waitKey(1) & 0xFF if not self.headless else 0xFF
            if key == ord('q'):
                break
            elif max_frames > 0 and self.frame_count >= max_frames:
                print(f"[INFO] Reached {max_frames} frames limit.")
                break
            elif key == ord('s') and save_dir:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(save_dir, f"plate_{ts}.jpg")
                cv2.imwrite(path, display_frame)
                # Also save the plate regions cropped
                for i, (_, text, _) in enumerate(cached_detections):
                    plate_path = os.path.join(save_dir, f"plate_{ts}_{i}.jpg")
                    # Find the region again for saving
                print(f"[INFO] Saved: {path}")

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Done. Processed {self.frame_count} frames.")

    def capture_single(self, save_path=None, camera_index=0):
        self.cap = ThreadedCapture(camera_index)
        time.sleep(0.5)

        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Failed to capture.")
            self.cap.release()
            return None

        detections = self.detect_plates(frame)
        display_frame = self.draw_plates(frame, detections)

        print(f"\n{'='*50}")
        print(f"Detected {len(detections)} license plate(s):")
        print(f"{'='*50}")
        for i, (_, text, conf) in enumerate(detections, 1):
            print(f"  {i}. {text} (confidence: {conf:.0%})")
        if not detections:
            print("  No license plates found.")
        print(f"{'='*50}\n")

        if save_path:
            cv2.imwrite(save_path, display_frame)
            print(f"[INFO] Saved: {save_path}")

        if not self.headless:
            cv2.imshow("License Plate Detected", display_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self.cap.release()
        return detections

    def process_image(self, image_path, save_path=None):
        if not os.path.exists(image_path):
            print(f"[ERROR] Not found: {image_path}")
            return None

        frame = cv2.imread(image_path)
        detections = self.detect_plates(frame)
        display_frame = self.draw_plates(frame, detections)

        print(f"\nDetected {len(detections)} license plate(s):")
        for i, (_, text, conf) in enumerate(detections, 1):
            print(f"  {i}. {text} (confidence: {conf:.0%})")

        if save_path:
            cv2.imwrite(save_path, display_frame)
            print(f"[INFO] Saved: {save_path}")

        if not self.headless:
            cv2.imshow("License Plate Detected", display_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return detections

    def process_video(self, video_path, save_dir=None, max_frames=0, output_video=None):
        """Process a video file for license plate detection."""
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"[INFO] Video: {os.path.basename(video_path)}")
        print(f"[INFO] Resolution: {width}x{height} | FPS: {fps:.1f} | Duration: {duration:.1f}s")
        print(f"[INFO] Total frames: {total_frames}")
        if max_frames > 0:
            print(f"[INFO] Processing: {max_frames} frames\n")
        else:
            print(f"[INFO] Processing: all frames\n")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Video writer for output
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        all_plates = {}  # plate_text -> list of (frame_num, confidence)
        frame_num = 0
        start_time = time.time()
        fps_deque = deque(maxlen=30)
        cached_detections = []

        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if max_frames > 0 and frame_num > max_frames:
                break

            # Run OCR every skip_frames
            if frame_num % self.skip_frames == 0:
                ocr_start = time.time()
                cached_detections = self.detect_plates(frame)
                ocr_time = time.time() - ocr_start

                # Track unique plates
                for (_, text, conf) in cached_detections:
                    if text not in all_plates:
                        all_plates[text] = []
                    all_plates[text].append((frame_num, conf))

            display_frame = self.draw_plates(frame, cached_detections)

            # Progress bar
            progress = frame_num / max_frames if max_frames > 0 else frame_num / total_frames
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '#' * filled + '-' * (bar_len - filled)
            elapsed = time.time() - start_time
            eta = (elapsed / frame_num * (max_frames if max_frames > 0 else total_frames - frame_num)) if frame_num > 0 else 0

            proc_fps = frame_num / max(elapsed, 1e-6)
            info = (f"\r[{bar}] {progress*100:.1f}% | "
                    f"Frame {frame_num}/{max_frames if max_frames > 0 else total_frames} | "
                    f"{proc_fps:.1f} fps | "
                    f"Plates: {len(all_plates)} unique | "
                    f"ETA: {eta:.0f}s")
            print(info, end='', flush=True)

            # Overlay on frame
            cv2.rectangle(display_frame, (0, 0), (350, 80), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Frame: {frame_num}/{max_frames if max_frames > 0 else total_frames}",
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Plates: {len(all_plates)} unique",
                        (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if writer:
                writer.write(display_frame)

            if not self.headless:
                cv2.imshow("License Plate Detection - Video", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        print()  # newline after progress bar
        cap.release()
        if writer:
            writer.release()
        if not self.headless:
            cv2.destroyAllWindows()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"VIDEO ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {frame_num} frames in {elapsed:.1f}s ({frame_num/max(elapsed,1e-6):.1f} fps)")
        print(f"Unique plates detected: {len(all_plates)}")
        print(f"{'='*60}")

        if all_plates:
            print(f"\nDetected License Plates:")
            print(f"{'-'*40}")
            for plate, occurrences in sorted(all_plates.items(), key=lambda x: -len(x[1])):
                frames_list = [f[0] for f in occurrences]
                avg_conf = sum(f[1] for f in occurrences) / len(occurrences)
                print(f"  {plate:15s} | Seen: {len(occurrences):3d}x | "
                      f"Avg confidence: {avg_conf:.0%} | "
                      f"Frames: {frames_list[:5]}{'...' if len(frames_list) > 5 else ''}")
            print(f"{'-'*40}")

            # Save plate summary to text file
            if save_dir:
                summary_path = os.path.join(save_dir, "plate_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Frames processed: {frame_num}\n")
                    f.write(f"Unique plates: {len(all_plates)}\n\n")
                    for plate, occurrences in sorted(all_plates.items(), key=lambda x: -len(x[1])):
                        frames_list = [f[0] for f in occurrences]
                        avg_conf = sum(f[1] for f in occurrences) / len(occurrences)
                        f.write(f"{plate} | Count: {len(occurrences)} | "
                                f"Avg conf: {avg_conf:.0%} | Frames: {frames_list}\n")
                print(f"\n[INFO] Summary saved: {summary_path}")

        return all_plates


def main():
    import argparse

    parser = argparse.ArgumentParser(description="License Plate Detection Pipeline")
    parser.add_argument("--mode", choices=["live", "capture", "image", "video"], default="live")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--input", type=str, help="Input path (image or video)")
    parser.add_argument("--output", type=str, help="Output save path")
    parser.add_argument("--save-dir", type=str, default="./plates")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--ocr-width", type=int, default=640)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--headless", action="store_true",
                        help="No GUI window (console-only output)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process in live mode (0=unlimited)")

    args = parser.parse_args()

    detector = LicensePlateDetector(
        gpu=not args.no_gpu,
        ocr_width=args.ocr_width,
        skip_frames=args.skip,
        headless=args.headless
    )

    if args.mode == "live":
        detector.run_live(save_dir=args.save_dir, camera_index=args.camera,
                          max_frames=args.max_frames)
    elif args.mode == "capture":
        detector.capture_single(save_path=args.output, camera_index=args.camera)
    elif args.mode == "image":
        if not args.input:
            print("[ERROR] --input required for image mode.")
            sys.exit(1)
        detector.process_image(args.input, save_path=args.output)
    elif args.mode == "video":
        if not args.input:
            print("[ERROR] --input required for video mode.")
            sys.exit(1)
        detector.process_video(
            args.input,
            save_dir=args.save_dir,
            max_frames=args.max_frames,
            output_video=args.output
        )


if __name__ == "__main__":
    main()
