import cv2
import numpy as np
import time

LOWER_YCRCB = np.array([0, 133, 77], dtype=np.uint8)
UPPER_YCRCB = np.array([255, 173, 127], dtype=np.uint8)

MIN_CONTOUR_AREA = 2000
SAFE_THRESHOLD = 150
WARNING_THRESHOLD = 80
DANGER_THRESHOLD = 40

def get_state(distance):
    if distance is None:
        return "SAFE"
    if distance > SAFE_THRESHOLD:
        return "SAFE"
    elif distance > WARNING_THRESHOLD:
        return "WARNING"
    else:
        if distance < DANGER_THRESHOLD:
            return "DANGER"
        else:
            return "WARNING"

def draw_state_overlay(frame, state, fps=None):
    h, w, _ = frame.shape
    if state == "SAFE":
        color = (0, 255, 0)
    elif state == "WARNING":
        color = (0, 255, 255)
    elif state == "DANGER":
        color = (0, 0, 255)
    else:
        color = (255, 255, 255)

    cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
    cv2.putText(frame, f"STATE: {state}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if state == "DANGER":
        text = "DANGER DANGER"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        if dt > 0:
            fps = 1.0 / dt

        h, w, _ = frame.shape
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, LOWER_YCRCB, UPPER_YCRCB)

        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.erode(skin_mask, kernel, 1)
        skin_mask = cv2.dilate(skin_mask, kernel, 2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundary_x = int(2 * w / 3)
        cv2.line(frame, (boundary_x, 0), (boundary_x, h), (0, 0, 255), 2)

        hand_center = None
        distance_to_boundary = None

        if contours:
            lower_contours = []
            upper_limit = h // 3

            for c in contours:
                if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(c)
                if y + h_box // 2 > upper_limit:
                    lower_contours.append(c)

            if not lower_contours:
                valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
            else:
                valid_contours = lower_contours

            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > MIN_CONTOUR_AREA:
                    cv2.drawContours(frame, [largest], -1, (255, 0, 0), 2)
                    M = cv2.moments(largest)

                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        hand_center = (cx, cy)

                        cv2.circle(frame, hand_center, 7, (0, 255, 0), -1)
                        cv2.putText(frame, "HAND", (cx - 20, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        distance_to_boundary = abs(boundary_x - cx)
                        cv2.line(frame, (cx, cy), (boundary_x, cy), (255, 255, 255), 1)
                        cv2.putText(frame, f"dist: {int(distance_to_boundary)} px",
                                    (cx + 10, cy + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        state = get_state(distance_to_boundary)
        draw_state_overlay(frame, state, fps)
        cv2.imshow("Hand DANGER Demo", frame)
        cv2.imshow("Skin Mask (debug)", skin_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
