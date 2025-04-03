import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# === Servo & Motor Setup ===
SERVO_PIN = 18
ESC_PIN = 19
GPIO_TRIGGER = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(ESC_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM for servo
esc = GPIO.PWM(ESC_PIN, 50)  # 50Hz PWM for BLDC ESC

servo.start(7.5)
esc.start(0)

# === Mediapipe Setup ===
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# === Camera Setup ===
cap = cv2.VideoCapture(0)

# === Variables to Track Movement ===
movements = []  # Store movement history

def set_thrust(speed):
    esc.ChangeDutyCycle(speed)

def move_servo(angle):
    duty = (angle / 18) + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(0.2)
    servo.ChangeDutyCycle(0)

# === Main Loop ===
try:
    tracking_mode = False
    initial_position_recorded = False
    face_detected = False
    hand_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_detection.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # === Detect Face ===
        face_detected = False
        face_center_x = 0
        face_width = 0

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w_box, h_box = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
                face_center_x = int((x + w_box / 2) * w)
                face_width = int(w_box * w)
                face_detected = True
                frame_center = w // 2

        # === Detect Hands ===
        hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

        # === Enable Tracking Only If Face & Both Hands Are Detected ===
        if face_detected and hand_count >= 2:
            tracking_mode = True

        # === Track Face and Move Boat ===
        if tracking_mode and face_detected:
            if not initial_position_recorded:
                movements.clear()  # Reset movement tracking
                initial_position_recorded = True

            if face_width > 200:
                set_thrust(0)  # Stop moving
                GPIO.output(GPIO_TRIGGER, False)
            else:
                if face_center_x < frame_center - 150:
                    move_servo(135)
                    movements.append("L")  # Log movement
                elif face_center_x > frame_center + 150:
                    move_servo(45)
                    movements.append("R")
                else:
                    move_servo(90)
                    movements.append("F")  # Forward movement
                
                set_thrust(7)
                GPIO.output(GPIO_TRIGGER, True)

        # === Display Video Feed ===
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Return to Initial Position ===
    print("Returning to Initial Position...")
    for move in reversed(movements):
        if move == "L":
            move_servo(45)  # Move right to compensate
        elif move == "R":
            move_servo(135)  # Move left to compensate
        elif move == "F":
            set_thrust(7)  # Move forward (reverse is needed for actual reversal)
        time.sleep(1)

    set_thrust(0)
    print("Returned to Start Position.")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
