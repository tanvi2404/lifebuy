import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# === GPIO Setup ===
SERVO_PIN = 18   # Servo for direction
ESC_PIN = 19
GPIO_TRIGGER = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # ESC for BLDC motor
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(ESC_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM for servo
esc = GPIO.PWM(ESC_PIN, 50)  # 50Hz PWM for BLDC ESC

servo.start(7.5)  # Middle position
esc.start(0)  # No thrust initially

# === Mediapipe Setup ===
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# === Camera Setup ===
cap = cv2.VideoCapture(0)

# === Servo Movement Tracking Dictionary ===
servo_position_log = {}  # Stores servo direction and duration
last_servo_position = 90  # Default position
last_update_time = time.time()  # Timestamp of last position update
boat_returning = False  # Flag for return mode

def set_thrust(speed):
    esc.ChangeDutyCycle(speed)

# === Servo Movement Function with Logging ===
def move_servo(angle):
    global last_servo_position, last_update_time

    current_time = time.time()
    time_spent = current_time - last_update_time  # Calculate time spent at last position

    # Store duration in dictionary
    if last_servo_position in servo_position_log:
        servo_position_log[last_servo_position] += time_spent
    else:
        servo_position_log[last_servo_position] = time_spent

    # Move the servo
    duty = (angle / 18) + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(0.2)  # Delay to avoid jitter
    servo.ChangeDutyCycle(0)

    # Update last position and time
    last_servo_position = angle
    last_update_time = time.time()

# === Function to Return to Start Point Using Reverse Navigation ===
def return_to_start():
    global boat_returning
    print("Returning to starting position...")

    reverse_log = {}  # Dictionary for reverse movement

    # Reverse the stored movements with mapped angles
    for angle, duration in reversed(servo_position_log.items()):
        if angle == 45:
            reverse_log[225] = duration
        elif angle == 90:
            reverse_log[270] = duration
        elif angle == 135:
            reverse_log[315] = duration

    # Execute reverse movements
    for angle, duration in reverse_log.items():
        print(f"Moving servo to {angle}° for {duration:.2f} seconds")
        move_servo(angle)
        set_thrust(7)  # Restart motor
        time.sleep(duration)  # Maintain servo position for same duration

    print("Reached starting position!")
    set_thrust(0)  # Stop motor
    boat_returning = False  # Reset flag

# === Main Face & Hand Tracking Loop ===
try:
    servo_angle = 90  # Initial servo position
    tracking_mode = False  # Only track when a face & both hands are detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape  # Get frame size
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe

        # === Detect Face ===
        face_results = face_detection.process(rgb_frame)
        face_detected = False
        face_center_x = 0
        face_width = 0

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w_box, h_box = (bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height)
                face_center_x = int((x + w_box / 2) * w)
                face_width = int(w_box * w)  # Face size in pixels
                face_detected = True
                frame_center = w // 2
                # Draw face box
                cv2.rectangle(frame, (int(x * w), int(y * h)),
                              (int((x + w_box) * w), int((y + h_box) * h)), (0, 255, 0), 2)

        # === Detect Hands ===
        hand_results = hands.process(rgb_frame)
        hand_count = 0

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_count += 1  # Count hands

        # === Enable Tracking Only If Face & Both Hands Are Detected ===
        if face_detected and hand_count >= 2:
            tracking_mode = True  # Start tracking
            print("Face & Both Hands Detected - Tracking Activated!")

        # === Face Tracking & Boat Control ===
        if tracking_mode and face_detected:
            if face_width > 200:  # Stop moving if face is close to camera
                print("Face is close, stopping boat.")
                set_thrust(0)  # Stop motor
                GPIO.output(GPIO_TRIGGER, False)

                print("Waiting for 15 seconds for the survivor to board...")
                time.sleep(15)  # Pause for 15 seconds before returning

                if not boat_returning:  # Start return only once
                    boat_returning = True
                    return_to_start()

            else:
                if face_center_x < frame_center - 150:  # Face is on the left
                    servo_angle = 135  # Move left
                elif face_center_x > frame_center + 150:  # Face is on the right
                    servo_angle = 45  # Move right
                else:
                    servo_angle = 90  # Center position

                move_servo(servo_angle)
                set_thrust(7)  # Move forward
                GPIO.output(GPIO_TRIGGER, True)
                print(f"Moving servo to: {servo_angle}°")

        # === Display the Video Feed ===
        cv2.imshow("Face & Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
