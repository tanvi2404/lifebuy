import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# === GPIO Setup ===
SERVO_PIN = 18   # Servo for direction
ESC_PIN = 19     # ESC for BLDC motor
GPIO_TRIGGER = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(ESC_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM for servo
esc = GPIO.PWM(ESC_PIN, 50)  # 50Hz PWM for BLDC ESC

servo.start(2.5)  # Initialize servo at 0°
time.sleep(1)
servo.ChangeDutyCycle(0)  # Stop initial jitter

esc.start(0)  # No thrust initially

# === Mediapipe Setup ===
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# === Camera Setup ===
cap = cv2.VideoCapture(0)

# === Servo Movement Tracking Dictionary ===
servo_position_log = []  # Stores servo direction history
last_servo_position = 90  # Default position
boat_returning = False  # Flag for return mode

# === Motor Control Functions ===
def set_thrust(speed):
    esc.ChangeDutyCycle(speed)

# === Servo Movement with Logging ===
def move_servo(angle):
    global last_servo_position

    # 🔥 Fix PWM Calculation 🔥
    duty = 2.5 + (angle / 18) * 10  # Adjusted for better movement
    print(f"Setting Servo to {angle}° -> PWM: {duty:.2f}")

    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)  # Allow time for movement
    servo.ChangeDutyCycle(0)  # Stop jitter

    if angle != last_servo_position:  # Log only changes
        servo_position_log.append(angle)
        last_servo_position = angle

# === Function to Return to Start Using Reverse Navigation ===
def return_to_start():
    global boat_returning
    print("Returning to starting position...")

    if not servo_position_log:
        print("No movement history recorded! Cannot return.")
        return

    # Reverse movement
    for angle in reversed(servo_position_log):
        move_servo(angle)
        set_thrust(7)  # Restart motor
        time.sleep(1)  # Maintain position for movement

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

        h, w, _ = frame.shape  
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

        # === Detect Face ===
        face_results = face_detection.process(rgb_frame)
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

                cv2.rectangle(frame, (int(x * w), int(y * h)),
                              (int((x + w_box) * w), int((y + h_box) * h)), (0, 255, 0), 2)

        # === Detect Hands ===
        hand_results = hands.process(rgb_frame)
        hand_count = 0

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_count += 1  

        # === Enable Tracking Only If Face & Both Hands Are Detected ===
        if face_detected and hand_count >= 2:
            tracking_mode = True  
            print("Face & Both Hands Detected - Tracking Activated!")

        # === Face Tracking & Boat Control ===
        if tracking_mode and face_detected:
            if face_width > 200:  # Stop moving if face is close to camera
                print("Face is close, stopping boat.")
                set_thrust(0)  
                GPIO.output(GPIO_TRIGGER, False)

                print("Waiting for 15 seconds for the survivor to board...")
                time.sleep(15)  

                if not boat_returning:  
                    boat_returning = True
                    return_to_start()

            else:
                if face_center_x < frame_center - 150:  
                    servo_angle = 135  
                elif face_center_x > frame_center + 150:  
                    servo_angle = 45  
                else:
                    servo_angle = 90  

                move_servo(servo_angle)
                set_thrust(7)  
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
