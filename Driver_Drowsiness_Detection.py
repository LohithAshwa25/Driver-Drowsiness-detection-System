import cv2
import mediapipe as mp
import time
import math
from pyfirmata import Arduino, util

# =================== ARDUINO SETUP ===================
com_port = 'COM5'  # Replace with your Arduino COM port
board = Arduino(com_port)

it = util.Iterator(board)
it.start()

# Define output pins
led_pin = board.get_pin('d:8:o')        # LED
buzzer_pin = board.get_pin('d:9:o')     # Buzzer
motor_pin = board.get_pin('d:10:p')     # Motor with PWM

# Initialize outputs
led_pin.write(0)
buzzer_pin.write(0)
motor_pin.write(1.0)  # Motor always HIGH initially (full voltage)

# =================== MEDIA PIPE SETUP ===================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144, 159, 145]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 386, 374]

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[6], eye[7])
    D = euclidean(eye[0], eye[3])
    return (A + B + C) / (3.0 * D)

# =================== PARAMETERS ===================
BLINK_THRESHOLD = 0.23
CLOSE_DURATION = 3  # seconds
EYE_HISTORY = []
MAX_HISTORY = 5

eye_closed_start = None
output_flag = False

# =================== FUNCTIONS ===================
def gradual_pwm_down(pin, step=0.02, delay=0.05):
    """Gradually reduce PWM from 1.0 to 0"""
    current = 1.0
    while current > 0:
        current -= step
        if current < 0:
            current = 0
        pin.write(current)
        time.sleep(delay)

def gradual_pwm_up(pin, step=0.02, delay=0.05):
    """Gradually increase PWM from current to 1.0"""
    current = pin.read() or 0
    while current < 1.0:
        current += step
        if current > 1.0:
            current = 1.0
        pin.write(current)
        time.sleep(delay)

# =================== CAMERA ===================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

        left_EYE = eye_aspect_ratio(left_eye)
        right_EYE = eye_aspect_ratio(right_eye) 
        EYE = (left_EYE + right_EYE) / 2.0

        # Smooth EYE
        EYE_HISTORY.append(EYE)
        if len(EYE_HISTORY) > MAX_HISTORY:
            EYE_HISTORY.pop(0)
        avg_EYE = sum(EYE_HISTORY) / len(EYE_HISTORY)

        cv2.putText(frame, f"EYE: {avg_EYE:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) # In Video 

        # =================== EYE CLOSURE DETECTION ===================
        if avg_EYE < BLINK_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            else:
                elapsed = time.time() - eye_closed_start
                if elapsed >= CLOSE_DURATION and not output_flag:
                    # LED and Buzzer ON immediately
                    led_pin.write(1)
                    buzzer_pin.write(1)
                    print("LED and Buzzer HIGH")
                    # Motor gradually decrease voltage
                    print("Motor gradually LOW")
                    gradual_pwm_down(motor_pin)
                    output_flag = True
        else:
            eye_closed_start = None
            if output_flag:
                # LED and Buzzer OFF immediately
                led_pin.write(0)
                buzzer_pin.write(0)
                print("LED and Buzzer LOW")
                # Motor gradually increase voltage back to full
                print("Motor gradually HIGH")
                gradual_pwm_up(motor_pin)
                output_flag = False

        # Draw eyes
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0,255,0), -1)

    cv2.imshow("Eye Closure Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Reset pins on exit
led_pin.write(0)
buzzer_pin.write(0)
motor_pin.write(1.0)  # motor back to full voltage
