import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Load different background videos for different gestures
background_videos = {
    "thumbs_up": cv2.VideoCapture(r"D:\A_DOWNLOADS\855431-hd_1920_1080_24fps.mp4"),
    "peace_sign": cv2.VideoCapture(r"D:\A_DOWNLOADS\122003-724732258_small.mp4"),
    "ok_sign": cv2.VideoCapture(r"D:\A_DOWNLOADS\104386-666197775_small.mp4")
}

# Function to reset video playback to the start
def reset_video(video):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

def is_thumbs_up(hand_landmarks):
    """Detect if the gesture is a 'thumbs up'."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    is_thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    is_index_finger_folded = index_finger_tip.y > thumb_mcp.y

    return is_thumb_up and is_index_finger_folded

def is_peace_sign(hand_landmarks):
    """Detect if the gesture is a 'peace sign'."""
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    is_index_extended = index_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    is_middle_extended = middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    is_ring_folded = ring_finger_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    is_pinky_folded = pinky_finger_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    return is_index_extended and is_middle_extended and is_ring_folded and is_pinky_folded

def is_ok_sign(hand_landmarks):
    """Detect if the gesture is an 'OK sign'."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    thumb_index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
    is_middle_finger_extended = middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    return thumb_index_distance < 0.05 and is_middle_finger_extended

while True:
    # Read the webcam frame
    ret_webcam, frame = cap.read()
    if not ret_webcam:
        print("Failed to capture webcam video")
        break

    # Flip the image horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and landmarks
    result = hands.process(rgb_frame)

    # Gesture detection and corresponding video
    active_video = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Check for each gesture and assign the corresponding video
            if is_thumbs_up(hand_landmarks):
                active_video = background_videos["thumbs_up"]
            elif is_peace_sign(hand_landmarks):
                active_video = background_videos["peace_sign"]
            elif is_ok_sign(hand_landmarks):
                active_video = background_videos["ok_sign"]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If no gesture detected, don't play a background video
    if active_video is not None:
        # Read the background video frame
        ret_bg, bg_frame = active_video.read()
        if not ret_bg:
            # If the background video ends, reset it
            reset_video(active_video)
            continue

        # Resize the background frame to match the webcam frame size
        bg_frame = cv2.resize(bg_frame, (frame.shape[1], frame.shape[0]))

        # Apply Gaussian blur to the background frame
        blurred_bg_frame = cv2.GaussianBlur(bg_frame, (51, 51), 0)

        # Apply grayscale filter to the top 100 pixels of the webcam frame
        gray_top = cv2.cvtColor(frame[0:100, :], cv2.COLOR_BGR2GRAY)
        gray_top_bgr = cv2.cvtColor(gray_top, cv2.COLOR_GRAY2BGR)
        frame[0:100, :] = gray_top_bgr

        # Blend the blurred background with the webcam frame
        alpha = 0.6
        blended_frame = cv2.addWeighted(blurred_bg_frame, 1 - alpha, frame, alpha, 0)
    else:
        blended_frame = frame  # Show normal webcam frame if no gesture is detected

    # Display the resulting frame
    cv2.imshow("frame", blended_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
for video in background_videos.values():
    video.release()
cv2.destroyAllWindows()