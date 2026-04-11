import cv2
import mediapipe as mp


def main():
    mp_hands = mp.solutions.hands 
    mp_draw = mp.solutions.drawing_utils 

    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5,
            max_num_hands = 2
            ) as hands:
        cv2.namedWindow("Camara", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camara", 600, 600)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Camara", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
