import cv2
import mediapipe as mp
import math
from time import sleep

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Gesture recognition function
def recognize_gesture(hand_landmarks):
    landmarks = {id: (lm.x, lm.y) for id, lm in enumerate(hand_landmarks.landmark)}

    xroot = landmarks[0][0]
    yroot = landmarks[0][1]
    #x postions of the fingers
    xtl = landmarks[1][0]
    xil = landmarks[5][0]
    xml = landmarks[9][0]
    xrl = landmarks[13][0]
    xll = landmarks[17][0]
    
    xtm1 = landmarks[2][0]
    xim1 = landmarks[6][0]
    xmm1 = landmarks[10][0]
    xrm1 = landmarks[14][0]
    xlm1 = landmarks[18][0]

    xtm2 = landmarks[3][0]
    xim2 = landmarks[7][0]
    xmm2 = landmarks[11][0]
    xrm2 = landmarks[15][0]
    xlm2 = landmarks[19][0]

    xtt = landmarks[4][0]
    xit = landmarks[8][0]
    xmt = landmarks[12][0]
    xrt = landmarks[16][0]
    xlt = landmarks[20][0]

    #y postions of the fingers
    ytl = landmarks[1][1]
    yil = landmarks[5][1]
    yml = landmarks[9][1]
    yrl = landmarks[13][1]
    yll = landmarks[17][1]
    
    ytm1 = landmarks[2][1]
    yim1 = landmarks[6][1]
    ymm1 = landmarks[10][1]
    yrm1 = landmarks[14][1]
    ylm1 = landmarks[18][1]

    ytm2 = landmarks[3][1]
    yim2 = landmarks[7][1]
    ymm2 = landmarks[11][1]
    yrm2 = landmarks[15][1]
    ylm2 = landmarks[19][1]

    ytt = landmarks[4][1]
    yit = landmarks[8][1]
    ymt = landmarks[12][1]
    yrt = landmarks[16][1]
    ylt = landmarks[20][1]

    # Gesture 'A': Thumb extended, other fingers closed {ytt = y distance thumb top}
    if (ytt<(yim1 and ymm1 and yrm1 and ylm1)) and (yroot>(yit and ymt and yrt and ylt)) and ((yit,ymt,yrt,ylt)>(yil,yml,yrl,yll)) and xroot<xtl and (ytt<yim1 and ytt<ymm1 and ytt<yrm1 and ytt<ylm1) and(xtt>xim1) and(ylt>ytt):
        return "A"
    elif xtt<xml and (((yit,ymt,yrt,ylt)<(yim2,ymm2,yrm2,ylm2)) and ((yim2,ymm2,yrm2,ylm2)<(yim1,ymm1,yrm1,ylm1))and ((yim1,ymm1,yrm1,ylm1)<(ytl,yml,yrl,yll))) and xroot<xtl and (ylt<ytt and yrt<ytt):
        return "B"
    elif (xroot<(xil and xml and xrl and xll and xtl) and yroot>(yil and yml and yrl and yll and ytl) and ((ytt-ymt)>0.15 and (ytt-ymt)<0.45)) and (xit>xtm2 and xmt > xim2 and xrt > xrm2 and xlt > xlm2) and (xtt>xil) and xroot<xtl:
        return "C"
    elif (yit<(yim1 and ymm1 and yrm1 and ylm1) and (ytt-ymt) <0.05) and ((xtt and xmt)<xil) and (yrt>yrl and ylt>yll) and xroot<xtl and (xtt<xil) and (yit<ymm1) and (ytt>ymt):
        return "D"
    elif (yit>yim1 and ymt>ymm1 and yrt>yrm1 and ylt>ylm1) and (ytt>(ytt and ymt and yrt and ylt)) and (xtt<xrt) and xroot<xtl:
        return "E"
    elif (ymt<ymm2 and yrt<yrm2 and ylt<ylm2) and (xll<xrl)and ((ytt-yit) <0.05) and xroot<xtl:
        return "F"
    elif (xit<xim2 and xim2<xim1 and xim1<xil) and (ytm2<ymt and ytm2<yrt and ytm2<yrt) and xroot>xtl and ytt>yil and (xmt>xtt and xrt>xtt and xlt>xtt) and (xmt>xtm2):
        return "G"
    elif ((xit<xim2 and xim2<xim1 and xim1<xil)) and (ytm2<ymt and ytm2<yrt and ytm2<yrt) and xroot>xtl and ytt>yil and yml>yil and (xrt>xtt and xlt>xtt) and (ytt>ymt):
        return "H"
    elif (ylt<yrm1 and ylt<ylm2) and (xtt<xim2) and (ytt<yit and ytt<ymt and ytt<yrt) and xroot<xtl:
        return "I"
    elif (ylt>yrt and ylt>ymt and ylt>yit and ylt>ytt) and (xtt>xit and xtt>xmt and xtt>xrt and xtt>xlt) and (yim1<ymm1 and ymm1<yrm1) and (ytt>yim1) and (yim1>yil):
        return "J"
    elif (yrt>ytm1 and ylt>ytm1) and (xtt>xmm1 and xtt<xim1) and (ymt<ymm1 and yit<yim1):
        return "K"
    elif (yrt>yrl and ylt>yll and ymt>yml) and (yit<ymt) and (xtt>xil) and (yit<yil) and(xtm2<xtt):
        return "L"
    elif (yit>yim1 and ymt>ymm1 and yrt>yrm1 and ylt>ylm1) and (xtt<xrm1 and ytt<ymt and ytt<yrt and ytt<yit):
        return "M"
    elif (yit>yim1 and ymt>ymm1 and yrt>yrm1 and ylt>ylm1) and (xtt<xmm1 and ytt<ymt and ytt<yit):
        return "N"
    elif ((ytt-yit)<0.05 and (ytt-ymt)<0.06 and (ytt-yrt)<0.07 and (ytt-ylt)<0.08) and ((ytm1-ylm1)>0.09) and ((xtm1-xit)>.08):
        return "O"
    elif (xit<xmt and yim2<ytt and ytt<ymm2 and ytt<yrm2 and ytt<ylm2 and xmt<xrm2 and xmt<xlm2):
        return "P"
    elif (yit>ymt and yit>yrt and yit>ylt and ytt>ymt and ytt>yrt and ytt>ylt) and (xtt-xit<.12):
        return "Q"
    elif (ytt<yrt and ytt<ylt and ytt>yit and ytt>ymt) and (xmt>xit) and (ymt>yit):
        return "R"
    elif (yit>yim1 and ymt>ymm1 and yrt>yrm1 and ylt>ylm1) and (xtt<xim1 and ytt>yim1 and ytt>ymm1) and (xtt>xmm1) :
        return "S"
    elif (yit>yim1 and ymt>ymm1 and yrt>yrm1 and ylt>ylm1) and (ytt<ymm1):
        return "T"
    elif (yrt>yrm1 and ylt>ylm1) and (ymt<yrm1 and yit<yrm1) and (xtt<xrl) and (xit-xmt<0.07):
        return "U"
    elif (yrt>yrm1 and ylt>ylm1) and (ymt<yrm1 and yit<yrm1) and (xtt<xrl) and (xit-xmt>0.1):
        return "V"
    elif (ylt>ylm1) and (ymt<yrm1 and yit<yrm1) and (xtt<xrl):
        return "W"
    elif (yit>yim2) and (ytt>yml and ymt>yml and yrt>yml and ylt>yml):
        return "X"
    elif (ylt<ytt) and (yit>ytm1 and ymt>ytm1 and yrt>ytm1) and (xll<xrl and xtm1>xil):
        return "Y"
    elif (xtt<xil and ytt>ymm1):
        return "Z"
# Function to start the hand gesture recognition
def hand_gesture_recognition():
    cap = cv2.VideoCapture(1)  # Change the index if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame horizontally for a later selfie-view display
        # frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks)
                if gesture:
                    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hand_gesture_recognition() 
