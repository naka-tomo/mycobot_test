import cv2
import mediapipe as mp
import numpy as np
from pymycobot.mycobot import MyCobot
import time


class HandDetector():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(False, 2, 1, 0.75)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, img, draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def get_position(self, img, hand_idx = 0, draw = True):
        pos_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_idx]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                pos_list.append([id, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
 
        return pos_list

def main():
    detector = HandDetector()
    capture = cv2.VideoCapture(0)

    mycobot = MyCobot("/dev/cu.SLAB_USBtoUART")
    pose = [200, 0, 100, -180, 0, -20]
    mycobot.sync_send_coords(pose, 10, 0)


    while(True):
        ret, frame = capture.read()

        img_size = (int(frame.shape[1]/2), int(frame.shape[0]/2))
        frame = cv2.resize(frame, img_size)
        frame = cv2.flip( frame, 1 )

        hand_img = detector.detect( frame )
        positions = detector.get_position( frame )
        if len(positions)>0:
            print(positions[5])
            hand_open = False

            finger_tip = np.array(positions[8][1:3])
            finger_root = np.array(positions[5][1:3])
            wrist = np.array(positions[0][1:3])

            # 手首と人差し指の付け根で計算
            if np.linalg.norm(finger_tip[:2]-wrist[:2]) > np.linalg.norm(finger_root[:2]-wrist[:2]):
                # ハンド開く
                mycobot.set_gripper_state( 0, 0 )
            else:
                mycobot.set_gripper_state( 1, 0 )

            # 手首，人示唆し指の根本，指先から各座標を計算
            x = 3.0 * (np.linalg.norm( finger_root[:2]-wrist[:2]  ) /img_size[1]-0.15)
            y = -(finger_root[0]-img_size[0]/2)/(img_size[0]/2)
            z = 1.5*((img_size[1]-finger_root[1])/img_size[1]-0.1)

            x = min( 1, max( 0, x) )
            z = min( 1, max( 0, z))
            print( x, y, z )

            # アーム座標を計算
            pose = [100+350*x, 300*y, 50+300*z, -180, 0, -20]
            mycobot.send_coords(pose, 10, 0)
            time.sleep(0.1)

        cv2.imshow('hands',hand_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()