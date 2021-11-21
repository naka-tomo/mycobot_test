import cv2
import numpy as np
from pymycobot.mycobot import MyCobot
import time
import copy 
from kbhit import *

def main():
    atexit.register(set_normal_term)
    set_curses_term()

    mycobot = MyCobot("/dev/cu.SLAB_USBtoUART")
    pose = [200, 0, 100, -180, 0, 0]

    hand_open = True    
    mycobot.sync_send_coords(pose, 10, 0)

    while 1:
        x = y = z = 0
        input_ch = None
        while kbhit():
            input_ch = getch()
            if input_ch=="A":     # ↑
                x = 10
            elif input_ch=="B":   # ↓
                x = -10
            elif input_ch=="C":   # →
                y = -10
            elif input_ch=="D":   # ←
                y = 10
            elif input_ch=="\n":  # Enter
                hand_open = not hand_open

        if input_ch!=None:
            print( "input", input_ch )

            # 現在の指定位置を記録
            prev_pose = copy.deepcopy( pose )

            # 入力に応じて角度を計算
            pose[0] = pose[0] + x
            pose[1] = pose[1] + y
            pose[2] = pose[2]

            print(pose)
            mycobot.send_coords(pose, 30, 0)

            if hand_open:
                mycobot.set_gripper_state( 0, 0 )
            else:
                mycobot.set_gripper_state( 1, 0 )

            # 動作範囲外であれば基の位置へ戻す
            dist = np.sqrt( pose[0]**2 + pose[1]**2 + pose[2]**2 )
            if dist > 280 or pose[0]<100:
                print("範囲外")
                pose[:] = prev_pose
        time.sleep(0.1)

if __name__=="__main__":
    main()