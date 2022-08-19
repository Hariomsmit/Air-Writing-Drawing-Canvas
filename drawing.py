#AUTHOR - HARI OM
#DATE - 26/06/2022

import mediapipe as mp
import cv2
import numpy as np
import time
from collections import deque

#contants
m_l = 150
max_x, max_y = 250+m_l, 50
current_tool ="Select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0

#get tools function
def getTool(x):
	if x < 50 + m_l:
		return "LINE"

	elif x<100 + m_l:
		return "RECTANGLE"

	elif x < 150 + m_l:
		return"DRAW"

	elif x<200 + m_l:
		return "CIRCLE"

	else:
		return "ERASE"

def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False



hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils


# drawing tools
tools = cv2.imread("C:\\Users\\hario\\Desktop\\Research Paper\\coding\\tools.png")
tools = tools.astype("uint8")

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')


cap = cv2.VideoCapture(0)
while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)

			if x < max_x and y < max_y and x > m_l:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					current_tool = getTool(x)
					print("your current tool set to : ", current_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40

			if current_tool == "DRAW":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y



			elif current_tool == "LINE":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frm, (xii, yii), (x, y), (50,152,0), thick)

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif current_tool == "RECTANGLE":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif current_tool == "CIRCLE":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False

			elif current_tool == "ERASE":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frm, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)

	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]

	frm[:max_y, m_l:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, m_l:max_x], 0.3, 0)

	cv2.putText(frm, current_tool, (270+m_l,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("PROGLINT paint app", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
 