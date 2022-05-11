import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils # meken puluwan kisiyam kanadankayak dunnama ama kandankaya lakunu karanna,condanka athara erak adinna
mpPose = mp.solutions.pose # memagin thamaii mideapipe API ekata katha karanne, mehidi karanne sharira eriyau trakking karana mideapipe api ekata kathakaranne
pose = mpPose.Pose()# model eka hadagannawa


cap = cv2.VideoCapture('Makhna- Bollywood dance cover- Team naach choreography.mp4')
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)# RGB image tika model ekata laba denawa
    if results.pose_landmarks: #results kiyana eke model eka haraha hoyala deepu landmarks thiyed
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #'results.pose_landmarks' magin marks tika img eke adinna   & 'mpPose.POSE_CONNECTIONS' magin kandanka athara rekawa adi
        #mpDraw.draw_landmarks(img, results.mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):# mehi enumerate kitana eken karanne,warahanen athule thiyana agaya wenas wenakota eyata index ekaka labadima
            # lm kiyana eken landmark wala kanadanka x,y,z lesa laba dei(namuth mehi getaluwa anupathayak lesa lebimai)
            h, w, c = img.shape # image eke usa, diga, palla labadei
            cx = int(lm.x * w)# anupathayak lesa lebena diga, intiger value ekaka karima
            cy = int(lm.y *h)#anupathayak lesa lebena usa, intiger value ekaka karima
            cv2.circle(img, (cx,cy), 5, (255, 0, 0), cv2.FILLED)
    # meka ganne fream dekak run wena wenasa balaganna
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("dansing", img)
    cv2.waitKey(10)
