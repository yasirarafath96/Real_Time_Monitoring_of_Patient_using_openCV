# -*- coding: utf-8 -*-

import imutils
import cv2
import numpy as np
from tkinter import font
import dlib
import saccademodel
from numpy.ma import hypot
from imutils import face_utils
import datetime
import imutils
import time
from playsound import playsound


# =============================================================================
# USER-SET PARAMETERS
# =============================================================================
def process():
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Number of frames to pass before changing the frame to compare the current
    # frame against
    FRAMES_TO_PERSIST = 10

    # Minimum boxed area for a detected motion to count as actual motion
    # Use to filter out noise or small objects
    MIN_SIZE_FOR_MOVEMENT = 2000

    # Minimum length of time where no motion is detected it should take
    #(in program cycles) for the program to declare that there is no movement
    MOVEMENT_DETECTED_PERSISTENCE = 100
    sflag=False
    ebflag=False
    mflag=False
    def is_speaking(prev_img, curr_img, debug=False, threshold=900, width=400, height=400):
        """
        Args:
            prev_img:
            curr_img:
        Returns:
            Bool value if a person is speaking or not
        """
        prev_img = cv2.resize(prev_img, (width, height))
        curr_img = cv2.resize(curr_img, (width, height))

        diff = cv2.absdiff(prev_img, curr_img)
        norm = np.sum(diff) / (width*height) * 100
        if debug:
            print(norm)
        return norm > threshold
    detector1 = dlib.get_frontal_face_detector()
    predictor1 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indices of the facial landmarks for mouth
    m_start, m_end = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

    def midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]),
                              facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                                 facial_landmarks.part(eye_points[4]))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        ratio = hor_line_lenght / ver_line_lenght
        return ratio

    # =============================================================================
    # CORE PROGRAM
    # =============================================================================


    # Create capture object
    cap = cv2.VideoCapture(5) # Flush the stream
    cap.release()
    cap = cv2.VideoCapture(0) # Then start the webcam

    # Init frame variables
    first_frame = None
    next_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0
    prev_mouth_img = None
    i = 0
    margin = 10
    # LOOP!
    while True:

        # Set transient motion detected as false
        transient_movement_flag = False
        
        # Read frame
        ret, frame = cap.read()
        text = "Unoccupied"

        # If there's an error in capturing
        if not ret:
            print("CAPTURE ERROR")
            continue

        # Resize and save a greyscale version of the image
        frame = imutils.resize(frame, width = 750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector1(gray)
        # detect faces in the grayscale frame
        rects = detector1(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor1(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth_shape = shape[m_start:m_end+1]

            leftmost_x = min(x for x, y in mouth_shape) - margin
            bottom_y = min(y for x, y in mouth_shape) - margin
            rightmost_x = max(x for x, y in mouth_shape) + margin
            top_y = max(y for x, y in mouth_shape) + margin

            w = rightmost_x - leftmost_x
            h = top_y - bottom_y

            x = int(leftmost_x - 0.1 * w)
            y = int(bottom_y - 0.1 * h)

            w = int(1.2 * w)
            h = int(1.2 * h)

            mouth_img = gray[bottom_y:top_y, leftmost_x:rightmost_x]

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            # for (x, y) in mouth_shape:
                # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # confer this
            
            if prev_mouth_img is None:
                prev_mouth_img = mouth_img
            if is_speaking(prev_mouth_img, mouth_img, threshold=900,
                                    debug=True):
                cv2.putText(frame, "Speaking", (50, 150), 1, 7, (255, 0, 0))
                sflag=True
                i += 1

            prev_mouth_img = mouth_img
        for face in faces:

            landmarks = predictor(gray, face)

            # detect blinking

            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            print("Blinking ratio==",blinking_ratio)
            if blinking_ratio > 6:
                ebflag=True
                cv2.putText(frame, "BLINKING", (50, 150), 1, 7, (255, 0, 0))

            # Gaze detection

            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [left_eye_region], True, 255, 2)
            cv2.fillPoly(mask, [left_eye_region], 255)
            left_eye = cv2.bitwise_and(gray, gray, mask=mask)

            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])
            gray_eye = left_eye[min_y: max_y, min_x: max_x]
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            height, width = threshold_eye.shape
            left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
            left_side_white = cv2.countNonZero(left_side_threshold)
            right_side_threshold = threshold_eye[0: height, int(width / 2): width]
            right_side_white = cv2.countNonZero(right_side_threshold)

            #cv2.putText(frame, str(left_side_white), (50, 100), 1, 2, (0, 0, 255), 3)
            #cv2.putText(frame, str(right_side_white), (50, 150), 1, 2, (0, 0, 255), 3)


            input_eye_points = [
                [0,67],
                [64, 0],
                [2,280],
                [63, 198],
                [184,167],
                [465,98],
                [188,90],
                [105,98],
                [209,78],[149,190]

            ]
            results = saccademodel.fit(input_eye_points)

            frame_rate = 300.0  # samples per second
            reaction_time = len(results['source_points']) / frame_rate
            duration = len(results['saccade_points']) / frame_rate
            #cv2.putText(frame, "Reaction time="+str(reaction_time), (50, 200), 1, 2, (0, 0, 255), 3)
            #cv2.putText(frame, str(duration), (50, 250), 1, 2, (0, 0, 255), 3)


            threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
            eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If the first frame is nothing, initialise it
        if first_frame is None: first_frame = gray    

        delay_counter += 1

        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if delay_counter > FRAMES_TO_PERSIST:
            delay_counter = 0
            first_frame = next_frame

            
        # Set the next frame to compare (the current frame)
        next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations = 2)
        __,cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)
            
            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
                
                # Draw a rectangle around big enough movements
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if transient_movement_flag == True:
            movement_persistent_flag = True
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

        # As long as there was a recent transient movement, say a movement
        # was detected    
        if movement_persistent_counter > 0:
            text = "Movement Detected " + str(movement_persistent_counter)
            movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"

        # Print the text on the screen, and display the raw and processed video 
        # feeds
        cv2.putText(frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        # For if you want to show the individual video frames
    #    cv2.imshow("frame", frame)
    #    cv2.imshow("delta", frame_delta)
        
        # Convert the frame_delta to color for splicing
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

        # Splice the two video frames together to make one long horizontal one
        cv2.imshow("frame",frame)
        if sflag or ebflag or mflag:
            playsound("alarm.wav")
            sflag=False
            ebflag=False
            mflag=False


        # Interrupt trigger by pressing q to quit the open CV program
        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break

    # Cleanup when closed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
