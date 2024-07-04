import cv2
import mediapipe as mp
import jetson_inference
import jetson_utils
import time

# Define the size of the new video frame
new_frame_width = 800
new_frame_height = 448

# Initialize the video capture
camera_id = "/dev/video0"
cam = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)
cam.set(cv2.CAP_PROP_FPS, 30)

# Initialize the face detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the object detection model
net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Previous frame
prev_frame_face = None
prev_frame_object = None

# Time vars
prev_time_face = time.time()
prev_time_object = time.time()
timer_duration = 3 # 3 seconds

while True:
   # Read a frame from the camera
   res, frame = cam.read()
   if not res:
       break

   # Detect FACES in the frame
   image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = face_detection.process(image_rgb)

   # Detect OBJECTS in the frame
   detectionsB = net.Detect(jetson_utils.cudaFromNumpy(frame))

   # Add the timer for Face and Obj frames
   if prev_frame_face is not None:
       if time.time() - prev_time_face >= timer_duration:
           prev_frame_face = None
           prev_time_face = time.time()

   if prev_frame_object is not None:
       if time.time() - prev_time_object >= timer_duration:
           prev_frame_object = None
           prev_time_obprev_frame_object = time.time()

   # Initialize the numnber of objects in Frame
   num_person = 0

   # Keep track of the # of objects in Frame
   for detection in detectionsB:
       if detection.ClassID==1: num_person += 1

   # If a FACE is detected
   if results.detections:
       for detection in results.detections:
           # Calculate the bounding box coordinates of the detected face
           l = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1])
           r = int(detection.location_data.relative_bounding_box.ymin * frame.shape[0])
           l1 = int(detection.location_data.relative_bounding_box.width * frame.shape[1])
           r1 = int(detection.location_data.relative_bounding_box.height * frame.shape[0])

           # Store the prev. coordinates
           last_face_coords = (l, r, l1, r1)

           # Center the frame on the detected face
           frame_x = max(0, l + (l1 - new_frame_width) // 2)
           frame_y = max(0, r + (r1 - new_frame_height) // 2)

           # Extract the region of interest from the original frame
           roi = frame[frame_y:frame_y + new_frame_height, frame_x:frame_x + new_frame_width]

           # Create a new frame of size 800x448
           new_frame = cv2.resize(roi, (new_frame_width, new_frame_height + 100))

           # Display the new frame
           cv2.imshow("Centered Object", new_frame)

           # Update the previous frame
           # Start a timer every time its updated
           prev_frame_face = new_frame.copy()
           prev_time_face = time.time()

   # If no Face but an OBJECT is detected
   elif not results.detections and detectionsB and num_person == 1:
       # Detect objects in the frame
       #detections = net.Detect(jetson_utils.cudaFromNumpy(frame))
       if detectionsB:
           for obj in detectionsB:
               # Get bounding box coordinates
               left = int(obj.Left)
               top = int(obj.Top)
               right = int(obj.Right)
               bottom = int(obj.Bottom)
             
               # Store the prev. coordinates
               last_obj_coords = (left, top, right, bottom)
             
               # Calculate the center of the bounding box
               center_x = left + (right - left) // 2
               center_y = top + (bottom - top - 350) // 2

               # Calculate the new frame position
               frame_x = max(0, center_x - new_frame_width // 2)
               frame_y = max(0, center_y - new_frame_height // 2)

               # Extract the region of interest from the original frame
               roi = frame[frame_y:frame_y + new_frame_height, frame_x:frame_x + new_frame_width]

               # Create a new frame of size 800x448
               new_frame = cv2.resize(roi, (new_frame_width, new_frame_height + 100))

               # Display the new frame
               cv2.imshow("Centered Object", new_frame)

               # Update the previous frame
               # Start a timer every time its updated
               prev_frame_object = new_frame.copy()
               prev_time_object = time.time()

   # If no Faces && no Objects are detected || Mult FACES || Mult OBJECTS are detected
   elif not results.detections and not detectionsB:
       # If an OBJECT was prevous detected
       if prev_frame_object is not None:
           left, top, right, bottom = last_obj_coords  # use last frames coords

           # Calculate the center of the bounding box
           center_x = left + (right - left) // 2
           center_y = top + (bottom - top - 300) // 2

           # Calculate the new frame position
           frame_x = max(0, center_x - new_frame_width // 2)
           frame_y = max(0, center_y - new_frame_height // 2)

           # Extract the region of interest from the original frame
           roi = frame[frame_y:frame_y + new_frame_height, frame_x:frame_x + new_frame_width]

           # Create a new frame of size 800x448
           new_frame = cv2.resize(roi, (new_frame_width, new_frame_height + 100))

           # Display the new frame
           cv2.imshow("Centered Object", new_frame)

       # If a FACE was prev
       elif prev_frame_face is not None:
           l, r, l1, r1 = last_face_coords # use last frames coords

           # Center the frame on the detected face
           frame_x = max(0, l + (l1 - new_frame_width) // 2)
           frame_y = max(0, r + (r1 - new_frame_height) // 2)

           # Extract the region of interest from the original frame
           roi = frame[frame_y:frame_y + new_frame_height, frame_x:frame_x + new_frame_width]

           # Create a new frame of size 800x448
           new_frame = cv2.resize(roi, (new_frame_width, new_frame_height + 100))

           # Display the new frame
           cv2.imshow("Centered Object", new_frame)

       # If multiple FACES or OBJECTS detected or NOTHING ever WAS
       else:
           # If prev_frame is None, display a basic resized window
           new_frame = cv2.resize(frame, (new_frame_width, new_frame_height))
           cv2.imshow("Centered Object", new_frame)

   # When Mulitple people are in frame
   elif num_person > 1:
       # If an OBJECT was prevous detected
       if prev_frame_object is not None:
           left, top, right, bottom = last_obj_coords  # use last frames coords

           # Calculate the center of the bounding box
           center_x = left + (right - left) // 2
           center_y = top + (bottom - top - 300) // 2

           # Calculate the new frame position
           frame_x = max(0, center_x - new_frame_width // 2)
           frame_y = max(0, center_y - new_frame_height // 2)

           # Extract the region of interest from the original frame
           roi = frame[frame_y:frame_y + new_frame_height, frame_x:frame_x + new_frame_width]

           # Create a new frame of size 800x448
           new_frame = cv2.resize(roi, (new_frame_width, new_frame_height + 100))

           # Display the new frame
           cv2.imshow("Centered Object", new_frame)

       # If a FACE was prev
       elif prev_frame_face is not None:
           l, r, l1, r1 = last_face_coords # use last frames coords

           # Center the frame on the detected face
           frame_x = max(0, l + (l1 - new_frame_width) // 2)
           frame_y = max(0, r + (r1 - new_frame_height) // 2)

           # Extract the region of interest from the original frame
           roi = frame[frame_y:frame_y + new_frame_height, frame_x:frame_x + new_frame_width]

           # Create a new frame of size 800x448
           new_frame = cv2.resize(roi, (new_frame_width, new_frame_height + 100))

           # Display the new frame
           cv2.imshow("Centered Object", new_frame)

       # If multiple FACES or OBJECTS detected or NOTHING ever WAS
       else:
           # If prev_frame is None, display a basic resized window
           new_frame = cv2.resize(frame, (new_frame_width, new_frame_height))
           cv2.imshow("Centered Object", new_frame)

   # Exit on key press '1'
   key = cv2.waitKey(1)
   if key == ord('1'):
       break

# Release resources
cam.release()
