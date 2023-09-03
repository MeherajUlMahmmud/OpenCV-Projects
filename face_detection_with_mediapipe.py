import cv2
import mediapipe as mp

MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
MP_FACE_MESH = mp.solutions.face_mesh
DRAWING_SPEC = MP_DRAWING.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with MP_FACE_MESH.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ) as FACE_MESH:
        results = FACE_MESH.process(img)
        if results.multi_face_landmarks:
            face_landmark = results.multi_face_landmarks[0]

            # Draw face landmarks and highlight specific landmarks (left eye, right eye, mouth and lips)
            MP_DRAWING.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=MP_FACE_MESH.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=MP_DRAWING_STYLES
                .get_default_face_mesh_tesselation_style(),
            )
            MP_DRAWING.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=MP_FACE_MESH.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=MP_DRAWING_STYLES
                .get_default_face_mesh_contours_style(),
                circle_radius=1,
            )
            MP_DRAWING.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=MP_FACE_MESH.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=DRAWING_SPEC,
                connection_drawing_spec=DRAWING_SPEC,
            )
            MP_DRAWING.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=MP_FACE_MESH.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=DRAWING_SPEC,
                connection_drawing_spec=DRAWING_SPEC,
            )
            MP_DRAWING.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=MP_FACE_MESH.FACEMESH_LIPS,
                landmark_drawing_spec=DRAWING_SPEC,
                connection_drawing_spec=DRAWING_SPEC,
            )

    # Add the window name in the bottom left corner in green font
    cv2.putText(img, 'Face Detection', (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Face Detection', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
