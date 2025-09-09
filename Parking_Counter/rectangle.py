import cv2
import numpy as np

# RTSP stream source
cap = cv2.VideoCapture("rtsp://ctrlpark_admin:mingae123@192.168.43.92:554/stream1")
if not cap.isOpened():
    print("Error opening RTSP stream.")
    exit()

# define multiple slot polygons (coordinates should be adjusted to match your stream's frame size)
slot_1 = np.array([[2, 513], [232, 430], [183, 332], [3, 390]], np.int32).reshape((-1, 1, 2))
slot_2 = np.array([[7, 391], [187, 333], [164, 272], [0, 317]], np.int32).reshape((-1, 1, 2))
slot_3 = np.array([[2, 319], [163, 273], [144, 231], [4, 266]], np.int32).reshape((-1, 1, 2))
slot_4 = np.array([[3, 267], [143, 233], [136, 201], [5, 231]], np.int32).reshape((-1, 1, 2))
slot_5 = np.array([[4, 231], [137, 202], [132, 178], [4, 207]], np.int32).reshape((-1, 1, 2))
slot_6 = np.array([[5, 207], [131, 177], [124, 161], [3, 186]], np.int32).reshape((-1, 1, 2))
slot_7 = np.array([[2, 189], [125, 162], [115, 149], [2, 168]], np.int32).reshape((-1, 1, 2))

# batch 2
slot_8 = np.array([[3, 169], [116, 148], [114, 137], [7, 157]], np.int32).reshape((-1, 1, 2))
slot_9 = np.array([[6, 159], [116, 136], [108, 128], [16, 146]], np.int32).reshape((-1, 1, 2))
slot_10 = np.array([[17, 144], [108, 128], [108, 121], [22, 136]], np.int32).reshape((-1, 1, 2))
slot_11 = np.array([[23, 136], [109, 118], [105, 113], [27, 128]], np.int32).reshape((-1, 1, 2))
slot_12 = np.array([[109, 113], [40, 125], [42, 116], [111, 106]], np.int32).reshape((-1, 1, 2))
slot_13 = np.array([[105, 108], [39, 123], [41, 113], [104, 104]], np.int32).reshape((-1, 1, 2))
slot_14 = np.array([[44, 107], [108, 96], [44, 98], [106, 90]], np.int32).reshape((-1, 1, 2))
slot_15 = np.array([[49, 93], [106, 84], [50, 90], [106, 79]], np.int32).reshape((-1, 1, 2))
slot_16 = np.array([[51, 83], [105, 74], [53, 82], [102, 72]], np.int32).reshape((-1, 1, 2))
slot_17 = np.array([[56, 76], [102, 69], [59, 76], [96, 72]], np.int32).reshape((-1, 1, 2))

#parking2
slot_18 =np.array([[585, 352], [844, 328], [750, 279], [484, 287]], np.int32).reshape((-1, 1, 2))
slot_19 = np.array([[484, 287], [743, 278], [651, 240], [414, 241]], np.int32).reshape((-1, 1, 2))
slot_20 = np.array([[414, 241], [650, 240], [581, 210], [366, 208]], np.int32).reshape((-1, 1, 2))
slot_21 = np.array([[364, 206], [579, 212], [516, 188], [326, 185]], np.int32).reshape((-1, 1, 2))
slot_22 = np.array([[326, 185], [517, 188], [465, 161], [296, 165]], np.int32).reshape((-1, 1, 2))
slot_23 = np.array([[296, 165], [470, 161], [427, 148], [267, 146]], np.int32).reshape((-1, 1, 2))
slot_24 = np.array([[267, 146], [434, 149], [385, 136], [250, 134]], np.int32).reshape((-1, 1, 2))
slot_25 = np.array([[256, 133], [395, 134], [368, 126], [238, 127]], np.int32).reshape((-1, 1, 2))
slot_26 = np.array([[238, 127], [369, 125], [345, 116], [223, 118]], np.int32).reshape((-1, 1, 2))
slot_27 = np.array([[223, 118], [343, 115], [325, 106], [211, 112]], np.int32).reshape((-1, 1, 2))
slot_28 = np.array([[211, 112], [326, 106], [308, 100], [201, 103]], np.int32).reshape((-1, 1, 2))
slot_29 = np.array([[201, 103], [307, 99], [292, 97], [193, 97]], np.int32).reshape((-1, 1, 2))
slot_30 = np.array([[193, 97], [291, 93], [280, 91], [187, 93]], np.int32).reshape((-1, 1, 2))
slot_31 = np.array([[187, 93], [281, 90], [268, 88], [181, 90]], np.int32).reshape((-1, 1, 2))
slot_32 = np.array([[181, 90], [268, 89], [261, 82], [176, 84]], np.int32).reshape((-1, 1, 2))

#parking 3
slot_34 = np.array([[916, 306], [936, 257], [869, 233], [838, 263]], np.int32).reshape((-1, 1, 2))
slot_35 = np.array([[837, 263], [868, 233], [804, 210], [758, 228]], np.int32).reshape((-1, 1, 2))
slot_36 = np.array([[758, 228], [805, 211], [743, 186], [691, 203]], np.int32).reshape((-1, 1, 2))
slot_37 = np.array([[690, 203], [741, 186], [699, 168], [633, 180]], np.int32).reshape((-1, 1, 2))
slot_38 = np.array([[633, 180], [697, 172], [646, 152], [577, 164]], np.int32).reshape((-1, 1, 2))
slot_39 = np.array([[577, 164], [646, 154], [597, 142], [522, 149]], np.int32).reshape((-1, 1, 2))
slot_40 = np.array([[522, 149], [597, 147], [487, 134], [564, 127]], np.int32).reshape((-1, 1, 2))
slot_41 = np.array([[488, 133], [564, 129], [458, 122], [526, 118]], np.int32).reshape((-1, 1, 2))
slot_42 = np.array([[459, 120], [521, 117], [428, 116], [499, 109]], np.int32).reshape((-1, 1, 2))
slot_43 = np.array([[429, 116], [499, 111], [477, 103], [406, 107]], np.int32).reshape((-1, 1, 2))
slot_44 = np.array([[407, 105], [478, 104], [452, 96], [386, 101]], np.int32).reshape((-1, 1, 2))
slot_45 = np.array([[387, 100], [451, 96], [432, 92], [364, 94]], np.int32).reshape((-1, 1, 2))
slot_46 = np.array([[366, 96], [433, 90], [409, 85], [346, 89]], np.int32).reshape((-1, 1, 2))
slot_47 = np.array([[346, 90], [408, 83], [392, 81], [327, 85]], np.int32).reshape((-1, 1, 2))
slot_48 = np.array([[327, 86], [392, 81], [378, 79], [312, 81]], np.int32).reshape((-1, 1, 2))
slot_49 = np.array([[314, 81], [376, 81], [361, 75], [303, 78]], np.int32).reshape((-1, 1, 2))
slot_50 = np.array([[304, 79], [359, 76], [347, 71], [294, 77]], np.int32).reshape((-1, 1, 2))
slot_51 = np.array([[294, 77], [347, 71], [338, 71], [280, 74]], np.int32).reshape((-1, 1, 2))
slot_52 = np.array([[281, 74], [339, 71], [272, 69], [330, 68]], np.int32).reshape((-1, 1, 2))


slots = [
    slot_1, slot_2, slot_3, slot_4, slot_5, slot_6,
    slot_7, slot_8, slot_9, slot_10, slot_11,
    slot_12, slot_13, slot_14, slot_15, slot_16, slot_17,

    #parking 2
    slot_18, slot_19, slot_20, slot_21, slot_22, slot_23, 
    slot_24, slot_25, slot_26, slot_27, slot_28, slot_29, 
    slot_30, slot_31, slot_32,

    #parking 3
    slot_34, slot_35, slot_36, slot_37, slot_38,
    slot_39, slot_40, slot_41, slot_42, slot_43, slot_44,
    slot_45, slot_46, slot_47, slot_48, slot_49, slot_50,
    slot_51, slot_52
]


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize if needed (optional)
    frame = cv2.resize(frame, (960, 540))

    frame_copy = frame.copy()

    # Draw polygons on the frame
    for slot in slots:
        cv2.polylines(frame_copy, [slot], isClosed=True, color=(255, 0, 255), thickness=2)

    cv2.imshow("Parking Slots Live", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
