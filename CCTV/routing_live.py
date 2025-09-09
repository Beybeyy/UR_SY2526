import cv2
import numpy as np
from ultralytics import YOLO
import heapq

# ------------------------------
# 1) Load YOLO model
# ------------------------------
model = YOLO("yolov8n.pt")

# ------------------------------
# 2) Connect to RTSP stream
# ------------------------------
cap = cv2.VideoCapture("rtsp://ctrlpark_admin:mingae123@192.168.43.92:554/stream1")
FRAME_WIDTH = 960   # Resize width (set to None to keep original)
FRAME_HEIGHT = 540  # Resize height

if not cap.isOpened():
    print("Error opening RTSP stream.")
    exit()

# ------------------------------
# 3) Define parking slots (polygons)
# ------------------------------
slot_1 = np.array([[499, 407], [604, 463], [863, 404], [755, 372]], np.int32).reshape((-1, 1, 2))
slot_2 = np.array([[752, 373], [631, 305], [426, 366], [497, 407]], np.int32).reshape((-1, 1, 2))
slots = [slot_1, slot_2]

def get_slot_center(slot):
    M = cv2.moments(slot)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

# ------------------------------
# 4) Lane graph base (fixed nodes and edges)
# ------------------------------
nodes_base = {
    "gate": (1156, 505),
    "node1": (800, 550),
    "node2": (650, 500),
}

edges_base = {
    "gate": ["node1"],
    "node1": ["node2"],
    "node2": [],  # We'll add slot nodes dynamically each frame
}

# ------------------------------
# 5) Dijkstra helpers
# ------------------------------
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def shortest_path(start, goal, nodes, edges):
    pq = [(0.0, start, [])]
    seen = set()
    while pq:
        cost, u, path = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        path = path + [u]
        if u == goal:
            return path, cost
        for v in edges.get(u, []):
            heapq.heappush(pq, (cost + euclidean(nodes[u], nodes[v]), v, path))
    return None, float("inf")

def path_length(path, nodes):
    if not path or len(path) < 2:
        return float("inf")
    return sum(euclidean(nodes[path[i]], nodes[path[i+1]]) for i in range(len(path)-1))

# ------------------------------
# 6) Main loop for RTSP frames
# ------------------------------
while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if FRAME_WIDTH and FRAME_HEIGHT:
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

    results = model(img, verbose=False)
    cars = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label in ["car", "truck", "bus"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, y2
                cars.append((cx, cy))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

    overlay = img.copy()
    available_slots = []
    available_count = 0

    for i, slot in enumerate(slots, start=1):
        occupied = any(cv2.pointPolygonTest(slot, (cx, cy), False) >= 0 for (cx, cy) in cars)
        color = (0, 255, 0) if not occupied else (0, 0, 255)
        if not occupied:
            cX, cY = get_slot_center(slot)
            if cX is not None and cY is not None:
                available_slots.append(((cX, cY), i, slot))
                available_count += 1
        cv2.polylines(overlay, [slot], True, color, 2)

    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    if available_slots:
        # Create a fresh copy of nodes and edges for routing
        nodes = dict(nodes_base)
        edges = {k: list(v) for k, v in edges_base.items()}

        # Add slot nodes and connect them to node2 (could be improved with nearest node)
        for (cxy, sid, _) in available_slots:
            sname = f"slot{sid}"
            nodes[sname] = cxy
            edges[sname] = []
            edges["node2"].append(sname)

        # Find best slot with shortest path from gate
        best = None
        best_len = float("inf")
        best_path = None
        for (_, sid, _) in available_slots:
            sname = f"slot{sid}"
            path, _ = shortest_path("gate", sname, nodes, edges)
            L = path_length(path, nodes)
            if L < best_len:
                best_len, best, best_path = L, sid, path

        if best_path:
            for i in range(len(best_path) - 1):
                p1, p2 = nodes[best_path[i]], nodes[best_path[i + 1]]
                cv2.line(img, p1, p2, (0, 255, 255), 3)

            chosen_poly = next(poly for (cxy, sid, poly) in available_slots if sid == best)
            cv2.polylines(img, [chosen_poly], True, (0, 255, 255), 3)
            cX, cY = nodes[f"slot{best}"]
            cv2.putText(img, f"Suggested Slot {best}", (cX - 60, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    text = f"Available: {available_count}/{len(slots)}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x = (img.shape[1] - text_w) // 2
    cv2.putText(img, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Parking with Routing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
