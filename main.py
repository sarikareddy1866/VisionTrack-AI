import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Choose input source
print("Choose input source:")
print("1. Webcam")
print("2. Video file")

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    video_path = input("Enter video file path: ").strip().strip('"')
    cap = cv2.VideoCapture(video_path)
else:
    print("Invalid choice.")
    raise SystemExit

if not cap.isOpened():
    print("Error: Could not open the webcam or video file.")
    cap.release()
    raise SystemExit

# Tracking and counting variables
previous_positions = {}
counted_ids = set()
alerted_ids = set()

entry_count = 0
exit_count = 0
alert_count = 0
people_detected = 0

print("VisionTrack AI started!")
print("Press 'q' or Esc in the video window to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended or frame could not be read.")
            break

        height, width = frame.shape[:2]

        # Virtual counting line
        line_y = int(height * 0.65)

        # Define restricted zone
        zone_x1 = int(width * 0.60)
        zone_y1 = int(height * 0.20)
        zone_x2 = int(width * 0.95)
        zone_y2 = int(height * 0.80)

        # Detect and track objects using ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False
        )

        # Draw bounding boxes, labels and tracking IDs
        output = results[0].plot()

        # Draw virtual counting line
        cv2.line(
            output,
            (0, line_y),
            (width, line_y),
            (0, 255, 255),
            2
        )

        # Draw restricted zone
        cv2.rectangle(
            output,
            (zone_x1, zone_y1),
            (zone_x2, zone_y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            output,
            "RESTRICTED ZONE",
            (zone_x1, max(20, zone_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        boxes = results[0].boxes
        people_detected = 0
        intruder_detected = False

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()

            for track_id, class_id, box in zip(
                ids, classes, coordinates
            ):
                # Monitor people only
                if model.names[class_id] != "person":
                    continue

                people_detected += 1

                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 1. ENTRY AND EXIT COUNTING
                previous_y = previous_positions.get(track_id)

                if previous_y is not None:
                    crossed_line = (
                        previous_y < line_y <= center_y
                        or
                        previous_y > line_y >= center_y
                    )

                    # Count each tracking ID only once
                    if crossed_line and track_id not in counted_ids:
                        if previous_y < line_y:
                            entry_count += 1
                            print(f"Person ID {track_id}: ENTRY")
                        else:
                            exit_count += 1
                            print(f"Person ID {track_id}: EXIT")

                        counted_ids.add(track_id)

                previous_positions[track_id] = center_y

                # 2. RESTRICTED-ZONE MONITORING
                inside_zone = (
                    zone_x1 <= center_x <= zone_x2
                    and
                    zone_y1 <= center_y <= zone_y2
                )

                if inside_zone:
                    intruder_detected = True

                    # Count each tracking ID's alert only once
                    if track_id not in alerted_ids:
                        alert_count += 1
                        alerted_ids.add(track_id)

                        print(
                            f"ALERT! Person ID {track_id} "
                            "entered the restricted zone!"
                        )

                    cv2.putText(
                        output,
                        f"INTRUDER ALERT! ID: {track_id}",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # Draw center point
                cv2.circle(
                    output,
                    (center_x, center_y),
                    5,
                    (255, 0, 255),
                    -1
                )

        # 3. LIVE ANALYTICS PANEL
        panel_height = 145
        panel_width = min(430, width - 20)

        cv2.rectangle(
            output,
            (10, 10),
            (10 + panel_width, panel_height),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            output,
            f"People detected: {people_detected}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Entries: {entry_count}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Exits: {exit_count}",
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

        cv2.putText(
            output,
            f"Total zone alerts: {alert_count}",
            (20, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255) if intruder_detected else (255, 255, 255),
            2
        )

        # Display output
        cv2.imshow("VisionTrack AI - Smart Monitoring", output)

        # Stop when q or Esc is pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    # Always release resources
    cap.release()
    cv2.destroyAllWindows()

print("Final Entry Count:", entry_count)
print("Final Exit Count:", exit_count)
print("Total Restricted-Zone Alerts:", alert_count)