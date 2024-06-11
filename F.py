import cv2
import os
import subprocess
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('./license_plate_detector.pt')  # Replace with the path to your YOLOv8 weights


def detect_license_plate(frame):
    results = model(frame)
    return results


def extract_plate_text(plate_image_path):
    # Run the OpenALPR command
    command = f'alpr -c us {plate_image_path}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Parse the output to extract the license plate number
    output = result.stdout.split('\n')
    if output and "No license plates found." not in output:
        for line in output:
            if 'confidence' in line:
                plate = line.split()[1]
                return plate
    return ""


def main():
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)  # Replace with the appropriate video source if needed

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect license plates in the frame
        results = detect_license_plate(frame)

        # Process each detected plate
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Extract the region of interest (ROI) and save as an image
                plate_image = frame[y1:y2, x1:x2]
                timestamp = int(time.time())
                plate_image_path = f'plate_{timestamp}.jpg'
                cv2.imwrite(plate_image_path, plate_image)

                # Recognize the license plate using OpenALPR
                plate_text = extract_plate_text(plate_image_path)
                print(f'Detected plate: {plate_text}')

                # Display the recognized plate text
                cv2.putText(frame, f'Plate: {plate_text}', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                            2)

        # Display the frame with detections
        cv2.imshow('License Plate Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

