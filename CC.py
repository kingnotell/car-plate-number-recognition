import cv2
from ultralytics import YOLO
import string
import easyocr
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# Initialize license plate detection tracking
detected_plates = defaultdict(
    lambda: {'count': 0, 'score': 0, 'text': '', 'entry_time': None, 'last_detected_time': None, 'exit_time': None})

# Load balance information from CSV
balance_dict = {}
try:
    balance_df = pd.read_csv('/home/big/Documents/T/balance.csv')
    balance_dict = pd.Series(balance_df.Balance.values, index=balance_df['License Plate']).to_dict()
except FileNotFoundError:
    print("Error: balance.csv file not found. Continuing without balance information.")


def format_license_row(text, is_number_row):
    """
    Format the license plate text row by converting characters using the mapping dictionaries.
    """
    text = text.strip().upper().replace(' ', '').replace('_', '').replace('.', '').replace('[', '').replace(']',
                                                                                                            '').replace(
        '"', '').replace('~', '')
    formatted_text = ''
    for char in text:
        if is_number_row:
            formatted_text += dict_char_to_int.get(char, char)
        else:
            formatted_text += char  # Do not convert characters for the letter row
    return formatted_text


def standardize_plate(text):
    """
    Standardize the license plate text to reduce variations.
    """
    return text.replace('€', 'C').replace('J', '3')


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    """
    detections = reader.readtext(license_plate_crop)
    print(f"OCR Detections: {detections}")

    if len(detections) == 1:
        text_row = detections[0][1].upper().replace(' ', '')
        score_row = detections[0][2]
        print(f"Detected Text: {text_row}, Score: {score_row}")

        formatted_text = format_license_row(text_row, is_number_row=False)  # Assume single row as letters + numbers

        if score_row > 0.5:
            print(f"Formatted License Plate: {formatted_text}")
            return formatted_text, score_row
    elif len(detections) >= 2:
        text_row_1, score_row_1 = detections[0][1].upper().replace(' ', ''), detections[0][2]
        text_row_2, score_row_2 = detections[1][1].upper().replace(' ', ''), detections[1][2]

        print(f"Detected Text Row 1: {text_row_1}, Score: {score_row_1}")
        print(f"Detected Text Row 2: {text_row_2}, Score: {score_row_2}")

        formatted_row_1 = format_license_row(text_row_1, is_number_row=False)
        formatted_row_2 = format_license_row(text_row_2, is_number_row=True)

        combined_text = formatted_row_1 + formatted_row_2
        combined_text = standardize_plate(combined_text)
        average_score = (score_row_1 + score_row_2) / 2

        if average_score > 0.5:
            print(f"Formatted License Plate: {combined_text}")
            return combined_text, average_score

    return None, None


def save_to_csv(plates, filename='detected_license_plates.csv'):
    """
    Save the detected license plates to a CSV file.
    """
    data = [{'License Plate': plate['text'], 'Best Score': plate['score'], 'Entry Time': plate['entry_time'],
             'Exit Time': plate['exit_time']} for plate in plates]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(data)} plates to {filename}")


def append_to_database(temp_csv='detected_license_plates.csv', db_csv='license_plate_database.csv'):
    """
    Append the data from the temporary CSV file to the database CSV file.
    """
    print(f"Appending data from {temp_csv} to {db_csv}")

    # Load the temporary CSV data
    temp_df = pd.read_csv(temp_csv)

    try:
        # Load the database CSV data
        db_df = pd.read_csv(db_csv)
        # Append the new data to the database
        updated_db_df = pd.concat([db_df, temp_df], ignore_index=True)
        # Assign unique IDs
        updated_db_df['ID'] = range(1, len(updated_db_df) + 1)
    except FileNotFoundError:
        # If the database file does not exist, use the temporary data as the initial database
        updated_db_df = temp_df
        # Assign unique IDs
        updated_db_df['ID'] = range(1, len(updated_db_df) + 1)

    # Save the updated database back to the CSV file
    updated_db_df.to_csv(db_csv, index=False)
    print(f"Appended data to {db_csv} and saved {len(updated_db_df)} entries")

    # Clear the temporary CSV file
    open(temp_csv, 'w').close()
    print(f"Cleared the temporary file {temp_csv}")


def is_plate_in_database(plate_text, db_csv='license_plate_database.csv'):
    """
    Check if a license plate text is already in the database CSV file.
    """
    try:
        db_df = pd.read_csv(db_csv)
        return plate_text in db_df['License Plate'].values
    except FileNotFoundError:
        # If the database file does not exist, return False
        return False


def adjust_confidence_if_balance_found(license_text, original_score):
    """
    Adjust the confidence score to 1.0 (100%) if the license plate has a balance entry.
    """
    if license_text in balance_dict:
        return 1.0
    return original_score


def check_and_save_plate(license_text, score):
    """
    Check if the license plate has a balance and is not already registered. If so, save it.
    """
    if license_text in balance_dict:
        plate_info = detected_plates[license_text]
        current_time = datetime.now()

        if plate_info['entry_time'] is None:
            # Register entry time
            plate_info['entry_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            plate_info['last_detected_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            plate_info['score'] = score
            plate_info['text'] = license_text
            save_to_csv([plate_info])
            append_to_database()
            print(f"Registered entry for license plate: {license_text} with score: {score}")
        else:
            # Update last detected time
            last_detected_time = datetime.strptime(str(plate_info['last_detected_time']), '%Y-%m-%d %H:%M:%S')
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            current_time_dt = datetime.strptime(current_time_str, '%Y-%m-%d %H:%M:%S')
            time_diff = current_time_dt - last_detected_time

            # Check if the detection is within 10 seconds of the initial entry time registration
            entry_time = datetime.strptime(plate_info['entry_time'], '%Y-%m-%d %H:%M:%S')
            if (current_time_dt - entry_time).total_seconds() > 10:
                if time_diff.total_seconds() > 10:
                    plate_info['last_detected_time'] = current_time_str

                    if time_diff.total_seconds() >= 10:  # Changed from 900 seconds (15 minutes) to 10 seconds for testing
                        # Deduct balance before registering exit time if more than 10 seconds have passed
                        balance_dict[license_text] -= 3
                        plate_info['exit_time'] = current_time_str
                        save_to_csv([plate_info])
                        append_to_database()
                        print(f"Registered exit for license plate: {license_text} with score: {score}")
                    else:
                        # Only update last detected time within 10 seconds
                        plate_info['exit_time'] = current_time_str
                        save_to_csv([plate_info])
                        append_to_database()
                        print(f"Updated last detected time for license plate: {license_text}")


# Load YOLO model
model = YOLO('license_plate_detector.pt')

# Start capturing from IP camera
#cap = cv2.VideoCapture('http://192.168.100.4:4747/video')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open camera feed.")
    exit()

frame_count = 0
confidence_threshold = 0.5  # Set a confidence threshold

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera feed.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Loop over the detected results and draw bounding boxes
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop the license plate region
                license_plate_crop = frame[y1:y2, x1:x2]

                # Save the cropped image for debugging
                cv2.imwrite(f'license_plate_crop_{frame_count}.jpg', license_plate_crop)

                # Read the license plate text
                license_text, score = read_license_plate(license_plate_crop)
                if license_text and score > confidence_threshold:
                    # Adjust confidence if balance found
                    score = adjust_confidence_if_balance_found(license_text, score)
                    confidence_percentage = score * 100
                    balance = balance_dict.get(license_text, "N/A")
                    cv2.putText(frame, f'{license_text} ({confidence_percentage:.2f}%) Balance: {balance}',
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    print(
                        f"License Plate: {license_text}, Confidence: {confidence_percentage:.2f}%, Balance: {balance}")

                    # Check and save the license plate if it has a balance and is not already registered
                    check_and_save_plate(license_text, score)

                else:
                    print("No valid license plate text detected or confidence too low")

    # Display the resulting frame
    cv2.imshow('License Plate Detection', frame)

    # Increment frame count for unique file names
    frame_count += 1

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
