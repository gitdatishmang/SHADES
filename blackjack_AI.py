import torch
import cv2
import numpy as np
from mss import mss
from twilio.rest import Client

# Load the model with your custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestModel.pt')

# Define the class names
class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
               '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
               '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
               'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 
               'Qc', 'Qd', 'Qh', 'Qs']

# Twilio API credentials
account_sid = 'KEY'
auth_token = 'KEY'
client = Client(account_sid, auth_token)

# Set to keep track of seen classes
seen_classes = set()

# Function to send a WhatsApp message
def send_whatsapp_message(remaining_classes):
    message_body = f"All 42 classes have been seen. Remaining classes are: {remaining_classes}"
    message = client.messages.create(
        from_='whatsapp:Number',
        body=message_body,
        to='whatsapp:Number'  # Replace with your WhatsApp number
    )
    print(f"WhatsApp message sent: {message.sid}")

# Function to run inference on the live desktop screen
def run_inference_on_screen():
    # Define the screen capture area (you can adjust the 'mon' dictionary)
    mon = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}
    sct = mss()
    
    while True:
        # Capture the screen
        screen = np.array(sct.grab(mon))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
        
        # Run inference
        results = model(frame)
        
        # Get the detected class IDs and confidence scores
        predictions = results.pred[0]
        class_ids = predictions[:, -1].cpu().numpy()  # Move tensor to CPU and then convert to numpy
        confidences = predictions[:, 4].cpu().numpy()  # Get confidence scores
        
        # Filter classes with confidence above 90%
        for i, confidence in enumerate(confidences):
            if confidence > 0.45:  # Only consider detections with > 90% confidence
                detected_class = class_names[int(class_ids[i])]
                seen_classes.add(detected_class)
        
        # Print the detected class names
        print(f"Detected classes: {seen_classes}")
        print(f"Total unique classes seen: {len(seen_classes)}")
        
        # Check if we've seen 42 unique classes
        if len(seen_classes) >= 421:
            remaining_classes = set(class_names) - seen_classes
            print(f"Remaining classes: {remaining_classes}")
            send_whatsapp_message(remaining_classes)  # Send WhatsApp message here
            break
        
        # Draw the bounding boxes and labels on the frame
        results.render()  # Updates the frame with bounding boxes and labels
        
        # Display the frame
        cv2.imshow('Screen Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cv2.destroyAllWindows()

# Example usage
run_inference_on_screen()
