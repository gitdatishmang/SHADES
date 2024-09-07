import torch
import cv2
import numpy as np
from mss import mss
from collections import deque
from itertools import combinations
from time import time, sleep
import threading
import base64
from openai import OpenAI
from deepface import DeepFace
from twilio.rest import Client

# Load the model with your custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestModel.pt')

# Define the class names (Texas Hold'em poker cards)
class_names = ['Tc', 'Td', 'Th', 'Ts', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
               '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
               '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
               'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 
               'Qc', 'Qd', 'Qh', 'Qs']

# Hand rankings (simplified for Texas Hold'em)
hand_ranks = {
    'High Card': 0,
    'One Pair': 1,
    'Two Pair': 2,
    'Three of a Kind': 3,
    'Straight': 4,
    'Flush': 5,
    'Full House': 6,
    'Four of a Kind': 7,
    'Straight Flush': 8,
    'Royal Flush': 9
}

# Set to keep track of seen classes
seen_classes = set()
buffer_time = 10  # Buffer time in seconds
seen_cards_buffer = deque(maxlen=buffer_time)

# OpenAI client
client = OpenAI()

# Shared flag to stop threads
stop_threads = False

# Shared state variables
current_poker_odds = None
current_suggested_move = None
current_emotion = None

# Lock for thread-safe updates
state_lock = threading.Lock()

account_sid = 'KEY'
auth_token = 'KEY'
client = Client(account_sid, auth_token)

# Function to evaluate the hand strength
def evaluate_hand(cards):
    # Simplified hand evaluation logic
    rank_count = {rank: 0 for rank in '23456789TJQKA'}
    suit_count = {suit: 0 for suit in 'cdhs'}
    
    for card in cards:
        rank = card[:-1]  # Strip the suit (e.g., 'Ac' -> 'A')
        suit = card[-1]   # Get the suit (e.g., 'Ac' -> 'c')
        rank_count[rank] += 1
        suit_count[suit] += 1
    
    counts = list(rank_count.values())
    is_flush = any(count >= 5 for count in suit_count.values())
    sorted_ranks = sorted(rank_count.keys(), key=lambda x: '23456789TJQKA'.index(x))
    
    is_straight = False
    for i in range(len(sorted_ranks) - 4):
        if sorted_ranks[i:i + 5] == sorted_ranks[i:i + 5]:
            is_straight = True
    
    if is_flush and is_straight:
        if 'A' in sorted_ranks:
            return 'Royal Flush'
        return 'Straight Flush'
    if 4 in counts:
        return 'Four of a Kind'
    if 3 in counts and 2 in counts:
        return 'Full House'
    if is_flush:
        return 'Flush'
    if is_straight:
        return 'Straight'
    if 3 in counts:
        return 'Three of a Kind'
    if counts.count(2) == 2:
        return 'Two Pair'
    if 2 in counts:
        return 'One Pair'
    return 'High Card'

# Function to calculate basic odds based on hand strength
def calculate_poker_odds(cards_in_hand):
    best_hand = evaluate_hand(cards_in_hand)
    odds = hand_ranks[best_hand] / 9  # Simplified odds calculation
    return f"{best_hand} ({odds * 100:.2f}% chance)"

# Function to run inference on the live desktop screen
def run_inference_on_screen():
    global stop_threads, current_poker_odds
    mon = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}
    sct = mss()
    last_print_time = time()
    
    while not stop_threads:
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
        current_seen_classes = set()
        for i, confidence in enumerate(confidences):
            if confidence > 0.30:  # Only consider detections with > 90% confidence
                detected_class = class_names[int(class_ids[i])]
                current_seen_classes.add(detected_class)
        
        # Update buffer with currently seen classes
        seen_cards_buffer.append((time(), current_seen_classes))
        
        # Remove cards that were seen more than 10 seconds ago
        while seen_cards_buffer and time() - seen_cards_buffer[0][0] > buffer_time:
            seen_cards_buffer.popleft()
        
        # Combine all cards seen within the buffer time
        combined_seen_classes = set().union(*[cards for timestamp, cards in seen_cards_buffer])
        
        # Calculate poker odds based on seen cards every 20 seconds
        current_time = time()
        if current_time - last_print_time >= 20 and len(combined_seen_classes) >= 5:  # Minimum 5 cards needed for a valid poker hand
            odds = calculate_poker_odds(combined_seen_classes)
            
            # Update the shared state variable for poker odds
            with state_lock:
                current_poker_odds = odds
            last_print_time = current_time
        
        # Draw the bounding boxes and labels on the frame
        results.render()  # Updates the frame with bounding boxes and labels
        
        # Display the frame
        cv2.imshow('Screen Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break
    
    # Release resources
    cv2.destroyAllWindows()

# Function to capture screenshot every 20 seconds and send it to OpenAI
def capture_and_send_screenshot():
    global stop_threads, current_suggested_move
    mon = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}
    sct = mss()
    
    while not stop_threads:
        # Capture the screen
        screen = np.array(sct.grab(mon))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
        
        # Convert the image to base64
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Send the image to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "I am playing a game of poker. Can you tell if I should raise, call next card, or fold?\n"
                        }
                    ]
                },
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Reply back with a short sentence. If the photo received has no playing cards in it respond with nothing."
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        with state_lock:
            current_suggested_move = response.choices[0].message.content
        
        # Wait for 20 seconds before capturing the next screenshot
        for _ in range(20):
            if stop_threads:
                break
            sleep(1)

# Function to analyze emotions with DeepFace in a separate thread
def analyze_emotions():
    global stop_threads, current_emotion
    with mss() as sct:
        monitor = sct.monitors[1]  # Full screen
        while not stop_threads:
            # Capture the screen
            screen_shot = sct.grab(monitor)

            # Convert the screen shot to a numpy array (BGR format)
            frame = np.array(screen_shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Analyze the frame using DeepFace
            objs = DeepFace.analyze(
                img_path=frame, 
                actions=['emotion'],
                enforce_detection=False  # Skip the detection step if no face is found
            )
            
            # Update the shared state variable for the detected emotion
            if objs:
                with state_lock:
                    current_emotion = objs[0]['dominant_emotion']

            # Display the resulting frame
            cv2.imshow('Emotion Detection', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True
                break

    cv2.destroyAllWindows()

# Function to send an email every 30 seconds with the current state
def send_email():
    global stop_threads
    while not stop_threads:
        with state_lock:
            email_message = (f"Poker Odds: {current_poker_odds}\n"
                             f"Suggested Move: {current_suggested_move}\n"
                             f"Detected Emotion: {current_emotion}")
        
        # Create the SMTP session
        try:
            message = client.messages.create(
            from_='whatsapp:Number',
            body=email_message,
            to='whatsapp:Number'
            )

        except Exception as e:
            print(f"Error sending email: {e}")

        sleep(30)  # Wait for 30 seconds before sending the next email


# Function to display the updated states
def display_state():
    global stop_threads
    while not stop_threads:
        with state_lock:
            if current_poker_odds:
                print(f"Current Poker Odds: {current_poker_odds}")
            if current_suggested_move:
                print(f"Suggested Move: {current_suggested_move}")
            if current_emotion:
                print(f"Detected Emotion: {current_emotion}")
        sleep(1)  # Update every second

# Start the inference, screenshot capture, emotion analysis, and email threads
if __name__ == "__main__":
    # Start the inference thread
    inference_thread = threading.Thread(target=run_inference_on_screen)
    inference_thread.start()
    
    # Start the screenshot capture thread
    screenshot_thread = threading.Thread(target=capture_and_send_screenshot)
    screenshot_thread.start()

    # Start the emotion analysis thread
    emotion_thread = threading.Thread(target=analyze_emotions)
    emotion_thread.start()
    
    # Start the state display thread
    display_thread = threading.Thread(target=display_state)
    display_thread.start()

    # Start the email sending thread
    email_thread = threading.Thread(target=send_email)
    email_thread.start()

    # Wait for all threads to finish (which will be when 'q' is pressed)
    inference_thread.join()
    screenshot_thread.join()
    emotion_thread.join()
    display_thread.join()
    email_thread.join()
