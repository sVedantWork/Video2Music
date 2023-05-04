import cv2
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image

def get_model():
    model = models.vgg19(weights='VGG19_Weights.DEFAULT')
    num_features = model.classifier[-1].in_features # num of features in last fully connected model layer
    model.classifier[-1] = nn.Linear(num_features, 7) # 7 unique emotions in both train and test folders resp
    return model



def predict_emotions(video_file_path, model, device, emotion_labels):
    # Load the video
    cap = cv2.VideoCapture(video_file_path)

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the list of emotions
    emotions = list(emotion_labels.keys())
 
    idx_to_class = {v: k for k, v in emotion_labels.items()}
    

    # Initialize the emotion count dictionary
    emotion_count = {emotion: 0 for emotion in emotions}

    model.to(device)

    # Loop over the frames of the video
    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # Preprocess the frame
        frame = transforms.Grayscale(num_output_channels=3)(frame) # Pre-trained model expects RGB
        frame = transforms.Resize((224, 224))(frame) # Want a small standard image size to reduce load on GPU
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(frame) # Normalize(mean_pixel_val, std_pixel_val)
        frame = frame.unsqueeze(0)

        frame = frame.to(device)

        # Pass the frame through the model to obtain the predicted emotion
        with torch.no_grad():
            outputs = model(frame)
            _, predicted = torch.max(outputs.data, 1)
            predicted_emotion = idx_to_class[predicted.item()]

        # Increment the count for the predicted emotion
        emotion_count[predicted_emotion] += 1

        # Wait for the specified amount of time before processing the next frame
        cv2.waitKey(1000 // fps)

    # Release the video capture
    cap.release()

    # Get the top 3 predicted emotions
    top_3_emotions = sorted(emotion_count.items(), key=lambda x: x[1], reverse=True)[:3]

    return top_3_emotions


def main(vid_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =  get_model()
    model.load_state_dict(torch.load('/home/vedant02/SchoolWork/DATA/Best_Model/vgg_modified.pth'))
    emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    emotions = predict_emotions(vid_file_path, model=model, device=device, emotion_labels=emotion_dict)
    print(emotions)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_file_path>")
    else:
        vid_file_path = sys.argv[1]
        main(vid_file_path)