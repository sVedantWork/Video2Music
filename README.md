# Video2Music: What is this project ?

The main objective of this project is to recommend music based on emotions captured from a short frontal-face video of the user. This objective is achieved in two parts : 1) A computer vision pipeline which takes a short-video and returns top 3 predicted emotions, 2) A recommendation system to recommend music based on the captured emotions.

# What dataset and libraries are used ?

This project makes use of PyTorch framework with libraries like torch and torchvision, along with sciKit-learn for developing both the pipelines, evaluating,
and integrating them. Other than this, the PIL, Numpy, Pandas, Matplotlib, and Transformers (from Hugging Face) libraries are used for supporting the development
of the project and visualizing the results. The "FER_2013_Kaggle" dataset is used to develop the face classification pipeline while the "MusicCaps" dataset is used 
for training the recommendation system pipeline.

# How to use this program ?

To test the program:
1) Download Vid2Music.ipynb, Vid2MusicRecommendation.py, and Video_Classification_Pipeline.py and put them in a common folder.
2) Get the apprpriate model weights from the links in weights.txt and modify the path in Video_Classification_Pipeline.py for the same.
3) Upload a short frontal-face video and get some amazing music recommendations.

To train on your own datasets:
1) In addition to step 1 from above, download Img_Emotion_Classifier.py and re-train the image classifier model on your own dataset or even different models by
   updating the appropriate sections commented in the code.
2) Use this new saved model with the Video_Classification_Pipeline.
3) To train the recommendation system pipeline, update Vid2MusicRecommendation.py with the appropriate text-to-music dataset and use those new weights for future
   predictions.    

# Future Development ?

Currently, the face classification pipeline only achieves 67% accuracy on its task while SOTA on this dataset is 72% so, some hyperparameter tuning or experimenting
with different models is required. Further, more class-balanced datasets for both pipelines could help improve the performance significantly thus, more research in this 
area is necessary as the Vid2Music domain hasn't been researched as much though it has many useful applications. Lastly, a more robust and throughly tested 
recommendation system as well as better metrics for evaluating both pipelines could help standardize the results so more research in that area should be explored.
