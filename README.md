# Heart Disease Prediction

A Machine Learning web application that predicts the likelihood of heart disease using Logistic Regression.
link - https://huggingface.co/spaces/grogu29/heart-disease-prediction

## Features

- Heart Disease Prediction
- Confidence Score
- Prediction Probability Pie Chart
- Personalized Prevention Tips

Built using:

- Python
- Scikit-learn
- Gradio

Points to note:
- Why we didn't choose the decision tree over LR?
- Its 80.3% vs 78.7% lead was one patient out of 61, so it may just be luck of that split.
- It's unstable, as described above, so that small lead might not repeat with another split.
- Its probabilities are coarse. With depth 4, it has at most 16 leaves, so the app's confidence score would only take a few values.

How can decision tree stability be improved? 
- by using random forest , it is a collection of many decision trees, and lot of them might give different outputs , so they balance each other out and hence create stability.
