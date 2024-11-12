# Osteoarthritis

This model was fine-tuned from ResNet18 with [The Knee Osteoarthritis Dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity/data) using Pytorch in 25 epochs, Adam optimizer, and a learning rate 0.001. The training accuracy is 94.41% while the evaluation with test set accuracy is 57%.

Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.55      0.65       639
           1       0.29      0.49      0.36       296
           2       0.59      0.57      0.58       447
           3       0.71      0.73      0.72       223
           4       0.82      0.71      0.76        51

    accuracy                           0.57      1656
   macro avg       0.64      0.61      0.61      1656
weighted avg       0.64      0.57      0.59      1656