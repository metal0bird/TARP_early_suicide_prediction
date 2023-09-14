## Risk Assessment System

In the project, we aimed to tackle the issue of early suicide prediction using data science and machine learning techniques. The project made use of the SuicideWatch dataset from Kaggle, which provided a large volume of relevant data for this problem. To achieve the highest possible accuracy in our predictions, we adopted a hybrid approach that combined deep learning models with a lexicon-based technique.

<img width="506" alt="Screenshot 2023-09-14 at 11 56 07 PM" src="https://github.com/metal0bird/TARP_early_suicide_prediction/assets/71923741/a98fd032-b58c-450d-a204-30711366d97e">

The deep learning models were trained on the dataset, and the lexicon-based approach was used to supplement the predictions made by the models. The combination of these two techniques allowed us to achieve a higher accuracy compared to if we had used only traditional machine learning algorithms.

<img width="641" alt="Screenshot 2023-09-14 at 11 56 13 PM" src="https://github.com/metal0bird/TARP_early_suicide_prediction/assets/71923741/dadd9da0-6b1f-4ae8-994d-723b0c8355a0">

Additionally, we designed a user interface that could be used to provide support to individuals who are potentially at risk of suicide. This interface allowed for easy and effective communication between the users and a support network, and it was integrated into the overall solution.

### Interface

  ![images](https://github.com/metal0bird/TARP_early_suicide_prediction/blob/main/assets/Screenshot%202023-07-14%20at%2010.07.49%20AM.png)

  ![image](https://github.com/metal0bird/TARP_early_suicide_prediction/blob/main/assets/Screenshot%202023-07-14%20at%2010.07.56%20AM.png)

  ![image](https://github.com/metal0bird/TARP_early_suicide_prediction/blob/main/assets/Screenshot%202023-07-14%20at%2010.08.08%20AM.png)

### Parametrics of models

<img width="658" alt="Screenshot 2023-09-14 at 11 56 26 PM" src="https://github.com/metal0bird/TARP_early_suicide_prediction/assets/71923741/b3199d21-293c-4e2d-8e5c-e9a4c0db2461">

### Results 

<img width="623" alt="Screenshot 2023-09-14 at 11 56 31 PM" src="https://github.com/metal0bird/TARP_early_suicide_prediction/assets/71923741/c27e1dcc-0153-4034-8126-e4049c6f0bf2">

- Evaluation Metrics: Performance of Bi-RNN and LSTM models can be assessed using metrics like accuracy, F1 score, precision, recall, and training time.
Training Time: Bi-RNN model demonstrates faster training compared to LSTM, as indicated by the training time graph.
- Accuracy and F1 Score: Bi-RNN slightly outperforms LSTM in terms of accuracy and F1 score, with a preference towards precision for Bi-RNN and recall for LSTM. Overall, Bi-RNN performs slightly better on this dataset, but model choice should align with the specific task and data.

In conclusion, our project demonstrated the potential of data science and machine learning techniques to address important social issues like early suicide prediction. The combination of deep learning models and lexicon-based approach allowed us to achieve improved accuracy and the user interface provided a means to support those in need.
