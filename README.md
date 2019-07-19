# Classification Methods-ChurnDataset


"Churn rate, in its broadest sense, is a measure of the number of individuals or items moving out of a collective group over a specific period" -WikiPedia

This dataset contatains iformation about customers of a  such as thei demography as well as phone service details and finally whether they left the service (churn=Yes) or stayed (churn=No). The dataset can be found and downloaded from the following link:

https://www.kaggle.com/blastchar/telco-customer-churn

At first data cleaning is perfomed to make sure there are no missing values.

Then, data exploratory involves a function that plots two pie charts for churned and not churned categories. The input of this function is column name. For example, when passing in "gender", it outputs two pie charts one for churned and one for not churned, both sliced by gender.

Three different Classification Techniques are explored to model the data: KNN, decision tree and logistic regression. In each case the accuracy of the model is tested by the evaluation metrics specific for that model. According to accuracy metrics, decision tree is the most accurate of all with an accuracy of 0.794.
