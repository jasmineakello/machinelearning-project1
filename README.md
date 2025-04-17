# Booking and Cancellion Machine learning Project

## ABSTRACT
We trained a machine learning model to predict whether a booking tenant would
cancel or keep their reservation. Using a training data set, the model was trained
and validated with the goal of optimizing the ROC- AUC score. The score evalu-
ates the model’s ability to distinguish between the cancellations and the successful
reservations. after testing multiple regression methods including the logistic re-
gression, we found that the logistic regression was out-performed by the random
classifier. This resulted in achieving the validation score of 0.94.
## 1 DATA PRE-PROCESSING
We started by loading the train.csv and test.csv files. The training data consisted of a
mixture of numerical and categorical characteristics, as well as the target variable label, indicating
cancellation (1) or not (0). No missing values or inconsistent formatting were present, so minimal
pre-processing was required. The column Id was excluded from the feature set since it did not have
a predictive value. Categorical features were already encoded as numerical integers, allowing direct
usage in tree-based models like Random Forest.

## 2 MODEL SELECTION
Two primary models were considered, Logistic Regression and Random Forest Classifier. Logistic
regression is a strong baseline model for binary classification tasks. We included standard scaling
as part of the pipeline. Random Forest Classifier is an ensemble model that aggregates multiple
decision trees and is known to handle complex patterns in data with minimal pre-processing. The
logistic regression model achieved a validation AUC of approximately 0.85. However, we had the
goal of exceeding a 0.90 AUC, prompting us to test a Random Forest model. The Random Forest
Classifier achieved a validation AUC of 0.94, making it the final model choice.
### 2.1 HYPERPARAMETER TUNING
The final model was trained using the following parameters:<br>
n_estimators = 100<br>
max_depth = None <br>
random_state = 42<br>
n_jobs = -1<br>
This configuration balanced performance and efficiency while preventing overfitting.

## 3 EXPERIMENT RESULTS
The following table summarizes the model performance:
The Random Forest model significantly outperformed logistic regression. It captured non-linear
relationships in the data and handled feature interactions more effectively.

The following table summarizes the model performance:
#### Model Validation AUC<br>
Logistic Regression 0.850<br>
Random Forest 0.940<br>

## 4 ANALYSIS
The logistic regression model provided a solid baseline with an AUC of approximately 0.85. This
indicates that the model has decent discriminative power, but it likely failed to capture complex non-
linear patterns and feature interactions present in the data. While logistic regression is interpretable
and fast, its performance plateaued despite attempts at regularization and hyperparameter tuning.
The Random Forest model significantly outperformed logistic regression, achieving a validation
AUC of 0.94. The high AUC score indicates that the model has excellent ability to rank reservations
by their likelihood of being canceled.
### 4.1 GRAPHICAL ANALYSIS
#### 4.1.1 LOGISTIC REGRESSION
The results in figure one, underline that a simple logistic regression model is an adequate choice for
the booking cancellation prediction task. The model converged quickly and yielded solid perfor-
mance (AUC  ̃0.84), showing that it can effectively identify linear patterns in the features to make
useful predictions. However, the plateau of performance at 84% AUC likely reflects this limitation.
The model has learned all the linearly separable structure in the dataset, and further gains may re-
quire a more expressive model. In summary, the logistic regression classifier appears well-suited for
this problem in terms of ease of training and robust generalization.
#### 4.1.2 RANDOM CLASSIFIER
In figure two, the ROC curve for the Random Classifier model yielded an Area Under the
Curve (AUC) of 0.94, indicating excellent overall distinguishing between cancellations and non-
cancellations. The training process shows how the model achieved this high AUC in a stable way.
Firstly, the validation AUC curve leveled off at 0.94 and stayed there means that this high discrim-
inatory power is consistently maintained across training. It corroborates that the model reaches a
reliable performance level and stays there. Together, these observations confirm that the ROC AUC
of 0.94 is a great result. This harmony between the ROC evaluation and the training/validation
curves boosts our trust in the model’s predictive ability and reliability.
## 5 CONCLUSION
In this project, we successfully developed and evaluated a machine learning model to predict whether
a booking tenant would cancel or keep their reservation. Moreover, this project highlights the im-
portance of both model selection (Logistic Regression and Random Classifier) and evaluation when
building predictive systems. The Random Forest model proved to be a powerful and appropriate
choice for understanding booking cancellation behavior, and the thorough validation process ensures
that it can be confidently applied to new reservations. Future improvements could involve testing
gradient boosting models or incorporating more contextual features to further enhance performance

## 6 MISCELLANEOUS
• All code was written in Python using the scikit-learn library.<br>
• Jupyter Notebook was used for development and experimentation.

## 7 CITATIONS, FIGURES, TABLES, REFERENCES
This project was built using the starter kit by Professor Xiaotian (Max) Han
