# Recruiter Performance Analyzer
The following project is a recruiter performance analyzer. Utilizing the Random Forest Classifier Model, the model can
analyze the performance and give a verdict of whether the recruiter needs improvement, average, or high performing with a 99.2% accuracy.
<br><br>
Labels for training and testing data are generated via K-Means clustering to ensure that labels are created based on trends
in the testing data. After the model has been trained and tested for accuracy, a confusion matrix and feature importance will be displayed to the user.
<br><br>
To run this file, make sure to download/clone this repository and run the ```model.py``` file. Ensure that ```recruiter_data-2.csv``` is in the same directory.
If you would like to supply your own CSV file for training, ensure that the column headers are in the order: name, submissions, time_to_first_submission, requirements_submitted_to, submissions_to_interview_number, interview_to_offer_number. Additionally, make sure to change the file name that is being referenced to create the ```data``` pandas dataframe.