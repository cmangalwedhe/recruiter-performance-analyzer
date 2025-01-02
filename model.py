import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# creating AI model to review performance of recruiters
class RecruitmentPerformanceModel:
    def __init__(self, random_state=24):
        self.random_state = random_state
        self.scaler = StandardScaler()

        # utilizes Random Forest Model
        self.model = RandomForestClassifier(
            n_estimators=300, # number of decision trees created to come to decision
            max_depth=None, # depth of each tree
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=self.random_state,
            class_weight='balanced', # determines weight automatically based on relationship
            bootstrap=True,
            oob_score=True
        )

    def preprocess_data(self, data):
        # processing data for model to interpret and train
        df = data.copy()

        # create more variables to decide performance more clearly
        df['placement_ratio'] = df['placements'] / df['submissions'].clip(lower=1)
        df['submission_per_req'] = df['submissions'] / df['requirements_submitted_to'].clip(lower=1)
        df['efficiency_score'] = 1 / (df['time_to_first_submission'] + 1)

        # features/headers used in csv file
        # self.features = [
        #    'submissions', 'placements', 'time_to_first_submission', 'submissions_to_interview_number',
        #    'interview_to_offer_number', 'placement_ratio', 'submission_per_req', 'efficiency_score'
        # ]

        self.features = {
            'submissions': 0,
            'placements': 0,
            'time_to_first_submission': 0,
            'submissions_to_interview_number': 0.20,
            'interview_to_offer_number': 0.25,
            'placement_ratio': 0.30,
            'submission_per_req': 0.125,
            'efficiency_score': 0.125
        }

        """self.features = {
            'placement_ratio': 0.25,
            'submissions_to_interview_number': 0.65,
            'interview_to_offer_number': 0.10
        }"""

        return df

    # labels to describe score and performance label set by weights of important decision-making categories
    """def create_labels(self, data):
        df = data.copy()
        composite_score = 0

        for k, v in self.features.items():
            composite_score += df[k] * v

        df['composite_score'] = composite_score
        df['performance_label'] = pd.qcut(df['composite_score'], q=3, labels=['Needs Improvement', 'Average', 'High Performer'])
        return df['performance_label']"""

    def create_labels(self, data):
        df = data.copy()

        # Select key metrics for clustering
        metrics = df[['placement_ratio', 'interview_to_offer_number', 'efficiency_score']]

        # Normalize the data
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(metrics)

        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(normalized_metrics)

        # Map clusters to performance labels
        centroids = kmeans.cluster_centers_
        # Calculate average performance score for each cluster
        cluster_scores = [(np.mean(centroid), i) for i, centroid in enumerate(centroids)]
        # Sort clusters by performance (lowest to highest)
        sorted_clusters = sorted(cluster_scores, key=lambda x: x[0])

        # Create mapping dictionary
        cluster_mapping = {
            sorted_clusters[0][1]: 'Needs Improvement',
            sorted_clusters[1][1]: 'Average',
            sorted_clusters[2][1]: 'High Performer'
        }

        # Map cluster numbers to labels
        return pd.Series([cluster_mapping[c] for c in clusters])

    def train(self, data, verbose=True):
        processed_data = self.preprocess_data(data)
        labels = self.create_labels(processed_data)

        X = processed_data[list(self.features.keys())]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=self.random_state)

        # normalizes and scales train and test models
        # model assumes that the data is normalized and helps reduce variance in the data
        # fit calculates the mean and standard deviation
        # transform applies normalization by subtracting mean and dividing std dev
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # builds forest of decision trees
        self.model.fit(X_train_scaled, y_train)

        # prediction
        y_pred = self.model.predict(X_test_scaled)

        if verbose:
            # describe and visualize efficacy of model
            print("\nModel Performance:")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            # Feature importance
            self.plot_feature_importance()

            # Confusion matrix
            self.plot_confusion_matrix(y_test, y_pred)

        return {
            'feature_importance': dict(zip(self.features, self.model.feature_importances_)),
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred)
        }


    def predict(self, new_data):
        # process data of new recruiters and create model
        processed_data = self.preprocess_data(new_data)
        X = processed_data[list(self.features.keys())]


        # normalize and scale model
        X_scaled = self.scaler.transform(X)

        # give prediction and predict (should be trained before doing so)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # create results dataframe which is copy of data passed in
        results = new_data.copy()
        results['predicted_performance'] = predictions

        # for every label, add probability of each class in model
        for i, label in enumerate(self.model.classes_):
            results[f'{label}_probability'] = probabilities[:, i]

        return results

    def plot_feature_importance(self):
        """
        Plot feature importance
        """
        importance = pd.DataFrame({
            'feature': list(self.features.keys()),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model.classes_,
                    yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()


# Create sample data
data = pd.read_csv("recruiter_data-2.csv")

# Create and train classifier
classifier = RecruitmentPerformanceModel()
results = classifier.train(data)

# Make predictions for new recruiters
new_recruiter = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'submissions': [25, 8],
    'placements': [8, 2],
    'time_to_first_submission': [2.5, 0.8],
    'requirements_submitted_to': [35, 12],
    'submissions_to_interview_number': [0.6, 3.7],
    'interview_to_offer_number': [0.4, 0]
})

predictions = classifier.predict(new_recruiter)
print("\nPredictions for new recruiters:")
print(predictions[['name', 'predicted_performance']])
