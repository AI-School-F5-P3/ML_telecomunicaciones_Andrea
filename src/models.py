import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
import optuna
import pickle

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    df = pd.read_csv(data_path)
    print("Columns in the dataset:", df.columns)
    return df

def perform_pca(df, numeric_features):
    X = df[numeric_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum())
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

def prepare_data(df, numeric_features, categorical_features):
    X = df.drop('custcat', axis=1)
    y = df['custcat']
    print("Features used for training:", numeric_features)

    # Ensure categorical features exist in the data
    available_categorical_features = [col for col in categorical_features if col in X.columns]
    print("Validated categorical features:", available_categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), available_categorical_features)
        ])
    
    # Fit the preprocessor to get the correct feature names
    X_encoded = preprocessor.fit_transform(X)
    feature_names = (numeric_features + 
                     list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(available_categorical_features)))
    
    # Convert to DataFrame with correct column names before SMOTE
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)
    
    # Get indices of categorical features for SMOTENC
    cat_indices = list(range(len(numeric_features), X_encoded.shape[1]))
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    
    X_resampled, y_resampled = smote_nc.fit_resample(X_encoded_df, y)
    
    # Convert back to DataFrame with proper column names
    X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
    y_resampled = pd.Series(y_resampled, name='custcat')
    
    return X_resampled, y_resampled, preprocessor

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(preprocessor, classifier):
    return Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])

def evaluate_model(model, X, y, X_test, y_test):
    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        cv_scores.append(model.score(X_val, y_val))
    
    # Cross-validation results
    mean_cv_score = np.mean(cv_scores)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {mean_cv_score}")
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nEnhanced Evaluation Metrics:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"Test Accuracy: {test_accuracy}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Overfitting measure
    overfitting_score = mean_cv_score - test_accuracy
    print("\nOverfitting Score (Mean CV Score - Test Accuracy):")
    print(f"{overfitting_score:.4f}")

def objective(trial, X, y, preprocessor):
    
    classifier_name = trial.suggest_categorical('classifier', ['RF', 'GB', 'XGB'])
    
    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        )
    elif classifier_name == 'GB':
        classifier = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        )
    else:  # XGB
        classifier = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        )
    
    model = create_model(preprocessor, classifier)
    
    # Implement stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    return np.mean(scores)

def optimize_hyperparameters(X, y, preprocessor):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=100)
    return study

def train_best_model(study, X_train, y_train, preprocessor):
    best_params = study.best_trial.params
    best_classifier = None

    if best_params['classifier'] == 'RF':
        best_classifier = RandomForestClassifier(**{k: v for k, v in best_params.items() if k != 'classifier'})
    elif best_params['classifier'] == 'GB':
        best_classifier = GradientBoostingClassifier(**{k: v for k, v in best_params.items() if k != 'classifier'})
    else:
        best_classifier = KNeighborsClassifier(**{k: v for k, v in best_params.items() if k != 'classifier'})

    best_model = create_model(preprocessor, best_classifier)
    best_model.fit(X_train, y_train)
    return best_model

def save_model(model, model_name, directory='models'):
    # Ensure the directory exists
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model as a pickle file
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model '{model_name}' saved at {model_path}")

def main():
    # Load the data
    df = load_data('teleCust.csv')
    
    # Define numeric and categorical features (as present in the original dataset)
    numeric_features = ['tenure', 'age', 'address', 'income', 'employ', 'reside']
    categorical_features = ['region', 'marital', 'ed', 'retire', 'gender']
    
    # Perform PCA 
    perform_pca(df, numeric_features)
    
    # Print original class distribution
    class_counts = df['custcat'].value_counts()
    print("Original class distribution:")
    print(class_counts)
    
    # Print original and resampled class distributions
    print("Original class distribution:", df['custcat'].value_counts())

    # Prepare data 
    X_resampled, y_resampled, preprocessor = prepare_data(df, numeric_features, categorical_features)

    # Check resampled dataset
    print("Resampled X columns:", X_resampled.columns)
    
    # Print resampled class distribution
    print("\nResampled class distribution:")
    print(pd.Series(y_resampled).value_counts())
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    
    # Models without the preprocessor, as the data has already been preprocessed
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)
    knn_model = KNeighborsClassifier()

    models = [dt_model, rf_model, gb_model, knn_model]
    model_names = ['Decision_Tree', 'Random_Forest', 'Gradient_Boosting', 'K-Nearest_Neighbors']

    # Evaluate and save models
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}:")
        model.fit(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Save the trained model
        save_model(model, name, directory='models')
    
    # Modify train_best_model function as well
    def train_best_model(study, X_train, y_train, preprocessor):
        best_params = study.best_trial.params
        best_classifier = None

        if best_params['classifier'] == 'RF':
            best_classifier = RandomForestClassifier(**{k: v for k, v in best_params.items() if k != 'classifier'})
        elif best_params['classifier'] == 'GB':
            best_classifier = GradientBoostingClassifier(**{k: v for k, v in best_params.items() if k != 'classifier'})
        else:
            best_classifier = KNeighborsClassifier(**{k: v for k, v in best_params.items() if k != 'classifier'})

        best_model = Pipeline([
            ('preprocessor', preprocessor), 
            ('classifier', best_classifier)
        ])
        best_model.fit(X_train, y_train)
        return best_model

if __name__ == "__main__":
    main()