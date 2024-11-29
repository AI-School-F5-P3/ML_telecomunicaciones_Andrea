import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import optuna

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    return pd.read_csv(data_path)

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
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    X = df.drop('custcat', axis=1)
    y = df['custcat']
    return X, y, preprocessor

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(preprocessor, classifier):
    return Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Predictions on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Classification Report
    cr = classification_report(y_test, y_test_pred)
    
    # Print results
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    # Calculate overfitting
    overfitting = train_accuracy - test_accuracy
    print(f"\nOverfitting: {overfitting:.4f}")

def objective(trial, X, y, preprocessor):
    classifier_name = trial.suggest_categorical('classifier', ['RF', 'GB', 'KNN'])
    
    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 10, 300),
            max_depth=trial.suggest_int('max_depth', 2, 32),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        )
    elif classifier_name == 'GB':
        classifier = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 10, 300),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            max_depth=trial.suggest_int('max_depth', 1, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        )
    else:  # KNN
        classifier = KNeighborsClassifier(
            n_neighbors=trial.suggest_int('n_neighbors', 1, 20),
            weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
            p=trial.suggest_int('p', 1, 2)
        )
    
    model = create_model(preprocessor, classifier)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return score.mean()

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

def main():
    df = load_data('teleCust1000t.csv')
    numeric_features = ['tenure', 'age', 'address', 'income', 'employ', 'reside']
    categorical_features = ['region', 'marital', 'ed', 'retire', 'gender']    
    perform_pca(df, numeric_features)
    X, y, preprocessor = prepare_data(df, numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    dt_model = create_model(preprocessor, DecisionTreeClassifier(random_state=42))
    rf_model = create_model(preprocessor, RandomForestClassifier(random_state=42))
    gb_model = create_model(preprocessor, GradientBoostingClassifier(random_state=42))
    knn_model = create_model(preprocessor, KNeighborsClassifier())
    
    models = [dt_model, rf_model, gb_model, knn_model]
    model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'K-Nearest Neighbors']
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}:")
        model.fit(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)
    
    study = optimize_hyperparameters(X, y, preprocessor)
    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('    {}: {}'.format(key, value))
    
    best_model = train_best_model(study, X_train, y_train, preprocessor)
    evaluate_model(best_model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()