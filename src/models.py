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

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    return pd.read_csv(data_path)

def advanced_feature_engineering(df):
    df['tenure_income_ratio'] = df['tenure'] / (df['income'] + 1)
    df['age_employment_ratio'] = df['age'] / (df['employ'] + 1)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Middle', 'Senior', 'Elder'])
    df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
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
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])
    
    X_encoded = preprocessor.fit_transform(X)
    
    # Get the indices of categorical columns after encoding
    cat_indices = [i for i, _ in enumerate(preprocessor.named_transformers_['cat'].get_feature_names_out())]
    
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X_encoded, y)
    
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
    
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", np.mean(cv_scores))

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print("\nEnhanced Evaluation Metrics:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

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

def main():
    df = load_data('teleCust1000t.csv')
    df = advanced_feature_engineering(df)
    numeric_features = ['tenure', 'age', 'address', 'income', 'employ', 'reside', 'tenure_income_ratio', 'age_employment_ratio']
    categorical_features = ['region', 'marital', 'ed', 'retire', 'gender', 'age_group', 'income_group']  
    perform_pca(df, numeric_features)
    X, y, preprocessor = prepare_data(df, numeric_features, categorical_features)

    # Add the class balance verification and SMOTE here
    class_counts = df['custcat'].value_counts()
    print(class_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Customer Categories')
    plt.xlabel('Customer Category')
    plt.ylabel('Count')
    plt.show()

    # Apply SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(pd.Series(y_resampled).value_counts())

    # Continue with the rest of your code, using X_resampled and y_resampled
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    
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