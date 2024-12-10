import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report
from imblearn.over_sampling import SMOTENC
import optuna
import os

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df, numeric_features, categorical_features):
    # Separate features and labels
    X = df.drop('custcat', axis=1)
    y = df['custcat']

    # OneHotEncoding and Scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Fit and transform
    X_preprocessed = preprocessor.fit_transform(X)
    feature_names = (
        numeric_features + 
        list(preprocessor.named_transformers_['cat']
             .get_feature_names_out(categorical_features))
    )
    
    # Convert to DataFrame
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

    # SMOTENC to handle class imbalance
    cat_indices = list(range(len(numeric_features), X_preprocessed.shape[1]))
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X_preprocessed, y)

    return X_resampled, y_resampled, preprocessor

def build_model(input_dim, layers, units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    for _ in range(layers - 1):  # Add hidden layers
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer (number of classes = 4, for "custcat")
    model.add(Dense(4, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def objective(trial, X, y):
    # Suggest hyperparameters
    layers = trial.suggest_int('layers', 1, 3)
    units = trial.suggest_int('units', 32, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    epochs = 50

    # K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Build and train the model
        model = build_model(X_train.shape[1], layers, units, dropout_rate, learning_rate)
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                  epochs=epochs, batch_size=batch_size, verbose=0, 
                  callbacks=[early_stopping])
        
        # Evaluate on validation set
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(accuracy)

    return np.mean(cv_scores)

def optimize_hyperparameters(X, y):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    return study

def train_best_model(study, X, y):
    best_params = study.best_params
    model = build_model(
        input_dim=X.shape[1],
        layers=best_params['layers'],
        units=best_params['units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=best_params['batch_size'], 
              verbose=1, callbacks=[early_stopping])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load dataset
    df = load_data('teleCust.csv')

    # Define numeric and categorical features
    numeric_features = ['tenure', 'age', 'address', 'income', 'employ', 'reside']
    categorical_features = ['region', 'marital', 'ed', 'retire', 'gender']

    # Preprocess data
    X_resampled, y_resampled, preprocessor = preprocess_data(df, numeric_features, categorical_features)

    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Hyperparameter optimization
    study = optimize_hyperparameters(X_train, y_train)
    print("Best hyperparameters:", study.best_params)

    # Train best model
    best_model = train_best_model(study, X_train, y_train)

    # Evaluate best model
    evaluate_model(best_model, X_test, y_test)