import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import optuna
from metaflow import FlowSpec, step, Parameter, current, card
from metaflow.cards import Markdown, Table, Image
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
def load_data_helper(filepath):
    """ Loads the weatherAUS.csv dataset... """
    print(f"Attempting to load data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f"Raw dataset loaded successfully. Initial shape: {df.shape}")
        cols_to_drop = ['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
        df = df.drop(columns=cols_to_drop, errors='ignore') 
        print(f"Dropped initial columns ({', '.join(cols_to_drop)}). Shape after drop: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading or performing initial processing on data from {filepath}: {e}")
        raise

def split_data_helper(df, target_column, test_ratio, val_ratio, random_state=42):
    """ Splits data into train, validation, and test sets... """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    df_clean_target = df.dropna(subset=[target_column]).copy()
    print(f"Shape after dropping rows with missing target ('{target_column}'): {df_clean_target.shape}")

    if df_clean_target.empty:
        raise ValueError("No data remaining after dropping rows with missing target.")

    X = df_clean_target.drop(target_column, axis=1)
    y = df_clean_target[target_column].map({'Yes': 1, 'No': 0})

    if y.isnull().any():
         raise ValueError("Target column contains NaN after mapping 'Yes'/'No'. Check input data for unexpected values.")
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1 (exclusive).")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")
    if test_ratio + val_ratio >= 1:
        raise ValueError("The sum of test_ratio and val_ratio must be less than 1.")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    print(f"Split test set: {X_test.shape[0]} samples.")
    if (1 - test_ratio) == 0:
         raise ValueError("Cannot calculate validation split ratio if test_ratio is 1.")
    relative_val_ratio = val_ratio / (1 - test_ratio)
    if not (0 < relative_val_ratio < 1):
         raise ValueError(f"Calculated relative validation ratio ({relative_val_ratio:.3f}) is not between 0 and 1. "
                          f"Check original test_ratio ({test_ratio}) and val_ratio ({val_ratio}).")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_ratio, random_state=random_state, stratify=y_temp
    )
    print(f"Split train/validation set. Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples.")

    print(
        f"Final data split shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


class MetaflowTrainingPipeline(FlowSpec):
    
    test_split_ratio = Parameter('test_split',
                                 help='Fraction of data for the test set.',
                                 default=0.2)
    validation_split_raio = Parameter('val_split',
                                      help='Fraction of the *original* data for the validation set.',
                                      default=0.2)
    n_trials = Parameter('n_trials',
                         help='Number of trials for Optuna hyperparameter optimization.',
                         default=20)
    mlflow_experiment_name = Parameter('experiment_name',
                                       help='Name for the MLflow experiment.',
                                       default="AUS_Weather_Prediction_Experiment") 
    registered_model_name = Parameter('model_name',
                                      help='Name for the registered model in MLflow Model Registry.',
                                      default="WeatherClassifierAUS")
    TARGET_COLUMN = "RainTomorrow"
    DATASET_SOURCE = "Kaggle Weather Dataset (Australia)"
    DATASET_VERSION = "1.0"
    RANDOM_STATE = 42
    DATASET_PATH = './dataset/weatherAUS.csv'
    def hpo_objective(self, trial, X_train_processed, y_train_hpo, X_val_processed, y_val_hpo):
        with mlflow.start_run(nested=True) as nested_run:
            logreg_C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
            max_iter = trial.suggest_int("max_iter", 100, 1500)
            penalty= trial.suggest_categorical("penalty", ["l1", "l2"])
            print(f"Trial {trial.number}: C={logreg_C:.4f}, solver={solver}, max_iter={max_iter}, penalty={penalty}")
            mlflow.set_tag("OptunaTrial", trial.number)
            mlflow.set_tag("OptunaTrial_C", logreg_C)
            mlflow.set_tag("OptunaTrial_solver", solver)
            mlflow.set_tag("OptunaTrial_max_iter", max_iter)
            mlflow.set_tag("OptunaTrial_penalty", penalty)
            mlflow.set_tag("mlflow.runName", f"OptunaTrial_{trial.number}_{solver}_C={logreg_C:.4f}") # More descriptive run name
            mlflow.set_tag("Status", "In Progress")
            mlflow.log_params({
                "C": logreg_C,
                "solver": solver,
                "max_iter": max_iter,
                "penalty": penalty,
                "optuna_trial_number": trial.number
            })

            model = LogisticRegression(
                C=logreg_C,
                solver=solver,
                max_iter=max_iter,
                penalty=penalty,
                random_state=self.RANDOM_STATE,
                n_jobs=-1
            )

            try:
                model.fit(X_train_processed, y_train_hpo)
            except Exception as e:
                print(f"Error during model fitting in HPO trial {trial.number}: {e}")
                mlflow.set_tag("Status", "Failed")

                return 0.0
            y_pred_val = model.predict(X_val_processed)
            y_proba_val = model.predict_proba(X_val_processed)[:, 1]
            accuracy = accuracy_score(y_val_hpo, y_pred_val)
            try:
                 logloss = log_loss(y_val_hpo, y_proba_val)
            except ValueError as e:

                print(f"Warning: Could not calculate log loss for trial {trial.number}: {e}")
                logloss = np.nan 

            mlflow.log_metric("validation_accuracy", accuracy)
            if not np.isnan(logloss):
                mlflow.log_metric("validation_logloss", logloss)
            mlflow.set_tag("mlflow.runName", f"OptunaTrial_{trial.number}_{solver}_C={logreg_C:.4f}")
            
            mlflow.set_tag("Status", "Completed")
        return accuracy


    @step
    def start(self):
        print("--- Pipeline Start ---")
        print(f"Metaflow Run ID: {current.run_id}")
        print(f"Executing Flow: {current.flow_name}")
        print(f"Parameters:")
        print(f"  Dataset Path: {self.DATASET_PATH}")
        print(f"  Test Split Ratio: {self.test_split_ratio}")
        print(f"  Validation Split Ratio: {self.validation_split_raio}")
        print(f"  Optuna Trials: {self.n_trials}")
        print(f"  MLflow Experiment: {self.mlflow_experiment_name}")
        print(f"  MLflow Registered Model: {self.registered_model_name}")

       
        mlflow.set_experiment(self.mlflow_experiment_name)
       

        run_name = f"MetaflowRun_{current.run_id}_{current.flow_name}"
        tags = {
            "metaflow_run_id": current.run_id,
            "metaflow_flow_name": current.flow_name,
            "pipeline_tool": "Metaflow",
            "ml_task": "Binary Classification",
            "project": "Weather Prediction",
            "dataset_path_param": self.DATASET_PATH, 
            "dataset_source": self.DATASET_SOURCE,
            "dataset_version": self.DATASET_VERSION,
            "target_variable": self.TARGET_COLUMN,
            "status": "Started", 
        }
        active_run = mlflow.start_run(run_name=run_name, tags=tags)
        self.main_mlflow_run_id = active_run.info.run_id
        print(f"Started main MLflow run. Run ID: {self.main_mlflow_run_id}")
        print(f"MLflow UI Tip: Search for run ID '{self.main_mlflow_run_id}' or tag 'metaflow_run_id={current.run_id}'")

        mlflow.log_params({
            "param_n_trials": self.n_trials,
            "param_test_split_ratio": self.test_split_ratio,
            "param_val_split_ratio": self.validation_split_raio,
        })

        self.next(self.load_raw_data)


    @step
    def load_raw_data(self):
        print("\n--- Step: Load Raw Data ---")
        self.raw_df = load_data_helper(self.DATASET_PATH)
        with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False):
             mlflow.log_metric("raw_data_rows", self.raw_df.shape[0])
             mlflow.log_metric("raw_data_cols_after_initial_drop", self.raw_df.shape[1])
             mlflow.set_tag("step_load_data", "Success")
        self.next(self.split_data)


    @step
    def split_data(self):
       
        print("\n--- Step: Split Data ---")
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data_helper(
            self.raw_df,
            self.TARGET_COLUMN,
            self.test_split_ratio,
            self.validation_split_raio,
            random_state=self.RANDOM_STATE
        )
        del self.raw_df
        print("Raw DataFrame deleted from memory.")
        with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False):
            mlflow.log_params({
                "split_train_shape": str(self.X_train.shape),
                "split_val_shape": str(self.X_val.shape),
                "split_test_shape": str(self.X_test.shape),
                "split_train_samples": len(self.y_train),
                "split_val_samples": len(self.y_val),
                "split_test_samples": len(self.y_test),
                "split_random_state": self.RANDOM_STATE,
            })
            mlflow.set_tag("step_split_data", "Success")
        self.next(self.preprocess_data)
    @step
    def preprocess_data(self):
        """
        Preprocess Step: Define and fit the preprocess  ing pipeline ONLY on training data.
        Transform train, validation, and test sets. Log the preprocessor with explicit signature.
        """
        print("\n--- Step: Preprocess Data ---")
        raintoday_col = 'RainToday'
        if raintoday_col in self.X_train.columns:
            print(f"Processing '{raintoday_col}' column separately...")
            raintoday_map = {'Yes': 1.0, 'No': 0.0}
            self.X_train.loc[:, raintoday_col + '_coded'] = self.X_train[raintoday_col].map(raintoday_map)
            self.X_val.loc[:, raintoday_col + '_coded'] = self.X_val[raintoday_col].map(raintoday_map)
            self.X_test.loc[:, raintoday_col + '_coded'] = self.X_test[raintoday_col].map(raintoday_map)
            rain_today_mode = self.X_train[raintoday_col + '_coded'].mode()[0]
            print(f"  Imputing NaNs in '{raintoday_col}_coded' with mode from train set: {rain_today_mode}")
            self.X_train[raintoday_col + '_coded'].fillna(rain_today_mode, inplace=True)
            self.X_val[raintoday_col + '_coded'].fillna(rain_today_mode, inplace=True)
            self.X_test[raintoday_col + '_coded'].fillna(rain_today_mode, inplace=True)
            self.X_train = self.X_train.drop(columns=[raintoday_col])
            self.X_val = self.X_val.drop(columns=[raintoday_col])
            self.X_test = self.X_test.drop(columns=[raintoday_col])
            print(f"  Original '{raintoday_col}' column dropped.")
            self.processed_raintoday_col = raintoday_col + '_coded'
        else:
            print(f"'{raintoday_col}' column not found or already processed.")
            self.processed_raintoday_col = None

        numeric_features = self.X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.processed_raintoday_col and self.processed_raintoday_col in numeric_features:
             numeric_features.remove(self.processed_raintoday_col)
        if self.processed_raintoday_col and self.processed_raintoday_col in categorical_features:
             categorical_features.remove(self.processed_raintoday_col)

        print(f"Identified feature types for ColumnTransformer:")
        print(f"  Numeric ({len(numeric_features)}): {numeric_features}")
        print(f"  Categorical ({len(categorical_features)}): {categorical_features}")
        if self.processed_raintoday_col:
            print(f"  Manually handled: {self.processed_raintoday_col}")

        numeric_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        print("Fitting preprocessor on training data ONLY...")
        self.preprocessor.fit(self.X_train)
        print("Preprocessor fitted.")

        try:
            self.processed_feature_names = self.preprocessor.get_feature_names_out()
            print(f"Total features after preprocessing: {len(self.processed_feature_names)}")
        except Exception as e:
             print(f"Warning: Could not automatically get feature names from preprocessor: {e}")
             self.processed_feature_names = None

        print("Transforming train, validation, and test sets...")
        X_train_processed_np = self.preprocessor.transform(self.X_train)
        X_val_processed_np = self.preprocessor.transform(self.X_val)
        X_test_processed_np = self.preprocessor.transform(self.X_test)
        print("Data transformation complete (results are NumPy arrays).")

        if self.processed_feature_names is not None:
            self.X_train_processed = pd.DataFrame(X_train_processed_np, columns=self.processed_feature_names, index=self.X_train.index)
            self.X_val_processed = pd.DataFrame(X_val_processed_np, columns=self.processed_feature_names, index=self.X_val.index)
            self.X_test_processed = pd.DataFrame(X_test_processed_np, columns=self.processed_feature_names, index=self.X_test.index)
            print("Transformed data converted back to DataFrames using extracted feature names.")
        else:
            self.X_train_processed = X_train_processed_np
            self.X_val_processed = X_val_processed_np
            self.X_test_processed = X_test_processed_np
            print("Warning: Storing processed data as NumPy arrays due to feature name extraction issue.")
        with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False):
            mlflow.log_param("preprocessing_numeric_features_count", len(numeric_features))
            mlflow.log_param("preprocessing_categorical_features_count", len(categorical_features))
            mlflow.log_param("preprocessing_numeric_strategy", "mean_impute_then_StandardScale")
            mlflow.log_param("preprocessing_categorical_strategy", "mode_impute_then_OneHotEncode")
            mlflow.log_param("preprocessing_unknown_category_handling", "ignore")
            mlflow.log_param("preprocessing_remainder", "passthrough")
            mlflow.log_param("processed_features_count", X_train_processed_np.shape[1])
            mlflow.set_tag("step_preprocess_data", "Success")

        #     # --- Manually create signature ---
        #     try:
        #         print("Attempting to create and log preprocessor signature...")
        #         # Input example: Raw data the preprocessor expects
        #         input_example = self.X_train.iloc[:5]
        #         # Output example: Result of transforming the input example
        #         output_example_np = self.preprocessor.transform(input_example)

        #         # Create DataFrame for output signature if possible
        #         if self.processed_feature_names is not None:
        #             output_example = pd.DataFrame(output_example_np, columns=self.processed_feature_names, index=input_example.index)
        #         else:
        #             # Fallback: Use NumPy array if names aren't available
        #             output_example = output_example_np
        #             print("Warning: Using NumPy array for output signature part as feature names weren't extracted.")

        #         signature = infer_signature(input_example, output_example)
        #         print("Signature inferred successfully.")

        #         # Log the preprocessor with the explicit signature
        #         mlflow.sklearn.log_model(
        #             sk_model=self.preprocessor,
        #             artifact_path="preprocessor",
        #             signature=signature, # Pass the created signature
        #             input_example=input_example # Still useful to provide the input example too
        #         )
        #         print("Preprocessor logged successfully with explicit signature.")

        #     except Exception as log_ex:
        #         print(f"Error creating signature or logging preprocessor: {log_ex}. Logging without signature/example.")
        #         # Fallback: Log without signature/example if the above fails
        #         mlflow.sklearn.log_model(self.preprocessor, "preprocessor")

        # self.preprocessor_uri = f"runs:/{self.main_mlflow_run_id}/preprocessor"
        # print(f"Preprocessor artifact URI: {self.preprocessor_uri}")

        if not isinstance(self.X_train_processed, pd.DataFrame):
             del self.X_train, self.X_val, self.X_test
             print("Original X dataframes deleted after converting processed data to NumPy.")

        self.next(self.tune_hyperparameters)


    @step
    def tune_hyperparameters(self):
        print(f"\n--- Step: Tune Hyperparameters ({self.n_trials} trials) ---")
        with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False) as parent_run:
            mlflow.set_tag("step_hpo", "In Progress")
            mlflow.log_param("hpo_tool", "Optuna")
            mlflow.log_param("hpo_num_trials", self.n_trials)
            mlflow.log_param("hpo_metric", "validation_accuracy")
            mlflow.log_param("hpo_direction", "maximize")
            study = optuna.create_study(
                direction="maximize",
                study_name=f"Weather_HPO_{current.run_id}",
            )
            study.optimize(
                lambda trial: self.hpo_objective(
                    trial,
                    self.X_train_processed,
                    self.y_train,
                    self.X_val_processed,
                    self.y_val
                ),
                n_trials=self.n_trials,
                n_jobs=1,
                show_progress_bar=True
            )
            self.best_hpo_params = study.best_params
            self.best_hpo_value = study.best_value
            print("\nHyperparameter Optimization Finished.")
            print(f"  Best Trial Number: {study.best_trial.number}")
            print(f"  Best Validation Accuracy (from HPO): {self.best_hpo_value:.4f}")
            print(f"  Best Hyperparameters found: {self.best_hpo_params}")
            mlflow.log_params({f"best_hpo_{k}": v for k, v in self.best_hpo_params.items()})
            mlflow.log_metric("best_hpo_validation_accuracy", self.best_hpo_value)
            mlflow.set_tag("step_hpo", "Success")
            try:
                fig_history = optuna.visualization.plot_optimization_history(study)
                history_path = "optuna_optimization_history.html"
                fig_history.write_html(history_path)
                mlflow.log_artifact(history_path)
                os.remove(history_path)
                fig_slice = optuna.visualization.plot_slice(study)
                slice_path = "optuna_slice_plot.html"
                fig_slice.write_html(slice_path)
                mlflow.log_artifact(slice_path)
                os.remove(slice_path)
                fig_param_importance = optuna.visualization.plot_param_importances(study)
                importance_path = "optuna_param_importance.html"
                fig_param_importance.write_html(importance_path)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                print("Optuna visualization plots logged as HTML artifacts to MLflow.")
            except ImportError:
                print("Could not log Optuna plots: `plotly` or `kaleido` might not be installed.")
            except Exception as e:
                print(f"Warning: Could not log Optuna visualization plots: {e}")
        self.next(self.train_final_model)

    
    @step
    def train_final_model(self):
        print("\n--- Step: Train Final Model ---")
        with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False):
            mlflow.set_tag("step_train_final", "In Progress")
            print(f"Using best hyperparameters from HPO: {self.best_hpo_params}")
            mlflow.log_params({f"final_model_{k}": v for k, v in self.best_hpo_params.items()})
            mlflow.log_param("final_model_type", "LogisticRegression")
            valid_lr_params = {k: v for k, v in self.best_hpo_params.items()
                               if k in ['C', 'solver', 'max_iter', 'penalty']}
            n_cv_folds = 5
            print(f"Performing {n_cv_folds}-fold cross-validation on the training set...")
            cv_model = LogisticRegression(
                **valid_lr_params,
                random_state=self.RANDOM_STATE,
                n_jobs=-1
            )
            cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.RANDOM_STATE)
            cv_scores = cross_val_score(cv_model, self.X_train_processed, self.y_train,
                                        cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"  Cross-validation accuracy scores: {np.round(cv_scores, 4)}")
            print(f"  Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            mlflow.log_metrics({
                "cv_mean_accuracy": cv_scores.mean(),
                "cv_std_accuracy": cv_scores.std(),
                "cv_min_accuracy": cv_scores.min(),
                "cv_max_accuracy": cv_scores.max(),
            })
            for i, score in enumerate(cv_scores):
                mlflow.log_metric(f"cv_fold_{i+1}_accuracy", score)

            print("Training final model on the full preprocessed training set...")
            self.final_model = LogisticRegression(
                 **valid_lr_params,
                random_state=self.RANDOM_STATE,
                n_jobs=-1
            )
            self.final_model.fit(self.X_train_processed, self.y_train)
            print("Final model trained.")

            y_pred_train = self.final_model.predict(self.X_train_processed)
            y_proba_train = self.final_model.predict_proba(self.X_train_processed)[:,1]
            y_pred_val = self.final_model.predict(self.X_val_processed)
            y_proba_val = self.final_model.predict_proba(self.X_val_processed)[:,1]
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            train_logloss = log_loss(self.y_train, y_proba_train)
            val_accuracy = accuracy_score(self.y_val, y_pred_val)
            val_logloss = log_loss(self.y_val, y_proba_val)
            print(f"  Performance on Full Sets (using final model):")
            print(f"    Training Accuracy:   {train_accuracy:.4f}")
            print(f"    Training LogLoss:    {train_logloss:.4f}")
            print(f"    Validation Accuracy: {val_accuracy:.4f}")
            print(f"    Validation LogLoss:  {val_logloss:.4f}")
            mlflow.log_metrics({
                "final_train_accuracy": train_accuracy,
                "final_train_logloss": train_logloss,
                "final_validation_accuracy": val_accuracy,
                "final_validation_logloss": val_logloss
            })

            print(f"Logging final model artifact 'final-model' and registering as '{self.registered_model_name}'...")
            try:
        
                if isinstance(self.X_train_processed, pd.DataFrame):
                    input_example_model = self.X_train_processed.iloc[:5]
                else: # Handle NumPy array case
                     num_samples = min(5, self.X_train_processed.shape[0])
                     input_example_model = self.X_train_processed[:num_samples, :]
                     if input_example_model.ndim == 1:
                        input_example_model = input_example_model.reshape(num_samples, -1)
                mlflow.sklearn.log_model(
                    sk_model=self.final_model,
                    artifact_path="final-model",
                    input_example=input_example_model,
                    registered_model_name=self.registered_model_name
                )
                print("Final model logged to 'final-model' artifact and registered.")
            except Exception as log_ex:
                print(f"Error logging/registering final model: {log_ex}. Logging without example/registration.")
                mlflow.sklearn.log_model(
                    sk_model=self.final_model,
                    artifact_path="final-model"
                )

            self.model_uri = f"runs:/{self.main_mlflow_run_id}/final-model"
            print(f"Final model artifact URI: {self.model_uri}")
            try:
                plt.figure(figsize=(8, 5))
                fold_indices = range(1, len(cv_scores) + 1)
                plt.bar(fold_indices, cv_scores, alpha=0.7, label='Fold Accuracy')
                plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean Acc: {cv_scores.mean():.4f}')
                plt.xlabel('CV Fold'); plt.ylabel('Accuracy'); plt.title(f'{n_cv_folds}-Fold CV Accuracy (Train Set)')
                plt.xticks(fold_indices); plt.ylim(bottom=max(0, cv_scores.min() - 0.05), top=1.0)
                plt.legend(); plt.tight_layout()
                cv_plot_path = "cross_validation_results.png"
                plt.savefig(cv_plot_path); mlflow.log_artifact(cv_plot_path); plt.close(); os.remove(cv_plot_path)
                print("Cross-validation plot logged to MLflow.")

                # Validation CM plot
                cm_val = confusion_matrix(self.y_val, y_pred_val)
                disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["No Rain", "Rain"])
                fig_val, ax_val = plt.subplots(figsize=(7, 6))
                disp_val.plot(ax=ax_val, cmap=plt.cm.Blues, values_format='d')
                plt.title('Validation Set Confusion Matrix (Final Model)'); plt.tight_layout()
                val_cm_path = "validation_confusion_matrix.png"
                plt.savefig(val_cm_path); mlflow.log_artifact(val_cm_path); plt.close(fig_val); os.remove(val_cm_path)
                print("Validation confusion matrix logged to MLflow.")

            except Exception as e:
                print(f"Warning: Could not log visualization artifacts during training: {e}")

            mlflow.set_tag("step_train_final", "Success")

        self.next(self.evaluate_on_test_set)


    @card(type='default')
    @step
    def evaluate_on_test_set(self):
        print("\n--- Step: Evaluate on Test Set ---")
        card_content = [Markdown("# Final Model Evaluation on Test Set")]
        test_cm_path = "test_confusion_matrix_card.png"
        try:
            with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False):
                mlflow.set_tag("step_evaluate_test", "In Progress")
                print(f"Loading final model from URI: {self.model_uri}")
                loaded_model = mlflow.sklearn.load_model(self.model_uri)
                print(f"Loading preprocessor")
                loaded_preprocessor = self.preprocessor
                print("Model and preprocessor loaded successfully.")

                print("Transforming test set using loaded preprocessor...")
                if not hasattr(self, 'X_test'):
                      raise RuntimeError("X_test is not available in the evaluation step. Ensure it's passed or re-loaded.")

                X_test_processed_eval = loaded_preprocessor.transform(self.X_test)
                print("Test set transformed for evaluation.")

                print("Predicting on processed test set...")
                y_pred_test = loaded_model.predict(X_test_processed_eval)
                y_proba_test = loaded_model.predict_proba(X_test_processed_eval)[:, 1]
                self.test_accuracy = accuracy_score(self.y_test, y_pred_test)
                self.test_logloss = log_loss(self.y_test, y_proba_test)
                self.test_confusion_matrix = confusion_matrix(self.y_test, y_pred_test)
                print(f"\n--- Test Set Performance ---")
                print(f"  Test Accuracy: {self.test_accuracy:.4f}")
                print(f"  Test LogLoss:  {self.test_logloss:.4f}")
                mlflow.log_metrics({
                    "test_set_accuracy": self.test_accuracy,
                    "test_set_logloss": self.test_logloss
                })
                tn, fp, fn, tp = self.test_confusion_matrix.ravel()
                mlflow.log_metrics({
                    "test_set_true_negatives": tn, "test_set_false_positives": fp,
                    "test_set_false_negatives": fn, "test_set_true_positives": tp
                })
                mlflow.set_tag("step_evaluate_test", "Success")
                mlflow.set_tag("final_status", "Completed")
                card_content.append(Markdown(f"**Metaflow Run ID:** `{current.run_id}`"))
                card_content.append(Markdown(f"**MLflow Run ID:** `{self.main_mlflow_run_id}`"))
                card_content.append(Markdown(f"**Registered Model:** `{self.registered_model_name}` (if registration succeeded)"))
                card_content.append(Markdown(f"**Model URI:** `{self.model_uri}`"))
                # card_content.append(Markdown(f"**Preprocessor URI:** `{self.preprocessor_uri}`"))
                card_content.append(Markdown("### Test Set Metrics"))
                metrics_data = [ ["Accuracy", f"{self.test_accuracy:.4f}"], ["LogLoss", f"{self.test_logloss:.4f}"] ]
                card_content.append(Table(metrics_data, headers=["Metric", "Value"]))
                card_content.append(Markdown("### Test Set Confusion Matrix"))
                try:
                    disp_test = ConfusionMatrixDisplay(confusion_matrix=self.test_confusion_matrix, display_labels=["No Rain", "Rain"])
                    fig, ax = plt.subplots(figsize=(7, 6))
                    disp_test.plot(ax=ax, cmap=plt.cm.Greens, values_format='d')
                    plt.title('Test Set Confusion Matrix'); plt.tight_layout()
                    plt.savefig(test_cm_path)
                    mlflow.log_artifact(test_cm_path, artifact_path="evaluation_plots")
                    card_content.append(Image.from_matplotlib(fig))
                    plt.close(fig)
                    if os.path.exists(test_cm_path): os.remove(test_cm_path)
                except Exception as plot_err:
                    print(f"Error generating/saving confusion matrix plot: {plot_err}")
                    card_content.append(Markdown(f"<font color='orange'>Could not generate confusion matrix plot: {plot_err}</font>"))

        except Exception as e:
            print(f"\n--- ERROR during test set evaluation or Card generation! ---")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            card_content.append(Markdown(f"### <font color='red'>Error During Evaluation</font>\n"
                                          f"Evaluation step failed. Check logs for details.\n"
                                          f"**Error:**\n```\n{e}\n```"))
            try:
                with mlflow.start_run(run_id=self.main_mlflow_run_id, nested=False):
                    mlflow.set_tag("step_evaluate_test", "Failed")
                    mlflow.set_tag("final_status", "Failed")
            except Exception as mlflow_err:
                print(f"Could not set failure tags in MLflow after evaluation error: {mlflow_err}")

        current.card.extend(card_content)
        print("Metaflow card content generated.")
        self.next(self.end)


    @step
    def end(self):
        print("\n--- Pipeline End ---")
        print(f"Metaflow Run ID: {current.run_id}")
        print(f"Main MLflow Run ID: {self.main_mlflow_run_id}")
        try:
             client = mlflow.tracking.MlflowClient()
             run_data = client.get_run(self.main_mlflow_run_id).data
             final_status = run_data.tags.get("final_status", "Unknown")
             if final_status == "Completed":
                 self.test_accuracy = run_data.metrics.get("test_set_accuracy", float('nan'))
                 self.test_logloss = run_data.metrics.get("test_set_logloss", float('nan'))
        except Exception as e:
            print(f"Warning: Could not fetch final status/metrics from MLflow: {e}")
            final_status = "Unknown (MLflow fetch failed)"

        print(f"Pipeline Final Status: {final_status}")

        if final_status == "Completed":
            print(f"\nFinal Test Set Metrics (from MLflow run):")
            print(f"  Accuracy: {self.test_accuracy:.4f}")
            print(f"  LogLoss:  {self.test_logloss:.4f}")
        elif final_status == "Failed":
             print("\nPipeline finished with errors. Please check logs and MLflow for details.")

        print("\n--- Next Steps ---")
        print(f"1. View Metaflow Card: python {__file__} card view {current.run_id}")
        print(f"2. Explore MLflow Run: Check the MLflow UI for experiment '{self.mlflow_experiment_name}' and run ID '{self.main_mlflow_run_id}'")
        print(f"3. Access Registered Model: Look for '{self.registered_model_name}' in the MLflow Model Registry.")
        

if __name__ == '__main__':
    MetaflowTrainingPipeline()