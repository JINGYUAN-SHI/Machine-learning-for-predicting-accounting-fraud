# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
    confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, precision_recall_curve, \
    average_precision_score, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, ndcg_score
from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.font_manager as fm
import math

warnings.filterwarnings('ignore')

# Set font to support Chinese/Unicode
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# Define evaluation function - add NDCG@k, where k = 1% of test set size
def calculate_metrics_two(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba) if y_proba is not None else 0

    # Calculate NDCG@k, where k = 1% of test set size
    k = max(1, math.ceil(len(y_true) * 0.01))  # at least 1
    if len(y_true) >= k:
        # Convert y_proba to 2D array because ndcg_score requires 2D input
        y_proba_2d = np.vstack([1 - y_proba, y_proba]).T
        y_true_2d = np.vstack([1 - y_true, y_true]).T
        ndcg = ndcg_score(y_true_2d, y_proba_2d, k=k)
    else:
        ndcg = np.nan

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_val,
        'f1': f1,
        'kappa': kappa,
        'auc': auc_score,
        'ndcg': ndcg,  # fixed column name
        'ndcg_k': k    # store k value
    }


# Read data
df = pd.read_excel(rC:\Users\Shi\PycharmProjects\RUSBoost\data2.xlsx")

# Ensure target variable is integer type
target_column = 'y'
df[target_column] = df[target_column].astype(int)

# Ensure fyear column exists
if 'fyear' not in df.columns:
    raise ValueError("Data must contain column 'fyear' for time split")

# Print basic dataset info
print(f"Data shape: {df.shape}")
print(f"Unique values of target '{target_column}': {df[target_column].unique()}")
print(f"Target variable '{target_column}' value counts:\n{df[target_column].value_counts()}")

# Check and handle missing values
print(f"Total missing values in dataset: {df.isna().sum().sum()}")
if df.isna().sum().sum() > 0:
    print("Missing values per column:")
    print(df.isna().sum())

    # Drop missing values in target
    df = df.dropna(subset=[target_column])
    print(f"Data shape after dropping missing targets: {df.shape}")

# Define feature sets - using raw financial and other features
feature_sets = {
    'Financial_Raw_And_Other_Features': [  # 28(X) + 3(M) + 8(A) + 3(B)
        'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
        'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
        'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28',
        'm1', 'm2', 'm3',
        'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8',
        'b1', 'b2', 'b3'
    ]
}

# Define test years
test_years = [2011, 2012, 2013, 2014]

# Store results for all test years
all_results = []
all_confusion_matrices = []
all_roc_data = []
all_pr_data = []  
all_feature_importances = []

# Analysis for each test year
for test_year in test_years:
    print(f"\n\n{'=' * 80}")
    print(f"Start analyzing test year: {test_year}")
    print(f"{'=' * 80}")

    # Training start and end years
    train_end_year = test_year - 2
    train_start_year = 2000

    # Ensure training period >= 10 years
    if train_end_year - train_start_year + 1 < 10:
        print(f"Training period < 10 years, skip test year {test_year}")
        continue

    # Loop through feature sets
    for set_name, features in feature_sets.items():
        print(f"\nAnalyzing feature set: {set_name}")

        # Check feature existence
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: missing features: {missing_features}")
            # Keep only existing features
            features = [f for f in features if f in df.columns]
            print(f"Using features: {features}")

        # Split train/test
        train_df = df[(df['fyear'] >= train_start_year) & (df['fyear'] <= train_end_year)]
        test_df = df[df['fyear'] == test_year]

        print(f"Train period: {train_start_year}-{train_end_year}, Test year: {test_year}")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        # Fill missing values
        for feature in features:
            if train_df[feature].isna().sum() > 0:
                # categorical feature
                if feature in ['b1']:  
                    mode_value = train_df[feature].mode()[0] if not train_df[feature].mode().empty else 0
                    train_df[feature] = train_df[feature].fillna(mode_value)
                    test_df[feature] = test_df[feature].fillna(mode_value)
                # numerical features
                else:
                    mean_value = train_df[feature].mean()
                    train_df[feature] = train_df[feature].fillna(mean_value)
                    test_df[feature] = test_df[feature].fillna(mean_value)

        # Extract X and y
        X_train = train_df[features].values
        y_train = train_df[target_column].values
        X_test = test_df[features].values
        y_test = test_df[target_column].values

        # Ensure no NaN in y
        if np.isnan(y_train).any() or np.isnan(y_test).any():
            print("Warning: NaN in target")
            continue

        # Ensure integer type
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Print class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
        print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

        # Standardize numerical features (exclude categorical b1)
        numeric_feature_indices = [i for i, f in enumerate(features) if f not in ['b1']]

        if numeric_feature_indices:
            scaler = StandardScaler()
            X_train_numeric = scaler.fit_transform(X_train[:, numeric_feature_indices])
            X_test_numeric = scaler.transform(X_test[:, numeric_feature_indices])

            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()

            for i, idx in enumerate(numeric_feature_indices):
                X_train_processed[:, idx] = X_train_numeric[:, i]
                X_test_processed[:, idx] = X_test_numeric[:, i]

            X_train = X_train_processed
            X_test = X_test_processed

        try:
            # RUSBoost with 1:1 sampling
            rusboost = RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2),
                n_estimators=100,
                learning_rate=0.5,
                sampling_strategy=1.0,  
                random_state=42
            )

            rusboost.fit(X_train, y_train)
            print("Model training success!")

            # Prediction
            y_pred_test = rusboost.predict(X_test)
            y_pred_proba_test = rusboost.predict_proba(X_test)[:, 1]

            # Metrics
            test_metrics = calculate_metrics_two(y_test, y_pred_test, y_pred_proba_test)

            # Save results
            result = {
                'year': test_year,
                'set_name': set_name,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_class_distribution': dict(zip(unique_train, counts_train)),
                'test_class_distribution': dict(zip(unique_test, counts_test))
            }
            result.update(test_metrics)
            all_results.append(result)

            print(f"\n{set_name} {test_year} test performance:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            all_confusion_matrices.append(cm)

            # ROC data
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
            roc_auc = auc(fpr, tpr)
            all_roc_data.append((fpr, tpr, roc_auc))

            # PR data
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
            pr_auc = average_precision_score(y_test, y_pred_proba_test)
            all_pr_data.append((precision, recall, pr_auc))

            # Feature importance
            print("Calculating feature importance...")
            try:
                if hasattr(rusboost, 'feature_importances_'):
                    feature_importance = rusboost.feature_importances_

                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': feature_importance,
                        'year': test_year
                    })

                    all_feature_importances.append(importance_df)
                    print("Feature importance calculated")

                    print(f"\n{test_year} feature importance:")
                    sorted_importance = importance_df.sort_values('importance', ascending=False)
                    print(sorted_importance)
                else:
                    print("Model has no feature_importances_ attribute")
            except Exception as e:
                print(f"Feature importance calculation failed: {str(e)}")

        except Exception as e:
            print(f"Model training failed, error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        print(f"\nFinished analysis for test year {test_year}, feature set {set_name}")

    print(f"\nCompleted analysis for test year {test_year}")
    print(f"{'=' * 80}")

# Aggregate metrics
if all_results:
    results_df = pd.DataFrame(all_results)

    avg_metrics = results_df.groupby('set_name').agg({
        'accuracy': ['mean', 'std'],
        'sensitivity': ['mean', 'std'],
        'specificity': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'kappa': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'ndcg': ['mean', 'std']
    }).round(4)

    print("\n\nAverage performance across all test years:")
    print(avg_metrics)

    # Save to Excel
    results_df.to_excel('all_results.xlsx', index=False)
    avg_metrics.to_excel('average_metrics.xlsx')
    print("Results saved to all_results.xlsx and average_metrics.xlsx")

    # Average confusion matrix
    if all_confusion_matrices:
        avg_cm = np.mean(all_confusion_matrices, axis=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(avg_cm, display_labels=['Non-Fraud', 'Fraud']).plot(
            cmap='Blues', ax=ax, values_format='.2f'
        )
        ax.set_title('Average Confusion Matrix (4 Test Years)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Average ROC
    if all_roc_data:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        for fpr, tpr, roc_auc in all_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Average ROC Curve (4 Test Years)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Average PR
    if all_pr_data:
        mean_recall = np.linspace(0, 1, 100)
        precisions = []
        aucs = []

        for precision, recall, pr_auc in all_pr_data:
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_precision)
            aucs.append(pr_auc)

        mean_precision = np.mean(precisions, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_recall, mean_precision, color='b',
                label=r'Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        positive_ratio = np.mean([result['test_class_distribution'].get(1, 0) /
                                  result['test_size'] for result in all_results])
        ax.plot([0, 1], [positive_ratio, positive_ratio], linestyle='--', lw=2, color='r',
                label=f'Random (AUC = {positive_ratio:.2f})', alpha=.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Average Precision-Recall Curve (4 Test Years)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Feature importance
    if all_feature_importances:
        all_importance_df = pd.concat(all_feature_importances, ignore_index=True)

        avg_importance = all_importance_df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        avg_importance = avg_importance.sort_values('mean', ascending=False)

        print("\nAverage feature importance:")
        print(avg_importance)

        plt.figure(figsize=(12, 10))
        y_pos = np.arange(len(avg_importance))
        plt.barh(y_pos, avg_importance['mean'], xerr=avg_importance['std'],
                 align='center', alpha=0.7, capsize=5)
        plt.yticks(y_pos, avg_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Average Feature Importance (4 Test Years)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        avg_importance.to_excel('feature_importance_results.xlsx', index=False)
        print("Feature importance results saved to feature_importance_results.xlsx")

print("\nAll test year analyses completed!")
