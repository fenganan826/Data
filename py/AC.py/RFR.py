import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import shap
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

shap.initjs()

file_path = r'F:\Activity_standardized.csv'
df = pd.read_csv(file_path)

X = df.drop('Activity(KgPP/mol cat)', axis=1)
y = df['Activity(KgPP/mol cat)']

categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth) if max_depth is not None else None
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return -mse

pbounds = {
    'n_estimators': (50, 200),
    'max_depth': (3, 15),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=10)

best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_depth = int(best_params['max_depth']) if 'max_depth' in best_params else None
best_min_samples_split = int(best_params['min_samples_split'])
best_min_samples_leaf = int(best_params['min_samples_leaf'])

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        random_state=42))
])

# ✅ 使用训练集训练
final_pipeline.fit(X_train, y_train)

# ✅ 使用验证集进行评估
y_pred_val = final_pipeline.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print("验证集评估结果：")
print(f'RMSE: {rmse_val}')
print(f'MAE: {mae_val}')
print(f'R²: {r2_val}')
print(f'最佳参数: n_estimators={best_n_estimators}, max_depth={best_max_depth}, '
      f'min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}')

# SHAP 分析
X_val_processed = final_pipeline.named_steps['preprocessor'].transform(X_val)
if hasattr(X_val_processed, 'toarray'):
    X_val_processed = X_val_processed.toarray()

model = final_pipeline.named_steps['regressor']
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val_processed)

def aggregate_shap_for_categorical_features(shap_values, X_processed, preprocessor, original_feature_names):
    all_feature_names = []
    all_feature_names.extend(numeric_cols)
    ohe = preprocessor.named_transformers_['cat']
    categorical_output_names = ohe.get_feature_names_out(input_features=categorical_cols)
    all_feature_names.extend(categorical_output_names)

    feature_mapping = {}
    for col in original_feature_names:
        if col in categorical_cols:
            indices = [i for i, name in enumerate(all_feature_names) if name.startswith(col)]
            feature_mapping[col] = indices
        else:
            idx = all_feature_names.index(col)
            feature_mapping[col] = [idx]

    n_samples, _ = shap_values.shape
    n_original_features = len(original_feature_names)
    aggregated_shap = np.zeros((n_samples, n_original_features))

    for i, col in enumerate(original_feature_names):
        indices = feature_mapping[col]
        if len(indices) > 1:
            for j in range(n_samples):
                feature_shap = [shap_values[j][idx] for idx in indices]
                max_idx = np.argmax(np.abs(feature_shap))
                aggregated_shap[j, i] = feature_shap[max_idx]
        else:
            for j in range(n_samples):
                aggregated_shap[j, i] = shap_values[j][indices[0]]

    return aggregated_shap, original_feature_names

aggregated_shap, new_feature_names = aggregate_shap_for_categorical_features(
    shap_values, X_val_processed, final_pipeline.named_steps['preprocessor'], X.columns.tolist()
)

plt.figure(figsize=(12, 8))
shap.summary_plot(aggregated_shap, X_val, feature_names=new_feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary_aggregated.png', dpi=300, bbox_inches='tight')
plt.show()

# ✅ 导出测试集真实值与预测值
y_pred_test = final_pipeline.predict(X_test)
results_df = pd.DataFrame({
    'True Value': y_test.values,
    'Predicted Value': y_pred_test
})
results_df.to_csv(r'F:\rf_test_predictions.csv', index=False)
print("测试集预测结果已保存至 F:\\rf_test_predictions.csv")
