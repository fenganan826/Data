import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import shap
import matplotlib.pyplot as plt

# 初始化SHAP
shap.initjs()

# 读取数据
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

def objective(n_neighbors, weights, p):
    n_neighbors = int(n_neighbors)
    weights = 'uniform' if weights < 0.5 else 'distance'
    p = int(p)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return -mse

pbounds = {
    'n_neighbors': (1, 50),
    'weights': (0, 1),
    'p': (1, 2)
}

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=10)

best_params = optimizer.max['params']
best_n_neighbors = int(best_params['n_neighbors'])
best_weights = 'uniform' if best_params['weights'] < 0.5 else 'distance'
best_p = int(best_params['p'])

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=best_n_neighbors,
                                      weights=best_weights,
                                      p=best_p))
])

# ✅ 使用训练集训练模型
final_pipeline.fit(X_train, y_train)

# ✅ 验证集预测与评估
y_pred_val = final_pipeline.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print("验证集评估结果：")
print(f'RMSE: {rmse_val}')
print(f'MAE: {mae_val}')
print(f'R²: {r2_val}')
print(f'最佳参数: n_neighbors={best_n_neighbors}, weights={best_weights}, p={best_p}')

# ✅ 测试集预测并导出结果
y_pred_test = final_pipeline.predict(X_test)
results_df = pd.DataFrame({
    'True Activity': y_test.values,
    'Predicted Activity': y_pred_test
})
results_df.to_csv(r'F:\knn_test_predictions.csv', index=False)
print("测试集预测结果已保存至 F:\\knn_test_predictions.csv")

# SHAP 分析
X_val_processed = final_pipeline.named_steps['preprocessor'].transform(X_val)
if hasattr(X_val_processed, 'toarray'):
    X_val_processed = X_val_processed.toarray()

model = final_pipeline.named_steps['regressor']

# SHAP 采样加速
X_sample = shap.sample(X_val, 100, random_state=42)
X_sample_processed = final_pipeline.named_steps['preprocessor'].transform(X_sample)
if hasattr(X_sample_processed, 'toarray'):
    X_sample_processed = X_sample_processed.toarray()

explainer = shap.KernelExplainer(model.predict, X_sample_processed)
shap_values = explainer.shap_values(X_sample_processed)

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
        for j in range(n_samples):
            if len(indices) > 1:
                feature_shap = [shap_values[j][idx] for idx in indices]
                max_idx = np.argmax(np.abs(feature_shap))
                aggregated_shap[j, i] = feature_shap[max_idx]
            else:
                aggregated_shap[j, i] = shap_values[j][indices[0]]

    return aggregated_shap, original_feature_names

aggregated_shap, new_feature_names = aggregate_shap_for_categorical_features(
    shap_values, X_sample_processed, final_pipeline.named_steps['preprocessor'], X.columns.tolist()
)

plt.figure(figsize=(12, 8))
shap.summary_plot(aggregated_shap, X_sample, feature_names=new_feature_names, show=False)
plt.tight_layout()
plt.show()
