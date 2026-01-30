import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import numpy as np
import shap
import matplotlib.pyplot as plt

# 初始化SHAP
shap.initjs()

# 读取CSV文件
file_path = r'F:\Activity_standardized.csv'
df = pd.read_csv(file_path)

# 特征和目标
X = df.drop('Activity(KgPP/mol cat)', axis=1)
y = df['Activity(KgPP/mol cat)']

# 特征类型
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# 预处理器
preprocessor = ColumnTransformer([
    ('num','passthrough', numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

# 贝叶斯优化目标函数
def objective(n_estimators, learning_rate, max_depth):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return -mse

# 超参数范围
pbounds = {
    'n_estimators': (50, 200),
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 10)
}

# 贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# 优化
optimizer.maximize(init_points=5, n_iter=10)

# 最佳参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_learning_rate = best_params['learning_rate']
best_max_depth = int(best_params['max_depth'])

# 最终模型
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
        max_depth=best_max_depth,
        random_state=42
    ))
])

# ✅ 在训练集上训练
final_pipeline.fit(X_train, y_train)

# 验证集预测
y_pred_val = final_pipeline.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

print(f'验证集均方根误差: {rmse_val}')
print(f'验证集 R² 分数: {r2_val}')
print(f'验证集平均绝对误差：{mae_val}')
print(f'最佳参数: n_estimators={best_n_estimators}, learning_rate={best_learning_rate}, max_depth={best_max_depth}')

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
        for j in range(n_samples):
            if len(indices) > 1:
                feature_shap_values = [shap_values[j][idx] for idx in indices]
                max_idx = np.argmax(np.abs(feature_shap_values))
                aggregated_shap[j, i] = feature_shap_values[max_idx]
            else:
                aggregated_shap[j, i] = shap_values[j][indices[0]]
    return aggregated_shap, original_feature_names

aggregated_shap, new_feature_names = aggregate_shap_for_categorical_features(
    shap_values, X_val_processed, final_pipeline.named_steps['preprocessor'], X.columns.tolist()
)

# SHAP可视化
plt.figure(figsize=(10, 8))
avg_shap_values = np.mean(np.abs(aggregated_shap), axis=0)
importance_df = pd.DataFrame({
    'Feature': new_feature_names,
    'Average SHAP Importance': avg_shap_values
}).sort_values('Average SHAP Importance', ascending=True)

plt.barh(importance_df['Feature'], importance_df['Average SHAP Importance'], color='#E30B5D')
plt.xlabel('mean(|SHAP value|)')
plt.ylabel('Features')
plt.title('Feature Importance Based on Average SHAP Values')
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(importance_df['Average SHAP Importance']):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.show()
