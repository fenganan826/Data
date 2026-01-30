import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import shap
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
shap.initjs()

# 读取数据
file_path = r'F:\Activity_standardized.csv'
df = pd.read_csv(file_path)

# 特征与目标
X = df.drop('Activity(KgPP/mol cat)', axis=1)
y = df['Activity(KgPP/mol cat)']

# 特征分类
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 数据划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

# 预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# 贝叶斯优化目标函数
def objective(n_estimators, max_samples, max_features, bootstrap, bootstrap_features):
    n_estimators = int(n_estimators)
    max_samples = float(max_samples)
    max_features = float(max_features)
    bootstrap = bool(round(bootstrap))
    bootstrap_features = bool(round(bootstrap_features))

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', BaggingRegressor(
            random_state=42,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return -mse

# 搜索空间
pbounds = {
    'n_estimators': (45, 200),
    'max_samples': (0.1, 1.0),
    'max_features': (0.1, 1.0),
    'bootstrap': (0, 1),
    'bootstrap_features': (0, 1)
}

# 贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42
)

# 启动优化
optimizer.maximize(
    init_points=5,
    n_iter=20,
)

# 提取最优参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_samples = best_params['max_samples']
best_max_features = best_params['max_features']
best_bootstrap = bool(round(best_params['bootstrap']))
best_bootstrap_features = bool(round(best_params['bootstrap_features']))

# 构建最终模型（使用最优参数）
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', BaggingRegressor(
        random_state=42,
        n_estimators=best_n_estimators,
        max_samples=best_max_samples,
        max_features=best_max_features,
        bootstrap=best_bootstrap,
        bootstrap_features=best_bootstrap_features))
])

# ✅ 只在训练集上训练
final_pipeline.fit(X_train, y_train)

# ✅ 在验证集上评估模型（此时验证集没有参与训练）
y_pred_val = final_pipeline.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print("\n验证集评估结果：")
print(f'RMSE: {rmse_val}')
print(f'MAE: {mae_val}')
print(f'R²: {r2_val}')


print(f'\n最优参数: n_estimators={best_n_estimators}, max_samples={best_max_samples}, '
      f'max_features={best_max_features}, bootstrap={best_bootstrap}, '
      f'bootstrap_features={best_bootstrap_features}')

# SHAP分析
X_test_processed = final_pipeline.named_steps['preprocessor'].transform(X_test)
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

model = final_pipeline.named_steps['regressor']

X_sample = shap.sample(X_test, 100)
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
        if len(indices) > 1:
            for j in range(n_samples):
                feature_shap_values = [shap_values[j][idx] for idx in indices]
                max_idx = np.argmax(np.abs(feature_shap_values))
                aggregated_shap[j, i] = feature_shap_values[max_idx]
        else:
            for j in range(n_samples):
                aggregated_shap[j, i] = shap_values[j][indices[0]]

    return aggregated_shap, original_feature_names

aggregated_shap, new_feature_names = aggregate_shap_for_categorical_features(
    shap_values, X_sample_processed, final_pipeline.named_steps['preprocessor'], X.columns.tolist()
)

plt.figure(figsize=(10, 8))
shap.summary_plot(aggregated_shap, X_sample,
                  feature_names=new_feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot_aggregated.png', dpi=300, bbox_inches='tight')
plt.show()

# ======== 导出测试集真实值与预测值到 F 盘 ========
y_pred_test = final_pipeline.predict(X_test)
results_df = pd.DataFrame({
    'True_Value': y_test.values,
    'Predicted_Value': y_pred_test
})
export_path = r'F:\bagging_test_predictions.csv'
results_df.to_csv(export_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 测试集真实值与预测值已保存至: {export_path}")
