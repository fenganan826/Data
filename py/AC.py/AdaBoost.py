import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
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

# 确定特征和目标变量
X = df.drop('Activity(KgPP/mol cat)', axis=1)
y = df['Activity(KgPP/mol cat)']

# 区分类别型和数值型特征
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 划分训练集、验证集和测试集，比例为 7:2:1
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

# 创建预处理流水线
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# 定义目标函数，用于贝叶斯优化
def objective(n_estimators, learning_rate, max_depth):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return -mse

# 定义参数搜索范围
pbounds = {
    'n_estimators': (50, 200),
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 10)
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# 进行优化
optimizer.maximize(
    init_points=5,
    n_iter=10,
)

# 获取最佳参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_learning_rate = best_params['learning_rate']
best_max_depth = int(best_params['max_depth'])

# 使用最佳参数构建最终模型
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', AdaBoostRegressor(n_estimators=best_n_estimators,
                                     learning_rate=best_learning_rate,
                                     random_state=42))
])

# 在训练集训练模型
final_pipeline.fit(X_train, y_train)

# 预测验证集（用于评估）
y_pred_val = final_pipeline.predict(X_val)

# 预测测试集（用于后续绘图）
y_pred_test = final_pipeline.predict(X_test)

# 使用验证集进行模型评估
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

print(f'验证集均方根误差: {rmse_val}')
print(f'验证集 R² 分数: {r2_val}')
print(f'验证集平均绝对误差：{mae_val}')
print(f'最佳参数: n_estimators={best_n_estimators}, learning_rate={best_learning_rate}, max_depth={best_max_depth}')

# SHAP分析
X_test_processed = final_pipeline.named_steps['preprocessor'].transform(X_test)
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

model = final_pipeline.named_steps['regressor']

X_sample = shap.sample(X_test, 100)
X_sample_processed = preprocessor.transform(X_sample)
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
plt.show()

# ======== 导出测试集真实值与预测值到 F 盘 ========
results_df = pd.DataFrame({
    'True_Value': y_test.values,
    'Predicted_Value': y_pred_test
})
output_path = r'F:\adaboost_test_predictions.csv'
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 测试集真实值与预测值已导出至：{output_path}")
