import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
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
file_path = r'F:/Activity_standardized.csv'
df = pd.read_csv(file_path)

# 特征与目标
X = df.drop('Activity(KgPP/mol cat)', axis=1)
y = df['Activity(KgPP/mol cat)']

# 特征分类
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ✅ 修改：将数据划分为 80% 训练集 + 20% 测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])


# 贝叶斯优化目标函数使用 10 折交叉验证
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

    # 使用 10 折交叉验证评估 MSE
    mse = cross_val_score(pipeline, X_trainval, y_trainval,
                          scoring='neg_mean_squared_error',
                          cv=KFold(n_splits=10, shuffle=True, random_state=42)).mean()
    return mse  # 贝叶斯优化最大化目标


# 搜索空间
pbounds = {
    'n_estimators': (32, 90),
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

print(f'\n最佳参数: n_estimators={best_n_estimators}, max_samples={best_max_samples}, '
      f'max_features={best_max_features}, bootstrap={best_bootstrap}, '
      f'bootstrap_features={best_bootstrap_features}')

# ✅ 使用整个训练集训练最终模型
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
final_pipeline.fit(X_trainval, y_trainval)

# ✅ 在测试集评估最终模型
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")

# ========== 学习曲线 ==========

train_mse = []
val_mse = []
train_r2 = []
val_r2 = []

# 设置 n_estimators_range，确保你在一个合适的范围内
n_estimators_range = np.arange(1, best_n_estimators + 10, 5)

# 使用 KFold 进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for n in n_estimators_range:
    fold_train_mse = []
    fold_val_mse = []
    fold_train_r2 = []
    fold_val_r2 = []

    for train_index, val_index in kf.split(X_trainval):
        X_train_fold, X_val_fold = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', BaggingRegressor(
                random_state=42,
                n_estimators=n,
                max_samples=best_max_samples,
                max_features=best_max_features,
                bootstrap=best_bootstrap,
                bootstrap_features=best_bootstrap_features
            ))
        ])

        model.fit(X_train_fold, y_train_fold)

        # 预测训练集和验证集
        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        # 计算每一折的 MSE 和 R²
        fold_train_mse.append(mean_squared_error(y_train_fold, y_train_pred))
        fold_val_mse.append(mean_squared_error(y_val_fold, y_val_pred))
        fold_train_r2.append(r2_score(y_train_fold, y_train_pred))
        fold_val_r2.append(r2_score(y_val_fold, y_val_pred))

    # 计算每个训练次数下的平均误差和 R²
    train_mse.append(np.mean(fold_train_mse))
    val_mse.append(np.mean(fold_val_mse))
    train_r2.append(np.mean(fold_train_r2))
    val_r2.append(np.mean(fold_val_r2))

    # 输出每个训练次数下每折的训练结果
    print(f"n_estimators = {n}:")
    for fold in range(10):
        print(f"Fold {fold + 1} 训练集 MSE: {fold_train_mse[fold]}, 训练集 R²: {fold_train_r2[fold]}")
        print(f"Fold {fold + 1} 验证集 MSE: {fold_val_mse[fold]}, 验证集 R²: {fold_val_r2[fold]}")

# 转换为 numpy 数组，便于绘图
train_mse = np.array(train_mse)
val_mse = np.array(val_mse)
train_r2 = np.array(train_r2)
val_r2 = np.array(val_r2)

# 创建双纵坐标图
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('训练次数 (n_estimators)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(n_estimators_range, train_mse, 'o-', label='训练集 MSE', color='tab:blue')
l2 = ax1.plot(n_estimators_range, val_mse, 's-', label='验证集 MSE', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(n_estimators_range, train_r2, 'o--', label='训练集 R²', color='tab:red')
l4 = ax2.plot(n_estimators_range, val_r2, 's--', label='验证集 R²', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)  # 设置 R² 的范围为 [0, 1]

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.title('Bagging回归器学习曲线（MSE + R²）')
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
