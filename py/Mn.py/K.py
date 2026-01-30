import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import shap

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化SHAP
shap.initjs()

# 读取 Excel 文件
file_path = r'F:/Mn_standardized.xlsx'
df = pd.read_excel(file_path)

# 确定特征和目标变量
X = df.drop('Mn(×104g/mol）', axis=1)
y = df['Mn(×104g/mol）']

# 区分类别型和数值型特征（保持不变）
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 划分训练集和测试集（80% 训练，20% 测试）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建预处理流水线（保持不变）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# 目标函数：在训练集上做 10 折交叉验证（贝叶斯优化使用）
def objective(n_neighbors, weights, p):
    n_neighbors = int(n_neighbors)
    if weights < 0.5:
        weights = 'uniform'
    else:
        weights = 'distance'
    p = int(p)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_mses = []

    for tr_idx, va_idx in kf.split(X_trainval):
        X_tr = X_trainval.iloc[tr_idx]
        y_tr = y_trainval.iloc[tr_idx]
        X_va = X_trainval.iloc[va_idx]
        y_va = y_trainval.iloc[va_idx]

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=n_neighbors,
                                              weights=weights,
                                              p=p))
        ])
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_va)

        fold_mses.append(mean_squared_error(y_va, y_pred))

        # 输出每一折训练集和验证集的结果
        print(f"Fold {len(fold_mses)} - 训练集 MSE: {mean_squared_error(y_tr, pipeline.predict(X_tr))}, "
              f"训练集 R²: {r2_score(y_tr, pipeline.predict(X_tr))}")
        print(f"Fold {len(fold_mses)} - 验证集 MSE: {mean_squared_error(y_va, y_pred)}, "
              f"验证集 R²: {r2_score(y_va, y_pred)}")

    # 贝叶斯优化最大化目标 → 返回 -MSE 的均值
    return -np.mean(fold_mses)

# 贝叶斯优化参数范围
pbounds = {
    'n_neighbors': (5, 20),
    'weights': (0, 1),
    'p': (1, 2)
}

# 执行贝叶斯优化（基于 10 折 CV）
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)
optimizer.maximize(init_points=5, n_iter=10)

# 获取最佳参数
best_params = optimizer.max['params']
best_n_neighbors = int(best_params['n_neighbors'])
if best_params['weights'] < 0.5:
    best_weights = 'uniform'
else:
    best_weights = 'distance'
best_p = int(best_params['p'])

# 使用最佳参数构建最终模型
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=best_n_neighbors,
                                      weights=best_weights,
                                      p=best_p))
])

# 使用整个训练集进行训练
final_pipeline.fit(X_trainval, y_trainval)

# ✅ 使用训练集评估模型
y_pred_train = final_pipeline.predict(X_trainval)
rmse_train = np.sqrt(mean_squared_error(y_trainval, y_pred_train))
mae_train = mean_absolute_error(y_trainval, y_pred_train)
r2_train = r2_score(y_trainval, y_pred_train)

print("训练集评估结果：")
print(f'RMSE: {rmse_train}')
print(f'MAE: {mae_train}')
print(f'R²: {r2_train}')
print(f'最佳参数: n_neighbors={best_n_neighbors}, weights={best_weights}, p={best_p}')

# ========== 双纵坐标轴学习曲线（MSE + R²） ==========

neighbor_range = np.arange(1, 51, 2)
train_mse, val_mse = [], []
train_r2, val_r2 = [], []

# 使用KFold进行交叉验证的训练集和验证集划分
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for k in neighbor_range:
    fold_train_mse = []
    fold_val_mse = []
    fold_train_r2 = []
    fold_val_r2 = []

    # 对每一折进行训练和验证
    for train_index, val_index in kf.split(X_trainval):
        X_train_fold, X_val_fold = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=k,
                                              weights=best_weights,
                                              p=best_p))
        ])
        model.fit(X_train_fold, y_train_fold)

        # 预测训练集和验证集
        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        # 计算每一折的MSE和R²
        fold_train_mse.append(mean_squared_error(y_train_fold, y_train_pred))
        fold_val_mse.append(mean_squared_error(y_val_fold, y_val_pred))
        fold_train_r2.append(r2_score(y_train_fold, y_train_pred))
        fold_val_r2.append(r2_score(y_val_fold, y_val_pred))

    # 计算每个训练次数下的平均误差和R²
    train_mse.append(np.mean(fold_train_mse))
    val_mse.append(np.mean(fold_val_mse))
    train_r2.append(np.mean(fold_train_r2))
    val_r2.append(np.mean(fold_val_r2))

train_mse = np.array(train_mse)
val_mse = np.array(val_mse)
train_r2 = np.array(train_r2)
val_r2 = np.array(val_r2)

# 创建双纵坐标图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：MSE
ax1.set_xlabel('邻居数（n_neighbors）')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(neighbor_range, train_mse, 'o-', label='训练 MSE', color='tab:blue')
l2 = ax1.plot(neighbor_range, val_mse, 's-', label='验证 MSE', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右轴：R²
ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(neighbor_range, train_r2, 'o--', label='训练 R²', color='tab:red')
l4 = ax2.plot(neighbor_range, val_r2, 's--', label='验证 R²', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)

# 合并图例
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)

plt.title('KNN 学习曲线（MSE + R²）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# ========== 测试集评估 ==========

y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")
