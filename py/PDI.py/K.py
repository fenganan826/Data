import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import shap
import matplotlib.pyplot as plt

# ============== 基础设置 ==============
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
shap.initjs()

# 读取 Excel文件
file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

# 特征与目标
X = df.drop('PDI', axis=1)
y = df['PDI']

# 区分类别型和数值型特征
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ============== 划分 80% 训练集 + 20% 测试集 ==============
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 预处理流水线（放进 Pipeline 内，避免数据泄漏）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# ============== 贝叶斯优化：训练集 10 折交叉验证 ==============
def objective(n_neighbors, weights, p):
    n_neighbors = int(n_neighbors)
    weights = 'uniform' if weights < 0.5 else 'distance'
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
            ('regressor', KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p
            ))
        ])

        pipeline.fit(X_tr, y_tr)
        y_pred_tr = pipeline.predict(X_tr)
        y_pred_va = pipeline.predict(X_va)

        # 每折指标
        tr_mse = mean_squared_error(y_tr, y_pred_tr)
        va_mse = mean_squared_error(y_va, y_pred_va)
        tr_r2  = r2_score(y_tr, y_pred_tr)
        va_r2  = r2_score(y_va, y_pred_va)

        fold_mses.append(va_mse)

        # —— 按参考代码格式逐折打印 —— #
        print(f"Fold {len(fold_mses)} - 训练集 MSE: {tr_mse}, 训练集 R²: {tr_r2}")
        print(f"Fold {len(fold_mses)} - 验证集 MSE: {va_mse}, 验证集 R²: {va_r2}")

    # BayesOpt 最大化目标 → 返回 -MSE 的均值
    return -np.mean(fold_mses)

# 搜索空间（可按需调节）
pbounds = {
    'n_neighbors': (5, 8),
    'weights': (0, 1),   # <0.5 → 'uniform'，否则 'distance'
    'p': (1, 2)          # 1: 曼哈顿；2: 欧氏
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)
optimizer.maximize(init_points=5, n_iter=10)

# 最优参数
best_params = optimizer.max['params']
best_n_neighbors = int(best_params['n_neighbors'])
best_weights = 'uniform' if best_params['weights'] < 0.5 else 'distance'
best_p = int(best_params['p'])
print(f"\n✅ 最佳参数: n_neighbors={best_n_neighbors}, weights={best_weights}, p={best_p}")

# ============== 最终模型：在整个训练集上重训 ==============
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(
        n_neighbors=best_n_neighbors,
        weights=best_weights,
        p=best_p
    ))
])
final_pipeline.fit(X_trainval, y_trainval)

# 训练集评估（可选）
y_pred_train = final_pipeline.predict(X_trainval)
rmse_train = np.sqrt(mean_squared_error(y_trainval, y_pred_train))
mae_train = mean_absolute_error(y_trainval, y_pred_train)
r2_train = r2_score(y_trainval, y_pred_train)
print("\n训练集评估结果：")
print(f"RMSE: {rmse_train}")
print(f"MAE : {mae_train}")
print(f"R²  : {r2_train}")

# ============== 学习曲线（基于训练集 10 折 CV 均值） ==============
# ★ 横坐标上限改为 200（步长 2）
neighbor_range = np.arange(1, 201, 2)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_mse, val_mse = [], []
train_r2,  val_r2  = [], []

for k in neighbor_range:
    fold_train_mse, fold_val_mse = [], []
    fold_train_r2,  fold_val_r2  = [], []

    for train_index, val_index in kf.split(X_trainval):
        X_train_fold, X_val_fold = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        # 防止某折训练样本数 < k 导致报错：使用可用的最大 k
        k_used = min(k, len(X_train_fold))

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(
                n_neighbors=k_used,
                weights=best_weights,
                p=best_p
            ))
        ])
        model.fit(X_train_fold, y_train_fold)

        y_train_pred = model.predict(X_train_fold)
        y_val_pred   = model.predict(X_val_fold)

        fold_train_mse.append(mean_squared_error(y_train_fold, y_train_pred))
        fold_val_mse.append(mean_squared_error(y_val_fold, y_val_pred))
        fold_train_r2.append(r2_score(y_train_fold, y_train_pred))
        fold_val_r2.append(r2_score(y_val_fold, y_val_pred))

    # 每个 k 的均值（用于绘图）
    train_mse.append(np.mean(fold_train_mse))
    val_mse.append(np.mean(fold_val_mse))
    train_r2.append(np.mean(fold_train_r2))
    val_r2.append(np.mean(fold_val_r2))

train_mse = np.array(train_mse)
val_mse   = np.array(val_mse)
train_r2  = np.array(train_r2)
val_r2    = np.array(val_r2)

# —— 双纵轴学习曲线（MSE + R²） —— #
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：MSE
ax1.set_xlabel('邻居数（n_neighbors）')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(neighbor_range, train_mse, 'o-', label='训练 MSE', color='tab:blue')
l2 = ax1.plot(neighbor_range, val_mse,   's-', label='验证 MSE', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右轴：R²
ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(neighbor_range, train_r2, 'o--', label='训练 R²', color='tab:red')
l4 = ax2.plot(neighbor_range, val_r2,   's--', label='验证 R²', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)

# 合并图例
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)

plt.title('KNN 学习曲线（MSE + R²，10折CV 平均）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# ============== 测试集评估 ==============
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test  = mean_absolute_error(y_test, y_pred_test)
r2_test   = r2_score(y_test, y_pred_test)

print("\n测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE : {mae_test}")
print(f"R²  : {r2_test}")
