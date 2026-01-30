import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# 目标函数：在训练集上做 10 折交叉验证（贝叶斯优化使用）
def objective(n_estimators, learning_rate, max_depth):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_mses = []
    fold_train_r2_scores = []
    fold_val_r2_scores = []

    # 每折交叉验证
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_trainval), 1):
        X_tr = X_trainval.iloc[tr_idx]
        y_tr = y_trainval.iloc[tr_idx]
        X_va = X_trainval.iloc[va_idx]
        y_va = y_trainval.iloc[va_idx]

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ))
        ])
        pipeline.fit(X_tr, y_tr)
        y_pred_tr = pipeline.predict(X_tr)
        y_pred_va = pipeline.predict(X_va)

        # 计算每折的MSE和R²
        train_mse = mean_squared_error(y_tr, y_pred_tr)
        val_mse = mean_squared_error(y_va, y_pred_va)
        train_r2 = r2_score(y_tr, y_pred_tr)
        val_r2 = r2_score(y_va, y_pred_va)

        fold_mses.append(val_mse)
        fold_train_r2_scores.append(train_r2)
        fold_val_r2_scores.append(val_r2)

        # 输出每折训练集和验证集的结果
        print(f"Fold {fold} 训练集 MSE: {train_mse}, 训练集 R²: {train_r2}")
        print(f"Fold {fold} 验证集 MSE: {val_mse}, 验证集 R²: {val_r2}")

    # 贝叶斯优化最大化目标 → 返回 -MSE 的均值
    return -float(np.mean(fold_mses))


# 贝叶斯优化参数范围
pbounds = {
    'n_estimators': (50, 90),
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 10)  # 保留但不改变模型结构
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
best_n_estimators = int(best_params['n_estimators'])
best_learning_rate = best_params['learning_rate']
best_max_depth = int(best_params['max_depth'])

# 输出最优参数结果
print("\n📌 贝叶斯优化得到的最佳超参数：")
print(f"最优 n_estimators: {best_n_estimators}")
print(f"最优 learning_rate: {best_learning_rate}")
print(f"最优 max_depth: {best_max_depth}")

# 构建最终模型（在整个训练集上训练）
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
        max_depth=best_max_depth,
        random_state=42
    ))
])

final_model.fit(X_trainval, y_trainval)

# ====== 模型测试集评估 ======
y_pred_test = final_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n✅ 测试集模型评估：")
print(f"测试集均方根误差 (RMSE): {rmse_test}")
print(f"测试集 R² 分数: {r2_test}")
print(f"测试集平均绝对误差 (MAE): {mae_test}")

# ========== 学习曲线：MSE + R² ==========
# 使用KFold进行交叉验证的训练集和验证集划分
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储每一折的训练和验证误差（在每个 n_estimators 下）
train_mse_folds = []
val_mse_folds = []
train_r2_folds = []
val_r2_folds = []

# 遍历不同的 n_estimators 值进行训练
for n_estimators in np.arange(50, 129, 10):  # 这里的步长可以调整
    fold_train_mse = []
    fold_val_mse = []
    fold_train_r2 = []
    fold_val_r2 = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_trainval), start=1):
        X_train_fold, X_val_fold = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', GradientBoostingRegressor(
                                    n_estimators=n_estimators,
                                    learning_rate=best_learning_rate,
                                    max_depth=best_max_depth,
                                    random_state=42))])

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
    train_mse_folds.append(np.mean(fold_train_mse))
    val_mse_folds.append(np.mean(fold_val_mse))
    train_r2_folds.append(np.mean(fold_train_r2))
    val_r2_folds.append(np.mean(fold_val_r2))

# 绘制双纵轴学习曲线（MSE + R²）
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：MSE
ax1.set_xlabel('训练次数 (n_estimators)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(np.arange(50, 129, 10), train_mse_folds, 'o-', color='tab:blue', label='训练集 MSE', linewidth=1.2)
l2 = ax1.plot(np.arange(50, 129, 10), val_mse_folds, 's-', color='tab:cyan', label='验证集 MSE', linewidth=1.2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右轴：R²
ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(np.arange(50, 129, 10), train_r2_folds, 'o--', color='tab:red', label='训练集 R²', linewidth=1.2)
l4 = ax2.plot(np.arange(50, 129, 10), val_r2_folds, 's--', color='tab:orange', label='验证集 R²', linewidth=1.2)
ax2.tick_params(axis='y', labelcolor='tab:red')

# 合并图例
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.title('学习曲线 - GBDT（MSE + R²）')
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
