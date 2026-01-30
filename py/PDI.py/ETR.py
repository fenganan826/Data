import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import shap

# 初始化SHAP
shap.initjs()

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

# 特征与目标
X = df.drop('PDI', axis=1)
y = df['PDI']

# 区分类别型和数值型特征
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 数据划分为 80% 训练集 和 20% 测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建预处理流水线
preprocessor = ColumnTransformer(
    transformers=[('num', 'passthrough', numeric_cols),
                  ('cat', OneHotEncoder(), categorical_cols)])

# 贝叶斯优化目标函数（使用 10 折交叉验证）
def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth) if max_depth is not None else None
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', ExtraTreesRegressor(n_estimators=n_estimators,
                                                                 max_depth=max_depth,
                                                                 min_samples_split=min_samples_split,
                                                                 min_samples_leaf=min_samples_leaf,
                                                                 random_state=42))])

    # 使用10折交叉验证评估模型
    scores = cross_val_score(pipeline, X_trainval, y_trainval,
                             scoring='neg_mean_squared_error',
                             cv=KFold(n_splits=10, shuffle=True, random_state=42))
    return scores.mean()  # 贝叶斯优化最大化目标函数

# 定义参数搜索范围，移除了max_features
pbounds = {
    'n_estimators': (32, 200),  # 更多树，提升稳定性
    'max_depth': (3, 10),  # 限制深度
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# 进行优化，设置迭代次数
optimizer.maximize(
    init_points=5,
    n_iter=10,
)

# 获取最佳参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_depth = int(best_params['max_depth']) if 'max_depth' in best_params else None
best_min_samples_split = int(best_params['min_samples_split'])
best_min_samples_leaf = int(best_params['min_samples_leaf'])

print(f'\n✅ 最佳参数: n_estimators={best_n_estimators}, max_depth={best_max_depth}, '
      f'min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}')

# ================== 使用整个训练集训练最终模型 ==================
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', ExtraTreesRegressor(n_estimators=best_n_estimators,
                                                                   max_depth=best_max_depth,
                                                                   min_samples_split=best_min_samples_split,
                                                                   min_samples_leaf=best_min_samples_leaf,
                                                                   random_state=42))])

final_pipeline.fit(X_trainval, y_trainval)

# ================== 在测试集评估最终模型 ==================
y_pred_test = final_pipeline.predict(X_test)

# 测试集评估
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n✅ 测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")

# ================== 学习曲线（基于训练集10折CV） ==================
# 你原来固定 10~200（步长10）的思路也保留，这里结合 KFold 得到更稳健的曲线
n_estimators_range = np.arange(10, 101, 5)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_mse_means, val_mse_means = [], []
train_r2_means, val_r2_means = [], []

for n in n_estimators_range:
    fold_train_mse, fold_val_mse = [], []
    fold_train_r2, fold_val_r2 = [], []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_trainval), start=1):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

        model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', ExtraTreesRegressor(
                                   n_estimators=n,
                                   max_depth=best_max_depth,
                                   min_samples_split=best_min_samples_split,
                                   min_samples_leaf=best_min_samples_leaf,
                                   random_state=42))])

        model.fit(X_tr, y_tr)

        # 训练/验证预测
        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)

        # 指标
        tr_mse = mean_squared_error(y_tr, y_tr_pred)
        va_mse = mean_squared_error(y_va, y_va_pred)
        tr_r2 = r2_score(y_tr, y_tr_pred)
        va_r2 = r2_score(y_va, y_va_pred)

        fold_train_mse.append(tr_mse)
        fold_val_mse.append(va_mse)
        fold_train_r2.append(tr_r2)
        fold_val_r2.append(va_r2)

        # —— 逐折详细打印（与参考思路一致）——
        print(f"n_estimators={n} | Fold {fold_id:02d} 训练集 MSE: {tr_mse:.6f}, 训练集 R²: {tr_r2:.6f}")
        print(f"n_estimators={n} | Fold {fold_id:02d} 验证集 MSE: {va_mse:.6f}, 验证集 R²: {va_r2:.6f}")

    # 每个 n 下的均值（用于绘图）
    train_mse_means.append(np.mean(fold_train_mse))
    val_mse_means.append(np.mean(fold_val_mse))
    train_r2_means.append(np.mean(fold_train_r2))
    val_r2_means.append(np.mean(fold_val_r2))

# 转为 numpy 数组
train_mse_means = np.array(train_mse_means)
val_mse_means = np.array(val_mse_means)
train_r2_means = np.array(train_r2_means)
val_r2_means = np.array(val_r2_means)

# —— 图1：MSE 学习曲线 —— #
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_mse_means, 'o-', color='blue', label='训练误差 (MSE)', linewidth=1.2, markersize=4)
plt.plot(n_estimators_range, val_mse_means, 's-', color='red', label='验证误差 (MSE)', linewidth=1.2, markersize=4)
plt.xlabel('训练次数（n_estimators）')
plt.ylabel('均方误差 (MSE)')
plt.title('学习曲线 - ExtraTrees Regressor（10折CV 平均）')
plt.legend(loc='best')
plt.grid(True)

y_min = min(train_mse_means.min(), val_mse_means.min())
y_max = max(train_mse_means.max(), val_mse_means.max())
y_ticks = np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1)
plt.yticks(y_ticks)
plt.xticks(np.arange(0, 210, 10))
plt.tight_layout()

# —— 图2：MSE + R² 双纵轴 —— #
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('训练次数（n_estimators）')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(n_estimators_range, train_mse_means, 'o-', color='tab:blue', label='训练误差 (MSE)', linewidth=1.2)
l2 = ax1.plot(n_estimators_range, val_mse_means, 's-', color='tab:cyan', label='验证误差 (MSE)', linewidth=1.2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(n_estimators_range, train_r2_means, 'o--', color='tab:red', label='训练集 R²', linewidth=1.2)
l4 = ax2.plot(n_estimators_range, val_r2_means, 's--', color='tab:orange', label='验证集 R²', linewidth=1.2)
ax2.tick_params(axis='y', labelcolor='tab:red')

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.title('学习曲线 - ExtraTrees（MSE + R²，10折CV 平均）')
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()
