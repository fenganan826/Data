import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import shap

# ================== 基础设置 ==================
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取 Excel 文件
file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

# 特征与目标
X = df.drop('PDI', axis=1)
y = df['PDI']

# 区分类别型和数值型特征
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ================== 数据划分：8:2 ==================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 预处理流水线
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ================== 贝叶斯优化：10 折交叉验证 ==================
def objective(n_estimators, learning_rate, max_depth):
    """
    在训练集上进行 10 折交叉验证，返回 - 平均验证 MSE。
    同时逐折打印训练/验证 MSE 和 R²。
    """
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)  # 保留但不使用

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_train_mses, fold_val_mses = [], []
    fold_train_r2_scores, fold_val_r2_scores = [], []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_trainval), start=1):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            ))
        ])

        pipeline.fit(X_tr, y_tr)

        # 训练集预测
        y_tr_pred = pipeline.predict(X_tr)
        train_mse = mean_squared_error(y_tr, y_tr_pred)
        train_r2 = r2_score(y_tr, y_tr_pred)

        # 验证集预测
        y_va_pred = pipeline.predict(X_va)
        val_mse = mean_squared_error(y_va, y_va_pred)
        val_r2 = r2_score(y_va, y_va_pred)

        # 记录
        fold_train_mses.append(train_mse)
        fold_val_mses.append(val_mse)
        fold_train_r2_scores.append(train_r2)
        fold_val_r2_scores.append(val_r2)

        # 打印逐折结果
        print(f"Fold {fold_id} 训练集 MSE: {train_mse:.6f}, 训练集 R²: {train_r2:.6f}")
        print(f"Fold {fold_id} 验证集 MSE: {val_mse:.6f}, 验证集 R²: {val_r2:.6f}")

    # 汇总均值
    mean_train_mse = float(np.mean(fold_train_mses))
    mean_val_mse = float(np.mean(fold_val_mses))
    mean_train_r2 = float(np.mean(fold_train_r2_scores))
    mean_val_r2 = float(np.mean(fold_val_r2_scores))

    print("-" * 60)
    print(f"[CV-10] 平均训练 MSE: {mean_train_mse:.6f}, 平均训练 R²: {mean_train_r2:.6f}")
    print(f"[CV-10] 平均验证 MSE: {mean_val_mse:.6f}, 平均验证 R²: {mean_val_r2:.6f}")
    print("-" * 60)

    return -mean_val_mse

# 搜索空间
pbounds = {
    'n_estimators': (32, 128),
    'learning_rate': (1e-2, 0.3),
    'max_depth': (3, 10)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)
optimizer.maximize(init_points=5, n_iter=10)

# 最优参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_learning_rate = best_params['learning_rate']
best_max_depth = int(best_params['max_depth'])

print("\n✅ 贝叶斯优化完成：")
print(f"最佳参数: n_estimators={best_n_estimators}, learning_rate={best_learning_rate:.6f}, max_depth={best_max_depth}")

# ================== 最终模型训练 ==================
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', AdaBoostRegressor(
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
        random_state=42
    ))
])

final_pipeline.fit(X_trainval, y_trainval)

# ================== 模型评估 ==================
# 训练集
y_pred_train = final_pipeline.predict(X_trainval)
rmse_train = np.sqrt(mean_squared_error(y_trainval, y_pred_train))
r2_train = r2_score(y_trainval, y_pred_train)
mae_train = mean_absolute_error(y_trainval, y_pred_train)

print("\n✅ 训练集模型评估：")
print(f"训练集 RMSE: {rmse_train}")
print(f"训练集 R²: {r2_train}")
print(f"训练集 MAE: {mae_train}")

# 测试集
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n✅ 测试集模型评估：")
print(f"测试集 RMSE: {rmse_test}")
print(f"测试集 R²: {r2_test}")
print(f"测试集 MAE: {mae_test}")

# ================== 学习曲线（10 折 CV 平均） ==================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
n_estimators_range = np.arange(10, 201, 10)

train_mse_folds, val_mse_folds = [], []
train_r2_folds, val_r2_folds = [], []

for n in n_estimators_range:
    fold_train_mse, fold_val_mse = [], []
    fold_train_r2, fold_val_r2 = [], []

    for tr_idx, va_idx in kf.split(X_trainval):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', AdaBoostRegressor(
                n_estimators=n,
                learning_rate=best_learning_rate,
                random_state=42
            ))
        ])
        model.fit(X_tr, y_tr)

        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)

        fold_train_mse.append(mean_squared_error(y_tr, y_tr_pred))
        fold_val_mse.append(mean_squared_error(y_va, y_va_pred))
        fold_train_r2.append(r2_score(y_tr, y_tr_pred))
        fold_val_r2.append(r2_score(y_va, y_va_pred))

    train_mse_folds.append(np.mean(fold_train_mse))
    val_mse_folds.append(np.mean(fold_val_mse))
    train_r2_folds.append(np.mean(fold_train_r2))
    val_r2_folds.append(np.mean(fold_val_r2))

# 转为数组
train_mse_folds = np.array(train_mse_folds)
val_mse_folds = np.array(val_mse_folds)
train_r2_folds = np.array(train_r2_folds)
val_r2_folds = np.array(val_r2_folds)

# —— 图1：MSE 曲线 —— #
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_mse_folds, 'o-', label='训练误差 (MSE)')
plt.plot(n_estimators_range, val_mse_folds, 's-', label='验证误差 (MSE)')
plt.xlabel('训练次数（n_estimators）')
plt.ylabel('均方误差 (MSE)')
plt.title('学习曲线 - AdaBoost Regressor（10折CV 平均）')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(np.arange(0, 210, 10))
plt.tight_layout()

# —— 图2：MSE + R² 双纵轴 —— #
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('训练次数（n_estimators）')
ax1.set_ylabel('均方误差 (MSE)')
l1 = ax1.plot(n_estimators_range, train_mse_folds, 'o-', label='训练误差 (MSE)')
l2 = ax1.plot(n_estimators_range, val_mse_folds, 's-', label='验证误差 (MSE)')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²')
l3 = ax2.plot(n_estimators_range, train_r2_folds, 'o--', label='训练集 R²')
l4 = ax2.plot(n_estimators_range, val_r2_folds, 's--', label='验证集 R²')
ax2.tick_params(axis='y')

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.title('学习曲线 - AdaBoost（MSE + R²，10折CV 平均）')
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()
