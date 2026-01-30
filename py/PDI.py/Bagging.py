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

# ================== 基础设置 ==================
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
shap.initjs()

# ================== 读取数据 ==================
file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

# 特征与目标
X = df.drop('PDI', axis=1)
y = df['PDI']

# 特征分类
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ================== 数据划分：80% 训练 + 20% 测试 ==================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# ================== 贝叶斯优化（10 折交叉验证） ==================
def objective(n_estimators, max_samples, max_features, bootstrap, bootstrap_features):
    """
    用 10 折交叉验证评估训练集上的负MSE（cross_val_score 返回的是 neg_mean_squared_error），
    取均值作为贝叶斯优化的目标（最大化）。
    """
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

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    neg_mse_cv_mean = cross_val_score(
        pipeline, X_trainval, y_trainval,
        scoring='neg_mean_squared_error',
        cv=cv
    ).mean()

    # 直接返回平均的负MSE，BayesOpt 会最大化它
    return float(neg_mse_cv_mean)

# 搜索空间
pbounds = {
    'n_estimators': (64, 200),
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
optimizer.maximize(init_points=5, n_iter=20)

# 提取最优参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_samples = best_params['max_samples']
best_max_features = best_params['max_features']
best_bootstrap = bool(round(best_params['bootstrap']))
best_bootstrap_features = bool(round(best_params['bootstrap_features']))

print(f'\n✅ 最佳参数: n_estimators={best_n_estimators}, max_samples={best_max_samples}, '
      f'max_features={best_max_features}, bootstrap={best_bootstrap}, '
      f'bootstrap_features={best_bootstrap_features}')

# ================== 使用整个训练集训练最终模型 ==================
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

# ================== 在测试集评估最终模型 ==================
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n✅ 测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")

# ================== 学习曲线（基于训练集10折CV） ==================
# 你原来固定 10~200（步长10）的思路也保留，这里结合 KFold 得到更稳健的曲线
n_estimators_range = np.arange(10, 201, 10)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_mse_means, val_mse_means = [], []
train_r2_means,  val_r2_means  = [], []

for n in n_estimators_range:
    fold_train_mse, fold_val_mse = [], []
    fold_train_r2,  fold_val_r2  = [], []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_trainval), start=1):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

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

        model.fit(X_tr, y_tr)

        # 训练/验证预测
        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)

        # 指标
        tr_mse = mean_squared_error(y_tr, y_tr_pred)
        va_mse = mean_squared_error(y_va, y_va_pred)
        tr_r2  = r2_score(y_tr, y_tr_pred)
        va_r2  = r2_score(y_va, y_va_pred)

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
train_r2_means  = np.array(train_r2_means)
val_r2_means    = np.array(val_r2_means)

# —— 图1：MSE 学习曲线 —— #
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_mse_means, 'o-', color='blue', label='训练误差 (MSE)', linewidth=1.2, markersize=4)
plt.plot(n_estimators_range, val_mse_means,   's-', color='red',  label='验证误差 (MSE)', linewidth=1.2, markersize=4)
plt.xlabel('训练次数（n_estimators）')
plt.ylabel('均方误差 (MSE)')
plt.title('学习曲线 - Bagging Regressor（10折CV 平均）')
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
l1 = ax1.plot(n_estimators_range, train_mse_means, 'o-', color='tab:blue',   label='训练误差 (MSE)', linewidth=1.2)
l2 = ax1.plot(n_estimators_range, val_mse_means,   's-', color='tab:cyan',   label='验证误差 (MSE)', linewidth=1.2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(n_estimators_range, train_r2_means,  'o--', color='tab:red',   label='训练集 R²', linewidth=1.2)
l4 = ax2.plot(n_estimators_range, val_r2_means,    's--', color='tab:orange',label='验证集 R²', linewidth=1.2)
ax2.tick_params(axis='y', labelcolor='tab:red')

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.title('学习曲线 - Bagging（MSE + R²，10折CV 平均）')
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()
