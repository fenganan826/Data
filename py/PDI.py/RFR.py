import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import shap

plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

X = df.drop('PDI', axis=1)
y = df['PDI']

categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_mses = []

    for tr_idx, va_idx in kf.split(X_trainval):
        X_tr = X_trainval.iloc[tr_idx]
        y_tr = y_trainval.iloc[tr_idx]
        X_va = X_trainval.iloc[va_idx]
        y_va = y_trainval.iloc[va_idx]

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            ))
        ])
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_va)
        fold_mses.append(mean_squared_error(y_va, y_pred))

    return -float(np.mean(fold_mses))

pbounds = {
    'n_estimators': (32, 128),
    'max_depth': (2, 8),
    'min_samples_split': (2, 6),
    'min_samples_leaf': (1, 2)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)
optimizer.maximize(init_points=5, n_iter=10)

# ===== 仅输出：调参超参数的最优整数值 =====
best_params = optimizer.max['params']
best_n_estimators      = int(round(best_params['n_estimators']))
best_max_depth         = int(round(best_params['max_depth']))
best_min_samples_split = int(round(best_params['min_samples_split']))
best_min_samples_leaf  = int(round(best_params['min_samples_leaf']))

print("\n🎯 最优超参数（用于模型训练）")
print(f"n_estimators: {best_n_estimators}")
print(f"max_depth: {best_max_depth}")
print(f"min_samples_split: {best_min_samples_split}")
print(f"min_samples_leaf: {best_min_samples_leaf}")

# ===== 用最优超参数训练最终模型 =====
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        random_state=42
    ))
])
final_pipeline.fit(X_trainval, y_trainval)

# ===== 测试集评估 =====
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n✅ 测试集模型评估：")
print(f"测试集均方根误差 (RMSE): {rmse_test}")
print(f"测试集 R² 分数: {r2_test}")
print(f"测试集平均绝对误差 (MAE): {mae_test}")

# ========== 学习曲线：MSE + R² ==========
kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_mse_folds, val_mse_folds, train_r2_folds, val_r2_folds = [], [], [], []

for n_estimators in np.arange(50, 201, 10):
    fold_train_mse, fold_val_mse, fold_train_r2, fold_val_r2 = [], [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(X_trainval), start=1):
        X_train_fold, X_val_fold = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=best_max_depth,
                                    min_samples_split=best_min_samples_split,
                                    min_samples_leaf=best_min_samples_leaf,
                                    random_state=42))])

        model.fit(X_train_fold, y_train_fold)

        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        fold_train_mse.append(mean_squared_error(y_train_fold, y_train_pred))
        fold_val_mse.append(mean_squared_error(y_val_fold, y_val_pred))
        fold_train_r2.append(r2_score(y_train_fold, y_train_pred))
        fold_val_r2.append(r2_score(y_val_fold, y_val_pred))

    train_mse_folds.append(np.mean(fold_train_mse))
    val_mse_folds.append(np.mean(fold_val_mse))
    train_r2_folds.append(np.mean(fold_train_r2))
    val_r2_folds.append(np.mean(fold_val_r2))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('训练次数 (n_estimators)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(np.arange(50, 201, 10), train_mse_folds, 'o-', color='tab:blue', label='训练集 MSE', linewidth=1.2)
l2 = ax1.plot(np.arange(50, 201, 10), val_mse_folds, 's-', color='tab:cyan', label='验证集 MSE', linewidth=1.2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(np.arange(50, 201, 10), train_r2_folds, 'o--', color='tab:red', label='训练集 R²', linewidth=1.2)
l4 = ax2.plot(np.arange(50, 201, 10), val_r2_folds, 's--', color='tab:orange', label='验证集 R²', linewidth=1.2)
ax2.tick_params(axis='y', labelcolor='tab:red')

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.title('学习曲线 - RandomForest（MSE + R²）')
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
