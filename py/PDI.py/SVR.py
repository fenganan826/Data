import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import shap
import matplotlib.pyplot as plt

# 字体设置，兼容负号 \u2212
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化SHAP（当前脚本未用到SHAP，仅保持与你原脚本一致）
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

# 划分训练集和测试集（80% 训练，20% 测试）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 预处理流水线
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# 目标函数：对训练集做 10 折交叉验证（用于贝叶斯优化）
def objective(C, gamma, epsilon):
    C = float(C)
    gamma = float(gamma)
    epsilon = float(epsilon)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_val_mses = []
    fold_train_mses, fold_train_r2s = [], []
    fold_val_r2s = []

    # 10 折交叉验证
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_trainval), start=1):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(C=C, gamma=gamma, epsilon=epsilon, kernel='rbf'))
        ])

        pipeline.fit(X_tr, y_tr)

        # 训练/验证集预测
        y_tr_pred = pipeline.predict(X_tr)
        y_va_pred = pipeline.predict(X_va)

        # 训练/验证集指标
        train_mse = mean_squared_error(y_tr, y_tr_pred)
        val_mse = mean_squared_error(y_va, y_va_pred)
        train_r2 = r2_score(y_tr, y_tr_pred)
        val_r2 = r2_score(y_va, y_va_pred)

        fold_train_mses.append(train_mse)
        fold_val_mses.append(val_mse)
        fold_train_r2s.append(train_r2)
        fold_val_r2s.append(val_r2)

        # —— 打印每折训练/验证结果 —— #
        print(
            f"Fold {fold_idx} | 训练 MSE: {train_mse:.6f}, 训练 R²: {train_r2:.6f} | "
            f"验证 MSE: {val_mse:.6f}, 验证 R²: {val_r2:.6f}"
        )

    # 输出每折的平均值
    mean_train_mse = np.mean(fold_train_mses)
    mean_val_mse = np.mean(fold_val_mses)
    mean_train_r2 = np.mean(fold_train_r2s)
    mean_val_r2 = np.mean(fold_val_r2s)

    print(f"10 折交叉验证 - 训练集 MSE 平均值: {mean_train_mse:.6f}")
    print(f"10 折交叉验证 - 验证集 MSE 平均值: {mean_val_mse:.6f}")
    print(f"10 折交叉验证 - 训练集 R² 平均值: {mean_train_r2:.6f}")
    print(f"10 折交叉验证 - 验证集 R² 平均值: {mean_val_r2:.6f}")
    print("-" * 70)

    # 以验证 MSE 的平均值作为优化目标（最小化 MSE -> 最大化 -MSE）
    return -mean_val_mse

# 定义参数搜索范围
pbounds = {
    'C': (0.01, 10),
    'gamma': (0.01, 1),
    'epsilon': (0.01, 0.5)
}

# 启动贝叶斯优化（基于 10 折 CV）
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)
optimizer.maximize(init_points=5, n_iter=10)

# 获取最优参数
best_params = optimizer.max['params']
best_C = float(best_params['C'])
best_gamma = float(best_params['gamma'])
best_epsilon = float(best_params['epsilon'])

print(f"\n⭐ 最优参数（CV-BO）: C={best_C:.6f}, gamma={best_gamma:.6f}, epsilon={best_epsilon:.6f}")

# 用最优参数在整个训练集上训练最终模型
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(C=best_C, gamma=best_gamma, epsilon=best_epsilon, kernel='rbf'))
])
final_pipeline.fit(X_trainval, y_trainval)

# ========== 测试集评估（补回） ==========
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n✅ 测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")
