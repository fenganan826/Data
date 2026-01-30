import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     ReLU, ELU, Concatenate, Activation)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from bayes_opt import BayesianOptimization

# ====================== 全局配置 ======================
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# ====================== 数据读取与预处理 ======================
file_path = r'F:/Mn_standardized.xlsx'
df = pd.read_excel(file_path)

numerical_cols = ['Cat(umol)', 'Al/M(molar)', 't/min', 'T/℃', 'R1', 'R2']
one_hot_cols = ['M_Zr', 'M_Ti', 'M_Hf', 'R3']
target_col = 'Mn(×104g/mol）'

X_num = df[numerical_cols].values
X_cat = df[one_hot_cols].values
y = df[target_col].values

# 8:2 划分（训练验证总体 trainval vs 测试集 test）
X_num_trainval, X_num_test, X_cat_trainval, X_cat_test, y_trainval, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# ====================== 自定义 R² 指标 ======================
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# ====================== 激活函数映射（与参考代码一致） ======================
activation_map = {
    'relu': ReLU(),
    'elu': ELU(),
    'swish': Activation(tf.nn.swish),
    'gelu': Activation(tf.nn.gelu)
}
activation_choices = list(activation_map.keys())

# ====================== 贝叶斯优化目标函数（10 折；逐折打印；结构=四层） ======================
def build_and_evaluate_model_cv(units_num, units_cat,
                                dropout_num1, dropout_num2, dropout_num3, dropout_num4,
                                dropout_cat1, dropout_cat2, dropout_cat3, dropout_cat4,
                                learning_rate, l2_reg, batch_size,
                                act_choice_idx, fusion_dim):

    # 基本类型转换
    units_num = int(units_num)
    units_cat = int(units_cat)
    batch_size = int(batch_size)
    fusion_dim = int(fusion_dim)
    act_choice = activation_choices[int(act_choice_idx)]
    act_layer = activation_map[act_choice]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    val_scores = []

    print("\n================== 开始 10 折交叉验证（用于贝叶斯优化） ==================")
    print(f"激活: {act_choice} | lr={learning_rate:.6g} | l2={l2_reg:.6g} | "
          f"units_num={units_num} | units_cat={units_cat} | fusion_dim={fusion_dim} | batch_size={batch_size}")
    print("=====================================================================\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_trainval), start=1):
        Xn_train, Xn_val = X_num_trainval[train_idx], X_num_trainval[val_idx]
        Xc_train, Xc_val = X_cat_trainval[train_idx], X_cat_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        # ===== 数值分支：四层 Dense + 四次 Dropout（首层含 BN） =====
        input_num = Input(shape=(Xn_train.shape[1],))
        x_num = Dense(units_num, kernel_regularizer=regularizers.l1_l2(1e-5, l2_reg))(input_num)
        x_num = BatchNormalization()(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num1)(x_num)

        x_num = Dense(units_num)(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num2)(x_num)

        x_num = Dense(units_num // 2)(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num3)(x_num)

        x_num = Dense(units_num // 4)(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num4)(x_num)

        # ===== 类别分支：四层 Dense + 四次 Dropout =====
        input_cat = Input(shape=(Xc_train.shape[1],))
        x_cat = Dense(units_cat)(input_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat1)(x_cat)

        x_cat = Dense(units_cat)(x_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat2)(x_cat)

        x_cat = Dense(units_cat // 2)(x_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat3)(x_cat)

        x_cat = Dense(units_cat // 4)(x_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat4)(x_cat)

        # 融合 + 输出
        x = Concatenate()([x_num, x_cat])
        x = Dense(fusion_dim, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=[input_num, input_cat], outputs=output)

        optimizer = AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.Huber(delta=1.0),  # BO 阶段使用 Huber（抗异常值）
                      metrics=['mae', r2_metric])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]

        print(f"▶️ 折 {fold:02d} 开始训练...")
        history = model.fit([Xn_train, Xc_train], y_train,
                            validation_data=([Xn_val, Xc_val], y_val),
                            epochs=200,
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=callbacks)

        # —— 打印每折最后一轮的训练/验证指标（注意：loss=Huber） ——
        tr_r2_last   = history.history['r2_metric'][-1]
        va_r2_last   = history.history['val_r2_metric'][-1]
        tr_loss_last = history.history['loss'][-1]
        va_loss_last = history.history['val_loss'][-1]
        print(f"折 {fold:02d} 训练集 R²: {tr_r2_last:.6f} | 验证集 R²: {va_r2_last:.6f}")
        print(f"折 {fold:02d} 训练集 Huber: {tr_loss_last:.6f} | 验证集 Huber: {va_loss_last:.6f}")

        # —— sklearn 评估验证集：MSE/R²（用于综合分数） ——
        y_val_pred = model.predict([Xn_val, Xc_val], verbose=0).flatten()
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        score = 0.3 * mse + 0.7 * (1 - r2)  # 越小越好
        val_scores.append(score)

        print(f"折 {fold:02d} 验证集 MSE: {mse:.6f} | 验证集 R²: {r2:.6f} | 综合分数(0.3*MSE+0.7*(1-R²)): {score:.6f}\n")

    mean_score = np.mean(val_scores)
    print("============== 10 折交叉验证完成 ==============")
    print(f"10 折验证综合分数均值（越小越好）: {mean_score:.6f}")
    print("（注：贝叶斯优化目标是最大化，因此会对该均值取负）")
    print("=============================================\n")

    # BO 目标函数要最大化，返回负号
    return -mean_score

# ====================== 贝叶斯优化参数空间（含四个 dropout） ======================
pbounds = {
    'units_num': (32, 128),
    'units_cat': (32, 156),
    'fusion_dim': (64, 128),
    'dropout_num1': (0.15, 0.2),
    'dropout_num2': (0.1, 0.2),
    'dropout_num3': (0.1, 0.3),
    'dropout_num4': (0.1, 0.3),
    'dropout_cat1': (0.1, 0.3),
    'dropout_cat2': (0.15, 0.3),
    'dropout_cat3': (0.1, 0.3),
    'dropout_cat4': (0.1, 0.3),
    'learning_rate': (3e-4, 8e-4),
    'l2_reg': (5e-6, 5e-5),
    'batch_size': (32, 64),
    'act_choice_idx': (0, len(activation_choices) - 1)
}

optimizer = BayesianOptimization(
    f=build_and_evaluate_model_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=10)

best = optimizer.max['params']
for key in ['units_num', 'units_cat', 'fusion_dim', 'batch_size', 'act_choice_idx']:
    best[key] = int(best[key])
best['act_choice'] = activation_choices[best['act_choice_idx']]

print("\n📌 最佳超参数：")
for k, v in best.items():
    print(f"{k}: {v}")

# ====================== 构建交叉验证学习曲线（10 折均值；损失=MSE；结构=四层） ======================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
num_epochs = 200  # 统一为200

train_mse_all = np.zeros((num_epochs, 10))
val_mse_all = np.zeros((num_epochs, 10))
train_r2_all = np.zeros((num_epochs, 10))
val_r2_all = np.zeros((num_epochs, 10))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_trainval), start=1):
    Xn_train, Xn_val = X_num_trainval[train_idx], X_num_trainval[val_idx]
    Xc_train, Xc_val = X_cat_trainval[train_idx], X_cat_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    act_layer = activation_map[best['act_choice']]

    # 数值分支：四层
    input_num = Input(shape=(Xn_train.shape[1],))
    x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(1e-5, best['l2_reg']))(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num1'])(x_num)
    x_num = Dense(best['units_num'])(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num2'])(x_num)
    x_num = Dense(best['units_num'] // 2)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num3'])(x_num)
    x_num = Dense(best['units_num'] // 4)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num4'])(x_num)

    # 类别分支：四层
    input_cat = Input(shape=(Xc_train.shape[1],))
    x_cat = Dense(best['units_cat'])(input_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat1'])(x_cat)
    x_cat = Dense(best['units_cat'])(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat2'])(x_cat)
    x_cat = Dense(best['units_cat'] // 2)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat3'])(x_cat)
    x_cat = Dense(best['units_cat'] // 4)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat4'])(x_cat)

    x = Concatenate()([x_num, x_cat])
    x = Dense(best['fusion_dim'], activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[input_num, input_cat], outputs=output)

    optimizer_final = AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg'])
    # 学习曲线阶段：显式使用 MSE 作为损失
    model.compile(optimizer=optimizer_final, loss='mse', metrics=[r2_metric])

    history = model.fit([Xn_train, Xc_train], y_train,
                        validation_data=([Xn_val, Xc_val], y_val),
                        epochs=num_epochs,
                        batch_size=best['batch_size'],
                        verbose=0)

    train_mse_all[:, fold-1] = history.history['loss']       # MSE
    val_mse_all[:, fold-1]  = history.history['val_loss']    # MSE
    train_r2_all[:, fold-1] = history.history['r2_metric']   # R²
    val_r2_all[:, fold-1]   = history.history['val_r2_metric']

# ====================== 学习曲线绘图（改为参考代码的风格；不改思路） ======================
mean_train_mse = train_mse_all.mean(axis=1)
mean_val_mse = val_mse_all.mean(axis=1)
mean_train_r2 = train_r2_all.mean(axis=1)
mean_val_r2 = val_r2_all.mean(axis=1)

epochs = range(1, num_epochs + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))

# 左轴：MSE（参考风格）
ax1.set_xlabel('训练次数 (epoch)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(epochs, mean_train_mse, 'o-', label='训练 MSE', color='tab:blue')
l2 = ax1.plot(epochs, mean_val_mse, 's-', label='验证 MSE', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右轴：R²（参考风格）
ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(epochs, mean_train_r2, 'o--', label='训练 R²', color='tab:red')
l4 = ax2.plot(epochs, mean_val_r2, 's--', label='验证 R²', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))

# 合并图例（参考风格）
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)

plt.title('训练过程学习曲线（MSE + R²）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# ====================== 最终模型与测试评估（用整个 trainval；损失=Huber） ======================
act_layer = activation_map[best['act_choice']]

input_num = Input(shape=(X_num.shape[1],))
x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(1e-5, best['l2_reg']))(input_num)
x_num = BatchNormalization()(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num1'])(x_num)
x_num = Dense(best['units_num'])(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num2'])(x_num)
x_num = Dense(best['units_num'] // 2)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num3'])(x_num)
x_num = Dense(best['units_num'] // 4)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num4'])(x_num)

input_cat = Input(shape=(X_cat.shape[1],))
x_cat = Dense(best['units_cat'])(input_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat1'])(x_cat)
x_cat = Dense(best['units_cat'])(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat2'])(x_cat)
x_cat = Dense(best['units_cat'] // 2)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat3'])(x_cat)
x_cat = Dense(best['units_cat'] // 4)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat4'])(x_cat)

x = Concatenate()([x_num, x_cat])
x = Dense(best['fusion_dim'], activation='relu')(x)
output = Dense(1)(x)

final_model = Model(inputs=[input_num, input_cat], outputs=output)

optimizer_final = AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg'])
# 与 BO 阶段保持一致：使用 Huber（更抗异常值）；如需与学习曲线一致可改 'mse'
final_model.compile(optimizer=optimizer_final,
                    loss=tf.keras.losses.Huber(delta=1.0),
                    metrics=['mae', r2_metric])

callbacks_final = [
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-6)
]

history_final = final_model.fit([X_num_trainval, X_cat_trainval], y_trainval,
                                epochs=200,
                                batch_size=best['batch_size'],
                                verbose=1,
                                callbacks=callbacks_final)

# ====================== 训练集与测试集的评估 ======================

# 对训练集进行评估
y_trainval_pred = final_model.predict([X_num_trainval, X_cat_trainval], verbose=0).flatten()
mse_trainval = mean_squared_error(y_trainval, y_trainval_pred)
mae_trainval = mean_absolute_error(y_trainval, y_trainval_pred)
r2_trainval = r2_score(y_trainval, y_trainval_pred)

print("\n📊 最终模型评估结果（训练集）：")
print(f"训练集 MSE: {mse_trainval}")
print(f"训练集 MAE: {mae_trainval}")
print(f"训练集 R²: {r2_trainval}")

# 对测试集进行评估
y_pred_test = final_model.predict([X_num_test, X_cat_test], verbose=0).flatten()

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n📊 最终模型评估结果（测试集）：")
print(f"测试集 MSE: {mse_test}")
print(f"测试集 MAE: {mae_test}")
print(f"测试集 R²: {r2_test}")

# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, c='dodgerblue', alpha=0.7, edgecolors='k', label='预测点')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='理想预测线')
plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('测试集：真实值 vs 预测值', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存预测结果
save_path = r'F:/test_pred_vs_true.xlsx'
pd.DataFrame({'真实值': y_test, '预测值': y_pred_test}).to_excel(save_path, index=False)
print(f"✅ 已将测试集真实值与预测值保存到 {save_path}")
