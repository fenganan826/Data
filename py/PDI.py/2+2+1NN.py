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

plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据读取与预处理
file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

numerical_cols = ['Cat(umol)', 'Al/M(molar)', 't/min', 'T/℃', 'R1', 'R2']
one_hot_cols = ['M_Zr', 'M_Ti', 'M_Hf', 'R3']
target_col = 'PDI'

X_num = df[numerical_cols].values
X_cat = df[one_hot_cols].values
y = df[target_col].values

# 固定种子
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# 8:2 划分训练集和测试集
X_num_trainval, X_num_test, X_cat_trainval, X_cat_test, y_trainval, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# 自定义 R²
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

activation_map = {
    'relu': ReLU(),
    'elu': ELU(),
    'swish': Activation(tf.nn.swish),
    'gelu': Activation(tf.nn.gelu)
}
activation_choices = list(activation_map.keys())

# 贝叶斯目标函数（10折交叉验证）
def build_and_evaluate_model_cv(units_num, units_cat, dropout_num, dropout_cat,
                                dropout_num_2, dropout_cat_2,
                                learning_rate, l2_reg, batch_size, act_choice_idx, fusion_dim):
    units_num = int(units_num)
    units_cat = int(units_cat)
    batch_size = int(batch_size)
    fusion_dim = int(fusion_dim)
    act_choice = activation_choices[int(act_choice_idx)]
    act_layer = activation_map[act_choice]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    val_scores = []  # 保存每一折的验证集得分

    # 进行10折交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_trainval)):
        print(f"第 {fold+1} 折交叉验证...")
        Xn_train, Xn_val = X_num_trainval[train_idx], X_num_trainval[val_idx]
        Xc_train, Xc_val = X_cat_trainval[train_idx], X_cat_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        input_num = Input(shape=(Xn_train.shape[1],))
        x_num = Dense(units_num, kernel_regularizer=regularizers.l1_l2(1e-5, l2_reg))(input_num)
        x_num = BatchNormalization()(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num)(x_num)
        x_num = Dense(units_num // 2)(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num_2)(x_num)

        input_cat = Input(shape=(Xc_train.shape[1],))
        x_cat = Dense(units_cat)(input_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat)(x_cat)
        x_cat = Dense(units_cat // 2)(x_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat_2)(x_cat)

        x = Concatenate()([x_num, x_cat])
        x = Dense(fusion_dim, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=[input_num, input_cat], outputs=output)
        model.compile(optimizer=AdamW(learning_rate=learning_rate, weight_decay=l2_reg),
                      loss=tf.keras.losses.Huber(delta=1.0),
                      metrics=['mae', r2_metric])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]

        # 训练模型
        history = model.fit([Xn_train, Xc_train], y_train,
                            validation_data=([Xn_val, Xc_val], y_val),
                            epochs=200,
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=callbacks)

        # 输出每折的训练集和验证集结果
        print(f"第 {fold+1} 折训练集 R²：{history.history['r2_metric'][-1]}")
        print(f"第 {fold+1} 折验证集 R²：{history.history['val_r2_metric'][-1]}")
        print(f"第 {fold+1} 折训练集 MSE：{history.history['loss'][-1]}")
        print(f"第 {fold+1} 折验证集 MSE：{history.history['val_loss'][-1]}")

        # 计算每折的评分
        y_val_pred = model.predict([Xn_val, Xc_val]).flatten()
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        score = 0.3 * mse + 0.7 * (1 - r2)
        val_scores.append(score)

    # 返回目标函数值（负的平均交叉验证得分）
    return -np.mean(val_scores)

# 贝叶斯优化设置
pbounds = {
    'units_num': (32, 128),
    'units_cat': (64, 156),
    'fusion_dim': (32, 128),
    'dropout_num': (0.15, 0.3),
    'dropout_cat': (0.05, 0.1),
    'dropout_num_2': (0.1, 0.25),
    'dropout_cat_2': (0.1, 0.15),
    'learning_rate': (1e-4, 5e-4),     # ✅ 降低学习率范围
    'l2_reg': (1e-5, 9e-5),            # ✅ 增强正则
    'batch_size': (32, 48),           # ✅ 稍微增大 batch size
    'act_choice_idx': (0, len(activation_choices) - 1)
}

optimizer = BayesianOptimization(
    f=build_and_evaluate_model_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=15, n_iter=60)

# 输出最优参数配置
print("\n✅ 最佳超参数配置：")
best = optimizer.max['params']
best['units_num'] = int(best['units_num'])
best['units_cat'] = int(best['units_cat'])
best['fusion_dim'] = int(best['fusion_dim'])
best['batch_size'] = int(best['batch_size'])
best['act_choice_idx'] = int(best['act_choice_idx'])
best['act_choice'] = activation_choices[best['act_choice_idx']]

for k, v in best.items():
    if k == 'act_choice_idx':
        print(f"{k} = {activation_choices[int(v)]} (index={int(v)})")
    else:
        print(f"{k} = {v}")

# 使用整个训练集训练最优模型
act_layer = activation_map[best['act_choice']]

input_num = Input(shape=(X_num.shape[1],))
x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(1e-5, best['l2_reg']))(input_num)
x_num = BatchNormalization()(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num'])(x_num)
x_num = Dense(best['units_num'] // 2)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_2'])(x_num)

input_cat = Input(shape=(X_cat.shape[1],))
x_cat = Dense(best['units_cat'])(input_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat'])(x_cat)
x_cat = Dense(best['units_cat'] // 2)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_2'])(x_cat)

x = Concatenate()([x_num, x_cat])
x = Dense(best['fusion_dim'], activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=[input_num, input_cat], outputs=output)
model.compile(optimizer=AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg']),
              loss=tf.keras.losses.Huber(delta=1.0),
              metrics=['mae', r2_metric])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_num_trainval, X_cat_trainval, y_trainval, test_size=0.2, random_state=42)

history = model.fit([X_num_train, X_cat_train], y_train,
                    validation_data=([X_num_val, X_cat_val], y_val),
                    epochs=300,
                    batch_size=best['batch_size'],
                    verbose=1,
                    callbacks=callbacks)

# 可视化学习曲线
history_dict = history.history
train_r2 = history_dict['r2_metric']
val_r2 = history_dict['val_r2_metric']
train_mse = history_dict['loss']
val_mse = history_dict['val_loss']
epochs = range(1, len(train_r2) + 1)

plt.figure(figsize=(12, 6))
plt.plot(train_r2, label='训练集 R²', color='blue')
plt.plot(val_r2, label='验证集 R²', color='orange')
plt.title('训练集和验证集 R² 随训练次数(epoch)的变化')
plt.xlabel('训练次数(epoch)')
plt.ylabel('R²')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(train_mse, label='训练集 MSE', color='blue')
plt.plot(val_mse, label='验证集 MSE', color='orange')
plt.title('训练集和验证集 MSE 随训练次数(epoch)的变化')
plt.xlabel('训练次数(epoch)')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
common_kwargs = dict(marker='o', linestyle='--', linewidth=1.5, markersize=5)
ax1.set_xlabel('训练次数 (epoch)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(epochs, train_mse, color='tab:blue', label='训练 MSE', **common_kwargs)
l2 = ax1.plot(epochs, val_mse, color='tab:cyan', label='验证 MSE', **common_kwargs)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(epochs, train_r2, color='tab:red', label='训练 R²', **common_kwargs)
l4 = ax2.plot(epochs, val_r2, color='tab:orange', label='验证 R²', **common_kwargs)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.title('训练过程学习曲线（MSE + R²）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# 测试集评估
y_pred_test = model.predict([X_num_test, X_cat_test]).flatten()

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n📊 最终模型评估结果（测试集）：")
print(f"测试集 MSE: {mse_test}")
print(f"测试集 MAE: {mae_test}")
print(f"测试集 R²: {r2_test}")

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

test_result = pd.DataFrame({
    '真实值': y_test,
    '预测值': y_pred_test
})
test_result.to_excel(r'F:/test_pred_vs_true.xlsx', index=False)
print("✅ 已将测试集真实值与预测值保存到 F:/test_pred_vs_true.xlsx")
