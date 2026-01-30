# ✅ 完整代码：包含全部隐藏层 Dropout 贝叶斯调参 + 激活函数调参 + 最优参数输出（显示名称）+ 模型构建 + 可视化 + 导出

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     ReLU, ELU, Concatenate, Activation, LeakyReLU)
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD, Nadam, Adamax, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from bayes_opt import BayesianOptimization
from tensorflow.keras import backend as K

plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据读取与预处理
df = pd.read_csv("Activity_standardized.csv")
numerical_cols = ['Cat(umol)', 'Al/M(molar)', 't/min', 'T/C', 'R1', 'R2']
one_hot_cols = ['M_Zr', 'M_Ti', 'M_Hf', 'R3']
target_col = 'Activity(KgPP/mol cat)'

X_num = df[numerical_cols].values
X_cat = df[one_hot_cols].values
y = df[target_col].values

X_num_train, X_num_temp, X_cat_train, X_cat_temp, y_train, y_temp = train_test_split(
    X_num, X_cat, y, test_size=0.3, random_state=42
)
X_num_val, X_num_test, X_cat_val, X_cat_test, y_val, y_test = train_test_split(
    X_num_temp, X_cat_temp, y_temp, test_size=1/3, random_state=42
)

# 自定义 R² 指标函数
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# 设置随机种子
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(42)

# 激活函数映射表
activation_map = {
    'relu': ReLU(),
    'elu': ELU(),
    'swish': Activation(tf.nn.swish),
    'gelu': Activation(tf.nn.gelu),
    'leaky_relu': LeakyReLU()
}
activation_choices = list(activation_map.keys())

# 贝叶斯调参空间
pbounds = {
    'units_num': (32, 200),
    'units_cat': (32, 128),
    'dropout_num_1': (0.05, 0.1),
    'dropout_num_2': (0.1, 0.2),
    'dropout_num_3': (0.1, 0.2),
    'dropout_num_4': (0.1, 0.3),
    'dropout_cat_1': (0.1, 0.3),
    'dropout_cat_2': (0.05, 0.2),
    'dropout_cat_3': (0.1, 0.2),
    'dropout_cat_4': (0.1, 0.5),
    'learning_rate': (1e-4, 5e-3),
    'l2_reg': (1e-6, 1e-3),
    'batch_size': (32, 64),
    'fusion_dim': (64, 200),
    'act_choice_idx': (0, len(activation_choices) - 1),
    'optimizer_choice_idx': (0, 5)
}

def build_and_evaluate_model(**params):
    units_num = int(params['units_num'])
    units_cat = int(params['units_cat'])
    batch_size = int(params['batch_size'])
    fusion_dim = int(params['fusion_dim'])
    act_choice = activation_choices[int(params['act_choice_idx'])]
    act_layer = activation_map[act_choice]

    optimizer = [
        AdamW(learning_rate=params['learning_rate'], weight_decay=params['l2_reg']),
        RMSprop(learning_rate=params['learning_rate']),
        SGD(learning_rate=params['learning_rate']),
        Nadam(learning_rate=params['learning_rate']),
        Adamax(learning_rate=params['learning_rate']),
        Ftrl(learning_rate=params['learning_rate'])
    ][int(params['optimizer_choice_idx'])]

    input_num = Input(shape=(X_num_train.shape[1],))
    x_num = Dense(units_num, kernel_regularizer=regularizers.l2(params['l2_reg']))(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(params['dropout_num_1'])(x_num)
    x_num = Dense(units_num)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(params['dropout_num_2'])(x_num)
    x_num = Dense(units_num // 2)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(params['dropout_num_3'])(x_num)
    x_num = Dense(units_num // 4)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(params['dropout_num_4'])(x_num)

    input_cat = Input(shape=(X_cat_train.shape[1],))
    x_cat = Dense(units_cat)(input_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(params['dropout_cat_1'])(x_cat)
    x_cat = Dense(units_cat)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(params['dropout_cat_2'])(x_cat)
    x_cat = Dense(units_cat // 2)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(params['dropout_cat_3'])(x_cat)
    x_cat = Dense(units_cat // 4)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(params['dropout_cat_4'])(x_cat)

    x = Concatenate()([x_num, x_cat])
    x = Dense(fusion_dim, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[input_num, input_cat], outputs=output)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.Huber(delta=1.0),
                  metrics=['mae', r2_metric])

    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)

    model.fit([X_num_train, X_cat_train], y_train,
              validation_data=([X_num_val, X_cat_val], y_val),
              epochs=300,
              batch_size=batch_size,
              verbose=0,
              callbacks=[early_stop, reduce_lr])

    y_val_pred = model.predict([X_num_val, X_cat_val]).flatten()
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    return - (0.3 * val_mse + 0.7 * (1 - val_r2))

optimizer = BayesianOptimization(
    f=build_and_evaluate_model,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=10, n_iter=50)

# 输出最优参数（显示激活函数名称）
print("\n🎯 最佳参数如下：")
best_params = optimizer.max['params']
for k, v in best_params.items():
    if k == 'act_choice_idx':
        idx = int(round(v))
        print(f"act_choice: {activation_choices[idx]} (index {idx})")
    else:
        print(f"{k}: {v}")

# 获取最优参数并构建最终模型
best = optimizer.max['params']
for key in ['units_num', 'units_cat', 'batch_size', 'fusion_dim', 'act_choice_idx', 'optimizer_choice_idx']:
    best[key] = int(best[key])
best['act_choice'] = activation_choices[best['act_choice_idx']]
act_layer = activation_map[best['act_choice']]


input_num = Input(shape=(X_num.shape[1],))
x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l2(best['l2_reg']))(input_num)
x_num = BatchNormalization()(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_1'])(x_num)
x_num = Dense(best['units_num'])(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_2'])(x_num)
x_num = Dense(best['units_num'] // 2)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_3'])(x_num)
x_num = Dense(best['units_num'] // 4)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_4'])(x_num)

input_cat = Input(shape=(X_cat.shape[1],))
x_cat = Dense(best['units_cat'])(input_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_1'])(x_cat)
x_cat = Dense(best['units_cat'])(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_2'])(x_cat)
x_cat = Dense(best['units_cat'] // 2)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_3'])(x_cat)
x_cat = Dense(best['units_cat'] // 4)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_4'])(x_cat)

x = Concatenate()([x_num, x_cat])
x = Dense(best['fusion_dim'], activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=[input_num, input_cat], outputs=output)
model.compile(optimizer=AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg']),
              loss=tf.keras.losses.Huber(delta=1.0),
              metrics=['mae', r2_metric])

checkpoint_path = "F:/best_model_checkpoint.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                   monitor='val_loss',
                                   save_best_only=True,
                                   save_weights_only=False,
                                   verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=15, min_lr=1e-6)

print("\n🚀 使用最佳参数开始训练模型（带checkpoint记录）...")
history = model.fit([X_num_train, X_cat_train], y_train,
                    validation_data=([X_num_val, X_cat_val], y_val),
                    epochs=300,
                    batch_size=best['batch_size'],
                    verbose=1,
                    callbacks=[early_stop, reduce_lr, model_checkpoint])

# 模型评估
print("\n📊 最终模型评估结果（验证集）：")
y_pred_val = model.predict([X_num_val, X_cat_val]).flatten()
print(f"MSE: {mean_squared_error(y_val, y_pred_val)}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_val)}")
print(f"R²: {r2_score(y_val, y_pred_val)}")

train_r2 = history.history['r2_metric']
val_r2 = history.history['val_r2_metric']
train_mse = history.history['loss']
val_mse = history.history['val_loss']

# ========== 双纵坐标轴学习曲线（MSE + R²） ==========
epochs = range(1, len(train_r2) + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))

# 左轴：MSE
ax1.set_xlabel('训练次数 (epoch)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(epochs, train_mse, 'o-', label='训练 MSE', color='tab:blue')
l2 = ax1.plot(epochs, val_mse, 's-', label='验证 MSE', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右轴：R²
ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(epochs, train_r2, 'o--', label='训练 R²', color='tab:red')
l4 = ax2.plot(epochs, val_r2, 's--', label='验证 R²', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))

# 合并图例
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)

plt.title('训练过程学习曲线（MSE + R²）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# 测试集预测与评估
y_pred_test = model.predict([X_num_test, X_cat_test]).flatten()
df_result = pd.DataFrame({'真实值': y_test, '预测值': y_pred_test})
df_result.to_csv("F:/test_predictions.csv", index=False, encoding='utf-8-sig')
print("\n✅ 测试集预测结果已保存至 F:/test_predictions.csv")

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

print("\n📊 最终模型评估结果（测试集）：")
print(f"测试集 MSE: {mean_squared_error(y_test, y_pred_test)}")
print(f"测试集 MAE: {mean_absolute_error(y_test, y_pred_test)}")
print(f"测试集 R²: {r2_score(y_test, y_pred_test)}")
