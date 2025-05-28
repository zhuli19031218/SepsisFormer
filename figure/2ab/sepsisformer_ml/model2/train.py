# -*- encoding:utf-8 -*-
import argparse
import math
import os
import time # 导入 time 模块
import torch.cuda
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, accuracy_score
import copy
import numpy as np
import pandas as pd

# 导入模型 - 假设它们在同一目录或可以通过 PYTHONPATH 访问
from Transformer import Transformer
from model_LSTM import Lstm
from model_GRU import GRU
from model_gpt import GPT

from dataloader import ay_dataloader # !!! 必须返回 DataLoader 对象 !!!
from utilis_file import mkdir, get_dir_name# 确保 NMTCritierion 定义或导入
from utilis_data import NMTCritierion # 确保 NMTCritierion 定义或导入
from torch.utils.tensorboard import SummaryWriter
from thop import clever_format, profile
# from torchsummary import summary # 如果需要可以取消注释
from draw_roc import draw_roc # 导入 ROC 绘制函数


class Train(object):
    def __init__(self, args):
        super(Train, self).__init__()
        # 所有参数
        self.args = args
        # 各种指标和最佳模型记录 (state_dict)
        self.pre_best_auc_model_state_dict, self.pre_best_acc_model_state_dict = None, None # 存储 state_dict
        self.pre_best_auc, self.pre_best_auc_acc, self.pre_best_acc, self.pre_best_acc_auc = \
            torch.tensor([0.], dtype=torch.float32), \
            torch.tensor([0.], dtype=torch.float32), \
            torch.tensor([0.], dtype=torch.float32), \
            torch.tensor([0.], dtype=torch.float32)

        # 文件夹名：使用指定的日志基础路径
        self.base_log_path = "/mnt/public/home/zhijiangwan/sepsisformer/sepsisformer/model2/logs/"
        # 构造日志目录名，包含关键超参数和时间戳
        current_time = time.strftime("%Y%m%d-%H%M%S") # 获取当前时间戳字符串
        # 修正: 使用时间戳替换 get_dir_name()
        self.dir_name = f"{self.args.model_name}_{self.args.logs_first}_factors{self.args.factors}" \
                        f"_lr{args.lr}_bs{args.batch_size}_epoch{args.epoch}_{current_time}"
        # 创建Logs文件夹
        self.logs_path = mkdir(dir_name=os.path.join(self.base_log_path, self.dir_name))
        print(f"日志将保存在: {self.logs_path}")

        # 获取数据集与标签 (!!! 重要: ay_dataloader 必须能处理 batch_size 并返回 DataLoader !!!)
        print(f"从 {args.data_path} 加载数据...")
        try:
            self.pre_train_dataset, self.pre_test_dataset, self.pre_total_train_len, self.pre_total_test_len = \
                ay_dataloader(path=args.data_path, batch_size=args.batch_size, logs=self.logs_path)
            print(f"训练样本数: {self.pre_total_train_len}, 测试样本数: {self.pre_total_test_len}")
            # 检查 dataloader 是否返回了预期的批次迭代器
            if not hasattr(self.pre_train_dataset, '__iter__') or not hasattr(self.pre_test_dataset, '__iter__'):
                 print("\n!!! 警告: Dataloader 可能未正确配置为返回批次迭代器。请检查 ay_dataloader 实现。脚本可能无法正常运行。 !!!\n")
            # 简单测试迭代器（可选）
            # next(iter(self.pre_train_dataset))
            # next(iter(self.pre_test_dataset))
            # print("Dataloader 迭代器初步检查通过。")
        except Exception as e:
            print(f"\n!!! 错误: 加载数据时出错: {e} !!!")
            print("请确保 dataloader.py 中的 ay_dataloader 函数正确实现了基于 torch.utils.data.Dataset 和 torch.utils.data.DataLoader 的批次加载。")
            raise # 抛出异常，终止程序

        # 模型选择与初始化
        print(f"初始化模型: {self.args.model_name}")
        try:
            if self.args.model_name == "Transformer":
                # --- 使用正确的参数名 ---
                self.pre_model = Transformer(
                    input_dim=self.args.factors,    # 使用 input_dim 传递特征数
                    model_dim=128,                  # 指定 Transformer 内部维度 (可以设为参数)
                    depth=8,                        # 可以设为参数
                    num_heads=8,                    # 可以设为参数
                    attn_drop_ratio=0.1,            # 注意力 dropout
                    drop_ratio=self.args.drop_ratio, # MLP dropout
                    drop_path_ratio=0.1             # 随机深度 (可以设为参数)
                ).to(self.args.device)
            elif self.args.model_name == "Lstm":
                self.pre_model = Lstm(factors=8,
                                      batch_size=self.args.batch_size,
                                      drop_ratio=self.args.drop_ratio,
                                      device=self.args.device # 如果 Lstm 内部需要
                                      ).to(self.args.device)
            elif self.args.model_name == "GRU":
                self.pre_model = GRU(factors=self.args.factors,
                                     batch_size=self.args.batch_size,
                                     num_layers=self.args.num_layers,
                                     drop_ratio=self.args.drop_ratio,
                                     device=self.args.device # 如果 GRU 内部需要
                                     ).to(self.args.device)
            elif self.args.model_name == "GPT":
                 self.pre_model = GPT(hidden_dim=self.args.factors, num_heads=8,
                                     num_layers=8,
                                     dropout=self.args.drop_ratio).to(self.args.device)
                 print("警告: 请再次确认 GPT 模型的输出层适用于二分类任务 (输出维度应为 2)。")
            else:
                raise ValueError(f"无法识别的模型名称: '{self.args.model_name}'")
            print(f"{self.args.model_name} 模型已加载到 {self.args.device}")
        except Exception as e:
            print(f"\n!!! 错误: 初始化模型 {self.args.model_name} 时出错: {e} !!!")
            
            raise
            # **************************************************
            # ********    添加加载预训练权重的逻辑    *********
            # **************************************************
        if args.pretrain:
                if args.loadmodel and os.path.exists(args.loadmodel):
                    print(f"\n加载预训练权重从: {args.loadmodel} (不加载 MLP 头)")
                    try:
                        # 加载预训练模型的 state_dict，并映射到当前设备
                        save_model_state_dict = torch.load(args.loadmodel, map_location=args.device)
                        # 有些 .pth 文件可能直接存的是 state_dict，有些可能存了一个字典包含 state_dict
                        if isinstance(save_model_state_dict, dict) and 'state_dict' in save_model_state_dict:
                            save_model_state_dict = save_model_state_dict['state_dict']
                        elif not isinstance(save_model_state_dict, dict):
                             raise TypeError("加载的模型文件不是有效的 state_dict 或包含 state_dict 的字典。")


                        model_dict = self.pre_model.state_dict()  # 获取当前模型的 state_dict
                        # 创建一个新字典，只包含需要加载的预训练权重
                        # 注意：这里的 'MLP' 筛选依赖于你的分类头模块名称是 'MLP'
                        # 如果你的分类头名称不同，需要修改下面的字符串 'MLP'
                        state_dict_to_load = {k: v for k, v in save_model_state_dict.items() if not 'MLP' in k}
                        print(f"预训练模型中将被加载的层数: {len(state_dict_to_load)}")
                        print(f"当前模型总层数: {len(model_dict)}")

                        # 检查加载的权重与当前模型对应层形状是否匹配
                        loaded_keys = set(state_dict_to_load.keys())
                        model_keys = set(model_dict.keys())
                        mismatched_shapes = []
                        matched_keys_count = 0

                        for k in loaded_keys:
                             if k in model_keys:
                                 if state_dict_to_load[k].shape == model_dict[k].shape:
                                     matched_keys_count += 1
                                 else:
                                     mismatched_shapes.append(f"  - 层 '{k}': 预训练形状 {state_dict_to_load[k].shape}, 当前模型形状 {model_dict[k].shape}")
                        print(f"预训练模型中与当前模型名称和形状都匹配的层数: {matched_keys_count}")
                        if mismatched_shapes:
                             print("\n!!! 警告: 检测到形状不匹配的层，这些层的权重将不会被加载: !!!")
                             for msg in mismatched_shapes:
                                 print(msg)
                             # 从待加载字典中移除形状不匹配的层
                             state_dict_to_load = {k: v for k, v in state_dict_to_load.items() if k not in [m.split("'")[1] for m in mismatched_shapes]}
                             print(f"移除形状不匹配层后，实际加载的层数: {len(state_dict_to_load)}")


                        # 更新当前模型的 state_dict
                        # model_dict 中的权重会被 state_dict_to_load 中对应键的值覆盖
                        model_dict.update(state_dict_to_load)

                        # 加载更新后的 state_dict 回模型
                        # strict=True (默认) 会检查 model_dict 的所有键是否都在加载的字典中（这里是更新后的 model_dict，所以总是匹配的）
                        # 但它也会检查加载的字典（更新后的 model_dict）的键是否都在模型自身的 keys() 中，这也是匹配的。
                        # 如果因为某些原因（例如上面移除了不匹配的层）导致加载不完全匹配，可能需要 strict=False，但这会隐藏问题。
                        # 推荐先用 strict=True 尝试。
                        self.pre_model.load_state_dict(model_dict, strict=True)
                        print("预训练权重加载完成 (MLP 头保持初始状态)。")

                    except FileNotFoundError:
                        print(f"\n!!! 错误: 预训练模型文件未找到: {args.loadmodel} !!!")
                        # 可以选择退出或继续（不加载预训练）
                        # exit(1)
                    except Exception as e:
                        print(f"\n!!! 错误: 加载预训练权重时出错: {e} !!!")
                        # 可以选择退出或继续
                        # exit(1)
                else:
                     # args.pretrain 为 True 但 args.loadmodel 无效
                    print(f"\n!!! 警告: 设置了 --pretrain 但提供的路径无效或文件不存在: '{args.loadmodel}'。将从头开始训练。 !!!\n")
            # **************************************************
            # ********    预训练权重加载逻辑结束    *********
            # **************************************************
        # 定义优化器和损失函数
        self.pre_optimizer = torch.optim.Adam(self.pre_model.parameters(), lr=args.lr, weight_decay=0)
        print(f"优化器: Adam, 学习率: {args.lr}, 权重衰减: 5e-4")

        #self.pre_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.pre_optimizer,
        #                                                                   T_max=args.epoch + 100,
        #                                                                   eta_min=1e-6, last_epoch=-1)
       # print("学习率调度器: CosineAnnealingLR")

        # 损失函数
        label_smoothing_value = 0.3
        try:
            self.pre_loss_fn = NMTCritierion(label_smoothing=label_smoothing_value).to(self.args.device)
            print(f"损失函数: NMTCritierion (标签平滑: {label_smoothing_value})")
        except NameError:
             print("警告: NMTCritierion 未定义，将使用 nn.CrossEntropyLoss。")
             self.pre_loss_fn = nn.CrossEntropyLoss().to(self.args.device)
             print(f"损失函数: nn.CrossEntropyLoss")


        # Target 字典初始化，用于存储结果
        self.target = {
            'model': self.args.model_name, 'data_path': self.args.data_path, 'log_path': self.logs_path,
            'lr': self.args.lr, 'batch_size': self.args.batch_size, 'epoch': self.args.epoch,
            'factors': self.args.factors, 'drop_ratio': self.args.drop_ratio, 'num_layers': self.args.num_layers,
            'best_acc': 0.0, 'best_acc_auc': 0.0, 'best_auc': 0.0, 'best_auc_acc': 0.0, 'best_acc_epoch': 0,
            'test_acc_on_best_acc_model': 0.0, 'test_auc_on_best_acc_model': 0.0,
            'tp_acc': 0, 'fp_acc': 0, 'tn_acc': 0, 'fn_acc': 0,
            'precision_acc': 0.0, 'recall_acc': 0.0, 'specificity_acc': 0.0,
            'f1_acc': 0.0, 'mcc_acc': 0.0,
            'test_acc_on_best_auc_model': 0.0, 'test_auc_on_best_auc_model': 0.0,
            'tp_auc': 0, 'fp_auc': 0, 'tn_auc': 0, 'fn_auc': 0,
            'precision_auc': 0.0, 'recall_auc': 0.0, 'specificity_auc': 0.0,
            'f1_auc': 0.0, 'mcc_auc': 0.0,
            }

        # TensorBoard 记录器
        self.write = SummaryWriter(log_dir=self.logs_path)
        print(f"TensorBoard 日志写入器已初始化, 日志目录: {self.logs_path}")

        # 打印用于启动 TensorBoard 的命令
        tensorboard_cmd = f"tensorboard --logdir=\"{self.logs_path}\""
        print(f"启动 TensorBoard: {tensorboard_cmd}")
        try:
            with open("tensorCmd.txt", mode='a') as f:
                f.write(tensorboard_cmd + '\n')
        except Exception as e:
            print(f"警告: 无法写入 tensorCmd.txt: {e}")

    def _prepare_input(self, data_batch):
        """准备模型输入：添加时间步维度，并在需要时进行维度转换。"""
        try:
            input_data = data_batch['data'].to(self.args.device, non_blocking=True)
            target_data = data_batch['target'].to(self.args.device, non_blocking=True)

            # 添加时间步维度: [batch_size, 1, features]
            input_data = torch.unsqueeze(input=input_data, dim=1)

            # 如果是 LSTM 或 GRU，可能需要将维度转换为 [seq_len, batch_size, features]
            if self.args.model_name in ["Lstm", "GRU"]:
                input_data = input_data.permute(1, 0, 2) # [1, B, F]

            return input_data, target_data
        except KeyError as e:
            print(f"\n!!! 错误: Dataloader 返回的批次数据中缺少键 '{e}'。请确保批次是包含 'data' 和 'target' 的字典。 !!!\n")
            raise
        except Exception as e:
            print(f"\n!!! 错误: 准备输入时出错: {e} !!!")
            raise


    def train(self):
        """执行模型训练和评估循环"""
        pre_best_acc_epoch = 0
        print(f"\n开始训练 {self.args.epoch} 个轮次...")
        for i in range(self.args.epoch):
            epoch_start_time = time.time()
            total_train_loss, total_train_correct = torch.tensor([0.], dtype=torch.float32), \
                                                      torch.tensor([0.], dtype=torch.float32)
            total_test_loss, total_test_correct = torch.tensor([0.], dtype=torch.float32), \
                                                     torch.tensor([0.], dtype=torch.float32)
            all_test_outputs, all_test_labels = torch.Tensor(), torch.Tensor()

            # --- 训练阶段 ---
            self.pre_model.train()
            processed_train_samples = 0
            num_train_batches = 0 # 初始化批次数
            try:
                 num_train_batches = len(self.pre_train_dataset) # 获取训练批次总数
                 if num_train_batches == 0:
                      print("\n!!! 错误: 训练数据加载器为空，无法进行训练。请检查数据路径和 dataloader 实现。 !!!\n")
                      return # 提前退出训练
            except TypeError: # 如果 len() 不适用于 dataloader
                 print("\n警告: 无法获取训练数据加载器的长度。平均损失将按批次计算，准确率可能不精确。建议修复 dataloader 使其具有长度。\n")
                 num_train_batches = -1 # 标记长度未知

            for batch_idx, pre_train_batch in enumerate(self.pre_train_dataset):
                pre_train_input, pre_train_target = self._prepare_input(pre_train_batch)
                pre_sub_train_output = self.pre_model(pre_train_input)
                pre_sub_train_loss = self.pre_loss_fn(pre_sub_train_output, pre_train_target)

                self.pre_optimizer.zero_grad()
                pre_sub_train_loss.backward()
                self.pre_optimizer.step()

                total_train_loss += pre_sub_train_loss.cpu().detach()
                batch_correct = (pre_sub_train_output.detach().argmax(axis=1) == pre_train_target).sum()
                total_train_correct += batch_correct.cpu()
                processed_train_samples += pre_train_target.size(0)

            # 更新学习率
            #self.pre_lr_scheduler.step()

            # 计算并记录平均训练损失和准确率
            avg_train_loss = total_train_loss / (num_train_batches if num_train_batches > 0 else (batch_idx + 1)) # 防止除零
            avg_train_acc = total_train_correct / processed_train_samples if processed_train_samples > 0 else torch.tensor(0.)
            self.write.add_scalar('Loss/Train', avg_train_loss.item(), i)
            self.write.add_scalar('Accuracy/Train', avg_train_acc.item(), i)
            self.write.add_scalar('LearningRate', self.pre_optimizer.param_groups[0]['lr'], i)


            # --- 评估阶段 (在测试集上) ---
            self.pre_model.eval()
            processed_test_samples = 0
            num_test_batches = 0 # 初始化批次数
            try:
                 num_test_batches = len(self.pre_test_dataset) # 获取测试批次总数
                 if num_test_batches == 0:
                      print("\n!!! 错误: 测试数据加载器为空，无法进行评估。请检查数据路径和 dataloader 实现。 !!!\n")
                      # 可能需要决定是否继续训练或停止
                      continue # 跳过本轮评估
            except TypeError:
                 print("\n警告: 无法获取测试数据加载器的长度。平均损失将按批次计算，准确率可能不精确。建议修复 dataloader 使其具有长度。\n")
                 num_test_batches = -1 # 标记长度未知

            with torch.no_grad():
                for batch_idx_test, pre_test_batch in enumerate(self.pre_test_dataset):
                    pre_test_input, pre_test_target = self._prepare_input(pre_test_batch)
                    pre_sub_test_output = self.pre_model(pre_test_input)
                    pre_sub_test_loss = self.pre_loss_fn(pre_sub_test_output, pre_test_target)

                    total_test_loss += pre_sub_test_loss.cpu()
                    batch_correct = (pre_sub_test_output.cpu().argmax(axis=1) == pre_test_target.cpu()).sum()
                    total_test_correct += batch_correct
                    processed_test_samples += pre_test_target.size(0)

                    all_test_labels = torch.cat([all_test_labels, pre_test_target.cpu()])
                    all_test_outputs = torch.cat([all_test_outputs, pre_sub_test_output.cpu()])

            if processed_test_samples != self.pre_total_test_len and self.pre_total_test_len > 0:
                 print(f"警告 (Epoch {i+1}): 处理的测试样本数 ({processed_test_samples}) != 总测试样本数 ({self.pre_total_test_len})")

            # 计算平均测试损失和准确率
            avg_test_loss = total_test_loss / (num_test_batches if num_test_batches > 0 else (batch_idx_test + 1))
            avg_test_acc = total_test_correct / processed_test_samples if processed_test_samples > 0 else torch.tensor(0.)

            # 计算 AUC
            current_epoch_auc = 0.0
            if processed_test_samples > 0: # 只有在处理了样本后才计算 AUC
                try:
                    labels_np = all_test_labels.numpy()
                    outputs_prob_np = all_test_outputs[:, 1].numpy()
                    if len(np.unique(labels_np)) > 1:
                        current_epoch_auc = roc_auc_score(labels_np, outputs_prob_np)
                    else:
                        # print(f"警告 (Epoch {i+1}): 测试集只包含一个类别 ({np.unique(labels_np)}), 无法计算 AUC。")
                        pass # 不打印，可能每个 epoch 都发生
                except Exception as e:
                    print(f"警告 (Epoch {i+1}): 计算 AUC 时出错: {e}。")

            # 记录测试指标到 TensorBoard
            self.write.add_scalar('Loss/Test', avg_test_loss.item(), i)
            self.write.add_scalar('Accuracy/Test', avg_test_acc.item(), i)
            self.write.add_scalar('AUC/Test', current_epoch_auc, i)

            # --- 模型保存逻辑 (修正后) ---
            # 检查并更新最佳 AUC 模型
            if current_epoch_auc > self.pre_best_auc:
                #print(f"Epoch {i+1}: 新的最佳 AUC = {current_epoch_auc:.4f} (旧 = {self.pre_best_auc:.4f}). 保存模型...")
                self.pre_best_auc_model_state_dict = copy.deepcopy(self.pre_model.state_dict()) # 保存 state_dict
                self.pre_best_auc_acc = avg_test_acc
                self.pre_best_auc = current_epoch_auc
                try:
                    torch.save(self.pre_best_auc_model_state_dict, os.path.join(self.logs_path, 'best_auc_model.pth'))
                except Exception as e:
                    print(f"错误: 保存 best_auc_model.pth 失败: {e}")


            # 检查并更新最佳准确率模型
            if avg_test_acc > self.pre_best_acc:
                #print(f"Epoch {i+1}: 新的最佳 Accuracy = {avg_test_acc:.4f} (旧 = {self.pre_best_acc:.4f}). 保存模型...")
                self.pre_best_acc_model_state_dict = copy.deepcopy(self.pre_model.state_dict()) # 保存 state_dict
                self.pre_best_acc = avg_test_acc
                self.pre_best_acc_auc = current_epoch_auc
                pre_best_acc_epoch = i + 1
                self.target['best_acc_epoch'] = pre_best_acc_epoch
                try:
                    torch.save(self.pre_best_acc_model_state_dict, os.path.join(self.logs_path, 'best_acc_model.pth'))
                except Exception as e:
                    print(f"错误: 保存 best_acc_model.pth 失败: {e}")


            # --- 打印 Epoch 摘要 ---
            if (i + 1) % 20 == 0: # 每个 epoch 打印一次
                 print(f"Epoch: {i + 1:04d}/{self.args.epoch} | "
                      f"Train Loss: {avg_train_loss.item():.4f} | "
                      f"Train Acc: {avg_train_acc.item():.4f} | "
                      f"Test Loss: {avg_test_loss.item():.4f} | "
                      f"Test Acc: {avg_test_acc.item():.4f} | "
                      f"Test AUC: {current_epoch_auc:.4f} | "
                      f"LR: {self.pre_optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {time.time() - epoch_start_time:.2f}s")

        # --- 训练结束 ---
        print("\n训练完成。")
        # 将最终的最佳指标存入 target 字典
        self.target['best_acc'] = self.pre_best_acc.item() if isinstance(self.pre_best_acc, torch.Tensor) else self.pre_best_acc
        self.target['best_acc_auc'] = self.pre_best_acc_auc
        self.target['best_auc'] = self.pre_best_auc
        self.target['best_auc_acc'] = self.pre_best_auc_acc.item() if isinstance(self.pre_best_auc_acc, torch.Tensor) else self.pre_best_auc_acc

        # 保存训练摘要到文件
        summary_file_path = os.path.join(self.logs_path, "training_summary.txt")
        try:
            with open(summary_file_path, mode="w", encoding='utf-8') as s:
                s.write("--- 训练摘要 ---\n")
                for key, value in self.target.items():
                     # 只写入基础配置和训练过程中的最佳指标
                     if key not in ['tp_acc', 'fp_acc', 'tn_acc', 'fn_acc', 'precision_acc', 'recall_acc', 'specificity_acc', 'f1_acc', 'mcc_acc',
                                   'tp_auc', 'fp_auc', 'tn_auc', 'fn_auc', 'precision_auc', 'recall_auc', 'specificity_auc', 'f1_auc', 'mcc_auc',
                                   'test_acc_on_best_acc_model','test_auc_on_best_acc_model',
                                   'test_acc_on_best_auc_model','test_auc_on_best_auc_model']:
                        s.write(f"{key}: {value}\n" if not isinstance(value, float) else f"{key}: {value:.4f}\n")
            print(f"训练摘要已保存至: {summary_file_path}")
        except Exception as e:
            print(f"错误: 无法保存训练摘要文件: {e}")

        self.write.close()
        print("TensorBoard 写入器已关闭。")


    def evaluate_model(self, model_state_dict, model_tag):
        """在测试集上评估给定的模型状态字典，计算详细指标，并保存 ROC 曲线。"""
        print(f"\n开始评估 {model_tag} 模型...".center(100, '='))

        if model_state_dict is None:
            print(f"{model_tag} 模型的状态字典不可用。跳过评估。")
            # 初始化相关指标为 0 或 NaN，避免后续保存出错
            metric_suffix = '_acc' if model_tag == 'BestAcc' else '_auc'
            self.target[f'test_acc_on_{model_tag.lower()}_model'] = 0.0
            self.target[f'test_auc_on_{model_tag.lower()}_model'] = 0.0
            # ... 可以为其他指标也设置默认值 ...
            return

        # 重新加载模型结构
        print(f"为评估重新加载模型结构: {self.args.model_name}")
        try:
            if self.args.model_name == "Transformer":
                # --- 使用与 __init__ 中一致的参数实例化 ---
                eval_model = Transformer(
                    input_dim=self.args.factors,
                    model_dim=128,  # 使用与 __init__ 中相同的 model_dim
                    depth=8,
                    num_heads=8,
                    attn_drop_ratio=0.1,
                    drop_ratio=self.args.drop_ratio,
                    drop_path_ratio=0.1 # 使用与 __init__ 中相同的 drop_path_ratio
                )
            elif self.args.model_name == "Lstm":
                eval_model = Lstm(factors=self.args.factors, drop_ratio=self.args.drop_ratio, device=self.args.device)
            elif self.args.model_name == "GRU":
                eval_model = GRU(factors=self.args.factors, num_layers=self.args.num_layers, drop_ratio=self.args.drop_ratio, device=self.args.device)
            elif self.args.model_name == "GPT":
                 eval_model = GPT(hidden_dim=self.args.factors, num_heads=8, num_layers=self.args.num_layers, dropout=self.args.drop_ratio)
            else:
                 print(f"错误: 无法识别的模型名称 '{self.args.model_name}' 用于评估。")
                 return
        except Exception as e:
            print(f"错误: 评估时重新实例化模型 {self.args.model_name} 失败: {e}")
            return

        # 载入状态字典
        try:
            eval_model.load_state_dict(model_state_dict)
            print(f"{model_tag} 模型状态字典加载成功。")
        except Exception as e:
            print(f"错误: 加载 {model_tag} 模型状态字典失败: {e}")
            return

        eval_model.to(self.args.device)
        eval_model.eval()

        # --- 在测试集上进行评估 ---
        all_test_outputs = torch.Tensor()
        all_test_labels = torch.Tensor()
        processed_test_samples = 0

        with torch.no_grad():
             # 检查测试数据集是否可迭代
            if not hasattr(self.pre_test_dataset, '__iter__'):
                print("\n!!! 错误: 测试数据加载器不可迭代，无法进行评估。 !!!\n")
                return

            for test_batch in self.pre_test_dataset:
                try:
                    test_input, test_target = self._prepare_input(test_batch)
                    sub_test_output = eval_model(test_input).cpu()

                    all_test_labels = torch.cat([all_test_labels, test_target.cpu()])
                    all_test_outputs = torch.cat([all_test_outputs, sub_test_output])
                    processed_test_samples += test_target.size(0)
                except Exception as e:
                    print(f"错误: 在评估 {model_tag} 模型时处理批次数据出错: {e}")
                    # 可以选择跳过此批次或终止评估
                    continue # 跳过当前批次

        if processed_test_samples == 0:
             print(f"\n!!! 错误: 未能成功处理任何测试样本进行 {model_tag} 评估。 !!!\n")
             return # 无法计算指标

        if processed_test_samples != self.pre_total_test_len and self.pre_total_test_len > 0:
            print(f"警告 ({model_tag} 评估): 处理的测试样本数 ({processed_test_samples}) != 总测试样本数 ({self.pre_total_test_len})")

        # --- 计算测试集指标 ---
        test_labels_np = all_test_labels.numpy()
        test_preds_np = all_test_outputs.argmax(axis=1).numpy()
        test_probs_np = torch.softmax(all_test_outputs, dim=1)[:, 1].numpy() # 确保使用 softmax 后的概率

        test_acc = accuracy_score(test_labels_np, test_preds_np)
        test_f1 = f1_score(test_labels_np, test_preds_np, average='binary', zero_division=0)
        test_mcc = matthews_corrcoef(test_labels_np, test_preds_np)
        test_auc = 0.0
        try:
            if len(np.unique(test_labels_np)) > 1:
                test_auc = roc_auc_score(test_labels_np, test_probs_np)
            else:
                print(f"警告 ({model_tag} 评估): 测试集只包含一个类别 ({np.unique(test_labels_np)}), AUC 设为 0.0。")
        except Exception as e:
            print(f"警告 ({model_tag} 评估): 计算 AUC 时出错: {e}")

        try:
            test_confusion = confusion_matrix(test_labels_np, test_preds_np, labels=[0, 1]) # 确保标签顺序
            if test_confusion.shape == (2, 2):
                tn, fp, fn, tp = test_confusion.ravel()
            else: # 处理非 2x2 情况
                # 尝试计算，如果只有一类预测正确
                tn, fp, fn, tp = 0, 0, 0, 0
                if test_confusion.shape == (1,1):
                     unique_label = np.unique(test_labels_np)[0]
                     if unique_label == 0 and test_preds_np[0] == 0: tn = test_confusion[0,0]
                     elif unique_label == 1 and test_preds_np[0] == 1: tp = test_confusion[0,0]
                print(f"警告 ({model_tag} 评估): 混淆矩阵形状异常: {test_confusion.shape}。 TP/FP/TN/FN 可能不准确。")

        except Exception as e:
             print(f"错误 ({model_tag} 评估): 计算混淆矩阵时出错: {e}")
             tn, fp, fn, tp = 0, 0, 0, 0


        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # --- 保存测试集指标到 self.target ---
        metric_suffix = '_acc' if model_tag == 'BestAcc' else '_auc'
        self.target[f'test_acc_on_{model_tag.lower()}_model'] = test_acc
        self.target[f'test_auc_on_{model_tag.lower()}_model'] = test_auc
        self.target[f'tp{metric_suffix}'] = int(tp)
        self.target[f'fp{metric_suffix}'] = int(fp)
        self.target[f'tn{metric_suffix}'] = int(tn)
        self.target[f'fn{metric_suffix}'] = int(fn)
        self.target[f'precision{metric_suffix}'] = precision
        self.target[f'recall{metric_suffix}'] = recall
        self.target[f'specificity{metric_suffix}'] = specificity
        self.target[f'f1{metric_suffix}'] = test_f1
        self.target[f'mcc{metric_suffix}'] = test_mcc

        # 将详细指标保存到文本文件
        metrics_file_path = os.path.join(self.logs_path, f"{model_tag}_test_metrics.txt")
        try:
            with open(metrics_file_path, mode="w", encoding='utf-8') as s:
                s.write(f"--- 测试集指标 ({model_tag} 模型) ---\n")
                s.write(f"准确率 (Accuracy):     {test_acc:.4f}\n")
                s.write(f"AUC:                  {test_auc:.4f}\n")
                s.write(f"F1 分数 (F1 Score):   {test_f1:.4f}\n")
                s.write(f"马修斯相关系数 (MCC): {test_mcc:.4f}\n")
                s.write("-" * 20 + "\n")
                s.write("混淆矩阵 (Confusion Matrix - [[TN, FP], [FN, TP]]):\n")
                s.write(f"{test_confusion}\n")
                s.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
                s.write("-" * 20 + "\n")
                s.write(f"精确率 (Precision/PPV):    {precision:.4f}\n")
                s.write(f"召回率 (Recall/Sensitivity): {recall:.4f}\n")
                s.write(f"特异度 (Specificity/TNR):  {specificity:.4f}\n")
            print(f"{model_tag} 模型的测试指标已保存至: {metrics_file_path}")
        except Exception as e:
            print(f"错误: 无法保存 {model_tag} 的指标文件: {e}")

        # --- 绘制并保存 ROC 曲线 ---
        roc_save_path = os.path.join(self.logs_path, f'{model_tag}_test_roc_curve.png')
        try:
            if len(np.unique(test_labels_np)) > 1 and len(test_probs_np) > 1:
                 draw_roc(test_labels_np, test_probs_np, f'{self.args.model_name} ({model_tag} - Test)', roc_save_path)
                 print(f"{model_tag} 模型的测试集 ROC 曲线已保存至: {roc_save_path}")
            else:
                 print(f"无法绘制 {model_tag} 的 ROC 曲线：类别不足或数据点不足。")
        except ImportError:
             print("错误: 无法导入 matplotlib 或 sklearn.metrics 进行 ROC 绘制。请确保已安装这些库。")
        except Exception as e:
            print(f"错误: 绘制或保存 {model_tag} 测试集 ROC 曲线时出错: {e}")


    def model_summary(self):
        """计算并打印模型的参数量和计算量 (FLOPs)"""
        if hasattr(self, 'pre_model') and self.pre_model is not None:
            try:
                from thop import profile, clever_format # 局部导入
                print("\n计算模型摘要 (参数量, FLOPs)...")

                # 在 CPU 上重新实例化模型以进行计算
                if self.args.model_name == "Transformer":
                     summary_model = Transformer(
                        input_dim=self.args.factors,
                        model_dim=128,  # 使用与 __init__ 中相同的 model_dim
                        depth=8,
                        num_heads=8,
                        attn_drop_ratio=0.1,
                        drop_ratio=self.args.drop_ratio,
                        drop_path_ratio=0.1 # 使用与 __init__ 中相同的 drop_path_ratio
                    ).to('cpu')
                elif self.args.model_name == "Lstm":
                    summary_model = Lstm(factors=self.args.factors, drop_ratio=self.args.drop_ratio, device='cpu').to('cpu')
                elif self.args.model_name == "GRU":
                    summary_model = GRU(factors=self.args.factors, num_layers=self.args.num_layers, drop_ratio=self.args.drop_ratio, device='cpu').to('cpu')
                elif self.args.model_name == "GPT":
                    summary_model = GPT(hidden_dim=self.args.factors, num_heads=8, num_layers=self.args.num_layers, dropout=self.args.drop_ratio).to('cpu')
                else:
                    print("模型摘要计算跳过：未知模型类型。")
                    return

                summary_model.eval()

                # 构造虚拟输入
                if self.args.model_name in ["Lstm", "GRU"]:
                     dummy_input = torch.randn(1, 1, self.args.factors).to('cpu') # [Seq, Batch, Feature] = [1, 1, F]
                else:
                     dummy_input = torch.randn(1, 1, self.args.factors).to('cpu') # [Batch, Seq, Feature] = [1, 1, F]

                macs, params = profile(summary_model, inputs=(dummy_input,), verbose=False)
                flops = macs * 2
                flops, params = clever_format([flops, params], "%.3f")
                print(f'总 FLOPs (估算): {flops}')
                print(f'总参数量: {params}')

                summary_file_path = os.path.join(self.logs_path, "training_summary.txt")
                try:
                    with open(summary_file_path, mode="a", encoding='utf-8') as s:
                        s.write(f"\n--- 模型复杂度 ({self.args.model_name}) ---\n")
                        s.write(f"总 FLOPs (thop 估算, MACs*2): {flops}\n")
                        s.write(f"总参数量 (thop): {params}\n")
                    print(f"模型复杂度信息已附加到: {summary_file_path}")
                except Exception as e:
                    print(f"警告: 无法附加模型复杂度信息: {e}")

            except ImportError:
                 print("警告: 无法导入 'thop'。跳过模型摘要计算。请运行 'pip install thop'")
            except Exception as e:
                print(f"计算模型摘要时出错: {e}")
        else:
             print("模型摘要计算跳过：模型未初始化。")