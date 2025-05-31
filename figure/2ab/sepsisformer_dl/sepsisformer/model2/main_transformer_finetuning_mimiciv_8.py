import os
import pandas as pd
import torch
import argparse
import numpy as np # 导入 numpy 用于设置种子
# from torch import nn # 不再直接需要
# from utilis_file import write_excel # 移除了 Excel 写入逻辑
from train import Train # 导入训练类

def hyperParameters():
    """定义和解析所有命令行参数。"""
    parser = argparse.ArgumentParser(description="在 MIMIC/eICU 数据上训练序列模型")

    # --- 硬件设置 ---
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='指定运行设备 (cuda 或 cpu)')



    parser.add_argument('--model_dim', type=int, default=128, help='Transformer internal dimension.')
    parser.add_argument('--depth', type=int, default=8, help='Transformer depth.')
    parser.add_argument('--num_heads', type=int, default=8, help='Transformer number of heads.')
    parser.add_argument('--drop_path_ratio', type=float, default=0.1, help='Stochastic depth rate.')

    # --- 模型超参数 ---
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='MLP 或 head 中的 Dropout 比率。')
    parser.add_argument('--num_layers', type=int, default=2,
                        help="模型的层数 (主要用于 GRU, GPT)。")
    
    
    
    
    
    
    
    
    # --- 模型选择 ---
    parser.add_argument('--model_name', type=str, default="Transformer",
                        choices=["Transformer", "Lstm", "GRU", "GPT"],
                        help='选择模型架构: Transformer, Lstm, GRU, GPT。')

    parser.add_argument('--factors', type=int, default=8,
                        help="输入特征的数量 (因子数)，应与数据匹配。")
    parser.add_argument('--data_path', type=str,
                        default="/mnt/public/home/zhijiangwan/sepsisformer/sepsisformer/data/8/mimic4_smote_8.csv",
                        help="数据集 CSV 文件的路径。")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率。')
    parser.add_argument('--epoch', type=int, default=450, # 可根据需要调整轮数
                        help='训练的总轮数。')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='数据加载的批量大小。')
    parser.add_argument('--pretrain',  default=True, # 使用 action='True'加载   False 不加载 
                        help='加载预训练权重 (不包括 MLP 头)。设置此项以启用。')
    parser.add_argument('--loadmodel', type=str, default= "/mnt/public/home/zhijiangwan/sepsisformer/sepsisformer/model2/logs/Transformer_Exp1_factors8_lr0.001_bs5000_epoch1400_20250426-113119/best_acc_model.pth",help='预训练模型文件 (.pth) 的路径。如果设置了 --pretrain，则此项为必需。')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    parser.add_argument('--seed', type=int, default=42,
                        help='设置随机种子以保证结果可复现。')
    

    
    # --- 新增：预训练相关参数 --- 


    # --- 日志和结果保存 ---
    parser.add_argument('--logs_first', type=str, default='Exp1',
                        help="日志目录名称的前缀，用于区分不同实验。")
    parser.add_argument('--results_dir', type=str,
                        default="/mnt/public/home/zhijiangwan/sepsisformer/sepsisformer/results",
                        help="保存最终指标摘要 CSV 文件的目录。")

    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    # 打印环境信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"可用 GPU 数量: {torch.cuda.device_count()}")
        try:
            current_device_index = torch.cuda.current_device()
            print(f"当前 GPU 索引: {current_device_index}")
            print(f"当前 GPU 名称: {torch.cuda.get_device_name(current_device_index)}")
        except Exception as e:
            print(f"获取 CUDA 设备信息时出错: {e}")

    # 获取命令行参数
    args = hyperParameters()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n选择的参数:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("-" * 30)

    # --- 执行训练和评估流程 ---
    entity = None # 初始化 entity 变量
    try:
        # 1. 初始化训练实体
        entity = Train(args)

        # 2. 执行训练
        entity.train()

        # 3. 评估最佳 ACC 模型
        #    注意: entity.pre_best_acc_model_state_dict 存储的是状态字典
        entity.evaluate_model(entity.pre_best_acc_model_state_dict, "BestAcc")

        # 4. 评估最佳 AUC 模型
        #    注意: entity.pre_best_auc_model_state_dict 存储的是状态字典
        entity.evaluate_model(entity.pre_best_auc_model_state_dict, "BestAuc")

        # 5. 计算并打印/保存模型复杂度信息
        entity.model_summary()

        # --- 保存最终的指标摘要 ---
        if entity: # 确保 entity 已经被成功初始化
            output_list = [entity.target]
            output_df = pd.DataFrame(output_list)

            os.makedirs(args.results_dir, exist_ok=True)
            csv_file = os.path.join(args.results_dir, f"{args.model_name}_summary_results.csv")

            if os.path.exists(csv_file):
                try:
                    existing_df = pd.read_csv(csv_file)
                    if set(existing_df.columns) == set(output_df.columns):
                        output_df = pd.concat([existing_df, output_df], ignore_index=True)
                    else:
                        print(f"警告: {csv_file} 的列名与当前结果不匹配，将覆盖原文件。")
                except Exception as e:
                    print(f"警告: 读取或合并现有结果文件 {csv_file} 时出错: {e}。将覆盖原文件。")

            try:
                output_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"\n指标摘要已保存至: {csv_file}")
            except Exception as e:
                print(f"错误: 无法保存结果到 CSV 文件 {csv_file}: {e}")
        else:
            print("\n错误: 训练实体未能成功初始化，无法保存结果。")


    except Exception as e:
        print(f"\n在执行过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n脚本执行完毕。")
