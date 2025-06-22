import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os
from pathlib import Path
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 配置变量
# =============================================================================

# 根目录路径（请修改为你的实际路径）
ROOT_DIR = r"D:\sync\docs&works\0612组间auc尝试\best_model_path_question_predictions\best_model_path"

# 列名配置
ORIROW_COLUMN = "orirow"
LATE_TRUES_COLUMN = "late_trues" 
LATE_MEAN_COLUMN = "late_mean"

# 分隔符和阈值
DELIMITER = "\t"
PREDICTION_THRESHOLD = 0.5

# 输出文件名
RAW_RESULTS_FILE = "raw_results_all_folds.csv"
FINAL_RESULTS_FILE = "final_results_averaged.csv"

# =============================================================================
# 核心处理函数（从原始代码复制）
# =============================================================================

def calculate_auc_manual(y_true, y_scores):
    """手动计算AUC值"""
    try:
        pairs = list(zip(y_scores, y_true))
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            raise ValueError("标签分布极端不均衡，无法计算AUC")
        
        auc = 0.0
        tp = 0
        fp = 0
        prev_tpr = 0.0
        prev_fpr = 0.0
        
        for i, (score, label) in enumerate(pairs):
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            if i == len(pairs) - 1 or pairs[i][0] != pairs[i + 1][0]:
                tpr = tp / n_pos
                fpr = fp / n_neg
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
                prev_tpr = tpr
                prev_fpr = fpr
        
        return auc
        
    except Exception as e:
        raise ValueError(f"AUC计算失败: {str(e)}")

def calculate_accuracy(y_true, y_scores, threshold=PREDICTION_THRESHOLD):
    """计算准确率"""
    y_pred = (y_scores >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def process_single_file(file_path):
    """
    处理单个txt文件，返回AUC列表和相关统计
    
    Returns:
    dict: 包含AUC列表、AUC标准差、平均ACC等信息
    """
    try:
        # 读取数据
        df = pd.read_csv(file_path, sep=DELIMITER, encoding='utf-8')
        
        # 检查必要的列
        required_columns = [ORIROW_COLUMN, LATE_TRUES_COLUMN, LATE_MEAN_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 数据清洗
        df = df.dropna(subset=required_columns)
        df[LATE_TRUES_COLUMN] = pd.to_numeric(df[LATE_TRUES_COLUMN], errors='coerce')
        df[LATE_MEAN_COLUMN] = pd.to_numeric(df[LATE_MEAN_COLUMN], errors='coerce')
        df = df.dropna(subset=[LATE_TRUES_COLUMN, LATE_MEAN_COLUMN])
        
        # 按orirow分组处理
        groups = df.groupby(ORIROW_COLUMN)
        
        auc_list = []
        acc_list = []
        processed_count = 0
        
        for orirow, group_data in groups:
            try:
                y_true = group_data[LATE_TRUES_COLUMN].values
                y_scores = group_data[LATE_MEAN_COLUMN].values
                
                # 检查数据有效性
                if len(y_true) < 2:
                    continue
                
                unique_labels = np.unique(y_true)
                if len(unique_labels) < 2:
                    continue
                
                # 计算AUC
                try:
                    auc = roc_auc_score(y_true, y_scores)
                except Exception:
                    auc = calculate_auc_manual(y_true, y_scores)
                
                # 计算准确率
                try:
                    acc = calculate_accuracy(y_true, y_scores, PREDICTION_THRESHOLD)
                except Exception:
                    acc = np.nan
                
                # 验证结果
                if not np.isnan(auc) and 0 <= auc <= 1:
                    auc_list.append(auc)
                    if not np.isnan(acc):
                        acc_list.append(acc)
                    processed_count += 1
                
            except Exception:
                continue
        
        if not auc_list:
            raise ValueError("没有成功处理任何有效数据")
        
        # 计算统计指标
        auc_std = np.std(auc_list, ddof=1) if len(auc_list) > 1 else 0.0
        auc_mean = np.mean(auc_list)
        acc_mean = np.mean(acc_list) if acc_list else np.nan
        acc_std = np.std(acc_list, ddof=1) if len(acc_list) > 1 else 0.0
        
        return {
            'auc_list': auc_list,
            'auc_std': auc_std,
            'auc_mean': auc_mean,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'n_students': len(auc_list),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'auc_list': [],
            'auc_std': np.nan,
            'auc_mean': np.nan,
            'acc_mean': np.nan,
            'acc_std': np.nan,
            'n_students': 0,
            'success': False,
            'error': str(e)
        }

def extract_fold_from_path(folder_name):
    """
    从文件夹名称中提取折数
    文件夹名格式：参数1_参数2_参数3_参数4_折数_其他参数...
    """
    try:
        parts = folder_name.split('_')
        if len(parts) >= 5:
            fold_str = parts[4]  # 第5个参数（索引4）是折数
            fold_num = int(fold_str)
            if 0 <= fold_num <= 4:
                return fold_num
        return None
    except (ValueError, IndexError):
        return None

def scan_directory_structure(root_dir):
    """
    扫描目录结构，找到所有的txt文件
    
    Returns:
    list: 每个元素包含 (dataset, model, fold, txt_file_path)
    """
    root_path = Path(root_dir)
    files_to_process = []
    
    if not root_path.exists():
        raise FileNotFoundError(f"根目录不存在: {root_dir}")
    
    print("扫描目录结构...")
    
    # 遍历数据集文件夹
    for dataset_dir in root_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"发现数据集: {dataset_name}")
        
        # 遍历模型文件夹
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            # 遍历折文件夹
            for fold_dir in model_dir.iterdir():
                if not fold_dir.is_dir():
                    continue
                
                # 提取折数
                fold_num = extract_fold_from_path(fold_dir.name)
                if fold_num is None:
                    continue
                
                # 查找txt文件
                txt_file = fold_dir / "qid_test_question_predictions.txt"
                if txt_file.exists():
                    files_to_process.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'fold': fold_num,
                        'txt_file': str(txt_file),
                        'fold_dir': fold_dir.name
                    })
    
    print(f"扫描完成，找到 {len(files_to_process)} 个待处理文件")
    return files_to_process

def process_all_files(root_dir):
    """
    处理所有文件的主函数
    """
    print(f"开始批量处理，根目录: {root_dir}")
    
    # 扫描目录结构
    files_to_process = scan_directory_structure(root_dir)
    
    if not files_to_process:
        raise ValueError("没有找到任何待处理的文件")
    
    # 处理所有文件
    raw_results = []
    
    print("开始处理文件...")
    for file_info in tqdm(files_to_process, desc="处理进度"):
        dataset = file_info['dataset']
        model = file_info['model']
        fold = file_info['fold']
        txt_file = file_info['txt_file']
        fold_dir = file_info['fold_dir']
        
        print(f"\n处理: {dataset}/{model}/fold_{fold}")
        
        # 处理单个文件
        result = process_single_file(txt_file)
        
        # 记录结果
        raw_results.append({
            'dataset': dataset,
            'model': model,
            'fold': fold,
            'fold_dir': fold_dir,
            'txt_file': txt_file,
            'auc_std': result['auc_std'],
            'auc_mean': result['auc_mean'],
            'acc_mean': result['acc_mean'],
            'acc_std': result['acc_std'],
            'n_students': result['n_students'],
            'success': result['success'],
            'error': result['error']
        })
        
        if result['success']:
            print(f"成功: AUC_std={result['auc_std']:.4f}, AUC_mean={result['auc_mean']:.4f}, ACC_std={result['acc_std']:.4f}, n_students={result['n_students']}")
        else:
            print(f"失败: {result['error']}")
    
    # 创建原始结果DataFrame
    raw_df = pd.DataFrame(raw_results)
    
    # 保存原始结果
    raw_output_path = Path(root_dir) / RAW_RESULTS_FILE
    raw_df.to_csv(raw_output_path, index=False, encoding='utf-8')
    print(f"\n原始结果已保存到: {raw_output_path}")
    
    # 创建最终结果（5折平均）
    print("计算5折平均结果...")
    final_results = []
    
    # 按数据集和模型分组
    for (dataset, model), group in raw_df.groupby(['dataset', 'model']):
        # 只考虑成功的结果
        success_group = group[group['success'] == True]
        
        if len(success_group) == 0:
            print(f"警告: {dataset}/{model} 没有成功的折")
            continue
        
        # 计算各项指标的平均值
        avg_auc_std = success_group['auc_std'].mean()
        avg_auc_mean = success_group['auc_mean'].mean()
        avg_acc_mean = success_group['acc_mean'].mean()
        avg_acc_std = success_group['acc_std'].mean()
        total_students = success_group['n_students'].sum()
        n_successful_folds = len(success_group)
        
        final_results.append({
            'dataset': dataset,
            'model': model,
            'avg_auc_std': avg_auc_std,
            'avg_auc_mean': avg_auc_mean,
            'avg_acc_mean': avg_acc_mean,
            'avg_acc_std': avg_acc_std,
            'total_students': total_students,
            'n_successful_folds': n_successful_folds,
            'success_rate': n_successful_folds / 5.0
        })
    
    # 创建最终结果DataFrame
    final_df = pd.DataFrame(final_results)
    
    # 保存最终结果
    final_output_path = Path(root_dir) / FINAL_RESULTS_FILE
    final_df.to_csv(final_output_path, index=False, encoding='utf-8')
    print(f"最终结果已保存到: {final_output_path}")
    
    # 显示统计信息
    print("\n=== 处理统计 ===")
    print(f"总文件数: {len(files_to_process)}")
    print(f"成功处理: {raw_df['success'].sum()}")
    print(f"失败处理: {(~raw_df['success']).sum()}")
    print(f"数据集数量: {raw_df['dataset'].nunique()}")
    print(f"模型数量: {raw_df['model'].nunique()}")
    
    print("\n=== 最终结果预览 ===")
    print(final_df.head(10))
    
    print(f"\n平均AUC标准差统计:")
    print(f"最小值: {final_df['avg_auc_std'].min():.4f}")
    print(f"最大值: {final_df['avg_auc_std'].max():.4f}")
    print(f"平均值: {final_df['avg_auc_std'].mean():.4f}")
    print(f"标准差: {final_df['avg_auc_std'].std():.4f}")
    
    print(f"\n平均ACC标准差统计:")
    print(f"最小值: {final_df['avg_acc_std'].min():.4f}")
    print(f"最大值: {final_df['avg_acc_std'].max():.4f}")
    print(f"平均值: {final_df['avg_acc_std'].mean():.4f}")
    print(f"标准差: {final_df['avg_acc_std'].std():.4f}")
    
    return {
        'raw_df': raw_df,
        'final_df': final_df,
        'raw_output_path': str(raw_output_path),
        'final_output_path': str(final_output_path)
    }

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    try:
        # 可以在这里修改根目录路径
        # ROOT_DIR = "your_custom_path"
        
        results = process_all_files(ROOT_DIR)
        print(f"\n✅ 批量处理完成!")
        print(f"原始结果文件: {results['raw_output_path']}")
        print(f"最终结果文件: {results['final_output_path']}")
        
    except Exception as e:
        print(f"\n❌ 批量处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)