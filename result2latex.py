import pandas as pd
import numpy as np
import os
from pathlib import Path

# =============================================================================
# 配置变量
# =============================================================================

# 输入CSV文件路径
INPUT_CSV_PATH = "final_results_averaged.csv"  # 请修改为你的文件路径

# 输出目录
OUTPUT_DIR = "latex_tables"

# 小数点保留位数
DECIMAL_PLACES = 4
VARIANCE_DECIMAL_PLACES = 6  # 方差保留6位小数

# LaTeX表格样式配置
LATEX_STYLE = {
    'position': 'H',  # 表格位置
    'centering': True,  # 居中
    'caption_position': 'top',  # 标题位置
    'label_prefix': 'tab:',  # 标签前缀
}

# =============================================================================
# 指标配置
# =============================================================================

# 定义所有需要生成的指标表格
METRIC_CONFIGS = [
    # AUC相关指标
    {
        'name': 'auc_std',
        'avg_col': 'avg_auc_std',
        'var_col': 'var_auc_std',
        'title': 'AUC Standard Deviation (5-Fold Average ± Variance)',
        'filename': 'auc_std_table.tex',
        'best_type': 'min'  # 标准差越小越好
    },
    {
        'name': 'auc_mean',
        'avg_col': 'avg_auc_mean',
        'var_col': 'var_auc_mean',
        'title': 'AUC Mean (5-Fold Average ± Variance)',
        'filename': 'auc_mean_table.tex',
        'best_type': 'max'  # 均值越大越好
    },
    {
        'name': 'auc_max',
        'avg_col': 'avg_auc_max',
        'var_col': 'var_auc_max',
        'title': 'AUC Maximum (5-Fold Average ± Variance)',
        'filename': 'auc_max_table.tex',
        'best_type': 'max'  # 最大值越大越好
    },
    {
        'name': 'auc_min',
        'avg_col': 'avg_auc_min',
        'var_col': 'var_auc_min',
        'title': 'AUC Minimum (5-Fold Average ± Variance)',
        'filename': 'auc_min_table.tex',
        'best_type': 'max'  # 最小值也是越大越好
    },
    {
        'name': 'auc_range',
        'avg_col': 'avg_auc_range',
        'var_col': 'var_auc_range',
        'title': 'AUC Range (5-Fold Average ± Variance)',
        'filename': 'auc_range_table.tex',
        'best_type': 'max'  # 范围越小越好
    },
    # ACC相关指标
    {
        'name': 'acc_mean',
        'avg_col': 'avg_acc_mean',
        'var_col': 'var_acc_mean',
        'title': 'ACC Mean (5-Fold Average ± Variance)',
        'filename': 'acc_mean_table.tex',
        'best_type': 'max'  # 均值越大越好
    },
    {
        'name': 'acc_std',
        'avg_col': 'avg_acc_std',
        'var_col': 'var_acc_std',
        'title': 'ACC Standard Deviation (5-Fold Average ± Variance)',
        'filename': 'acc_std_table.tex',
        'best_type': 'min'  # 标准差越小越好
    },
    {
        'name': 'acc_max',
        'avg_col': 'avg_acc_max',
        'var_col': 'var_acc_max',
        'title': 'ACC Maximum (5-Fold Average ± Variance)',
        'filename': 'acc_max_table.tex',
        'best_type': 'max'  # 最大值越大越好
    },
    {
        'name': 'acc_min',
        'avg_col': 'avg_acc_min',
        'var_col': 'var_acc_min',
        'title': 'ACC Minimum (5-Fold Average ± Variance)',
        'filename': 'acc_min_table.tex',
        'best_type': 'max'  # 最小值也是越大越好
    },
    {
        'name': 'acc_range',
        'avg_col': 'avg_acc_range',
        'var_col': 'var_acc_range',
        'title': 'ACC Range (5-Fold Average ± Variance)',
        'filename': 'acc_range_table.tex',
        'best_type': 'max'  # 范围越小越好
    }
]

# =============================================================================
# 辅助函数
# =============================================================================

def format_value_with_variance(avg_val, var_val, avg_decimal_places=DECIMAL_PLACES, var_decimal_places=VARIANCE_DECIMAL_PLACES):
    """
    格式化数值为 均值±方差 格式
    
    Parameters:
    avg_val: 平均值
    var_val: 方差
    avg_decimal_places: 平均值小数点保留位数
    var_decimal_places: 方差小数点保留位数
    
    Returns:
    str: 格式化后的字符串
    """
    if pd.isna(avg_val) or pd.isna(var_val):
        return "N/A"
    
    # 格式化为指定小数位数
    avg_str = f"{avg_val:.{avg_decimal_places}f}"
    var_str = f"{var_val:.{var_decimal_places}f}"
    
    return f"{avg_str} ± {var_str}"

def escape_latex_text(text):
    """
    转义LaTeX特殊字符
    """
    # LaTeX特殊字符转义
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def find_best_values_by_column(df, avg_col, best_type='max'):
    """
    找出每列(数据集)的最优值索引
    
    Parameters:
    df: 输入数据框
    avg_col: 平均值列名
    best_type: 'max' 或 'min'，表示最优值类型
    
    Returns:
    dict: {dataset: best_model} 的映射
    """
    best_values = {}
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        if len(dataset_data) == 0:
            continue
            
        # 找出该数据集下的最优值
        if best_type == 'max':
            best_idx = dataset_data[avg_col].idxmax()
        else:  # min
            best_idx = dataset_data[avg_col].idxmin()
        
        if pd.notna(best_idx):
            best_model = df.loc[best_idx, 'model']
            best_values[dataset] = best_model
    
    return best_values

def create_pivot_table_with_bold(df, avg_col, var_col, best_type='max'):
    """
    创建数据透视表，行为模型，列为数据集，并标记最优值
    
    Parameters:
    df: 输入数据框
    avg_col: 平均值列名
    var_col: 方差列名
    best_type: 'max' 或 'min'，表示最优值类型
    
    Returns:
    DataFrame: 透视表，内容为格式化的 均值±方差，最优值会被标记
    """
    # 找出每列的最优值
    best_values = find_best_values_by_column(df, avg_col, best_type)
    
    # 创建格式化的值列
    def format_with_bold(row):
        formatted_val = format_value_with_variance(row[avg_col], row[var_col])
        
        # 检查是否为最优值
        dataset = row['dataset']
        model = row['model']
        
        if dataset in best_values and best_values[dataset] == model:
            return f"\\textbf{{{formatted_val}}}"
        else:
            return formatted_val
    
    df['formatted_value'] = df.apply(format_with_bold, axis=1)
    
    # 创建透视表
    pivot_table = df.pivot(index='model', columns='dataset', values='formatted_value')
    
    # 填充缺失值
    pivot_table = pivot_table.fillna("N/A")
    
    return pivot_table

def generate_latex_table(pivot_table, title, label):
    """
    生成LaTeX表格代码
    
    Parameters:
    pivot_table: 数据透视表
    title: 表格标题
    label: 表格标签
    
    Returns:
    str: LaTeX表格代码
    """
    n_cols = len(pivot_table.columns)
    n_rows = len(pivot_table.index)
    
    # 生成列格式（左对齐第一列，居中其余列）
    col_format = 'l' + 'c' * n_cols
    
    # 开始生成LaTeX代码
    latex_lines = []
    
    # 表格环境开始
    latex_lines.append(r'\begin{table}[H]')
    latex_lines.append(r'\centering')
    latex_lines.append(f'\\caption{{{title}}}')
    latex_lines.append(f'\\label{{{LATEX_STYLE["label_prefix"]}{label}}}')
    
    # tabular环境开始
    latex_lines.append(f'\\begin{{tabular}}{{{col_format}}}')
    latex_lines.append(r'\toprule')
    
    # 表头：第一列为Model，其余列为数据集名称
    header_parts = ['Model'] + [escape_latex_text(col) for col in pivot_table.columns]
    header_line = ' & '.join(header_parts) + r' \\'
    latex_lines.append(header_line)
    latex_lines.append(r'\midrule')
    
    # 数据行
    for model in pivot_table.index:
        row_parts = [escape_latex_text(model)]
        for dataset in pivot_table.columns:
            value = pivot_table.loc[model, dataset]
            # 不需要再次转义，因为已经包含LaTeX命令
            row_parts.append(str(value))
        
        row_line = ' & '.join(row_parts) + r' \\'
        latex_lines.append(row_line)
    
    # 表格结束
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    return '\n'.join(latex_lines)

# =============================================================================
# 主处理函数
# =============================================================================

def convert_csv_to_latex_tables(input_csv_path=INPUT_CSV_PATH, output_dir=OUTPUT_DIR):
    """
    将CSV文件转换为多个LaTeX表格
    
    Parameters:
    input_csv_path: 输入CSV文件路径
    output_dir: 输出目录路径
    """
    
    print(f"开始处理文件: {input_csv_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"输入文件不存在: {input_csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv_path)
        print(f"成功读取数据: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        raise ValueError(f"读取CSV文件失败: {str(e)}")
    
    # 检查必要的列是否存在
    required_base_cols = ['dataset', 'model']
    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"输出目录: {output_path.absolute()}")
    
    # 生成每个指标的LaTeX表格
    generated_files = []
    
    for config in METRIC_CONFIGS:
        metric_name = config['name']
        avg_col = config['avg_col']
        var_col = config['var_col']
        title = config['title']
        filename = config['filename']
        best_type = config['best_type']
        
        print(f"\n生成表格: {metric_name} (最优类型: {best_type})")
        
        try:
            # 检查列是否存在
            if avg_col not in df.columns or var_col not in df.columns:
                print(f"警告: 跳过 {metric_name}，缺少列 {avg_col} 或 {var_col}")
                continue
            
            # 创建透视表（包含加粗最优值）
            pivot_table = create_pivot_table_with_bold(df, avg_col, var_col, best_type)
            print(f"透视表维度: {len(pivot_table.index)} 模型 × {len(pivot_table.columns)} 数据集")
            
            # 生成LaTeX代码
            latex_code = generate_latex_table(pivot_table, title, metric_name)
            
            # 保存到文件
            output_file = output_path / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            
            generated_files.append(str(output_file))
            print(f"已保存: {output_file}")
            
        except Exception as e:
            print(f"生成 {metric_name} 表格失败: {str(e)}")
            continue
    
    # 生成汇总文件
    print(f"\n生成汇总文件...")
    summary_file = output_path / "all_tables.tex"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        # 写入LaTeX文档头部
        f.write(r'\documentclass{article}' + '\n')
        f.write(r'\usepackage[utf8]{inputenc}' + '\n')
        f.write(r'\usepackage{booktabs}' + '\n')
        f.write(r'\usepackage{float}' + '\n')
        f.write(r'\usepackage{geometry}' + '\n')
        f.write(r'\geometry{margin=1in}' + '\n')
        f.write(r'\begin{document}' + '\n\n')
        
        # 包含所有生成的表格
        for config in METRIC_CONFIGS:
            if any(config['filename'] in gf for gf in generated_files):
                f.write(f'\\input{{{config["filename"]}}}' + '\n')
                f.write(r'\clearpage' + '\n\n')
        
        f.write(r'\end{document}' + '\n')
    
    print(f"汇总文件已保存: {summary_file}")
    
    # 显示统计信息
    print(f"\n=== 处理完成 ===")
    print(f"成功生成 {len(generated_files)} 个表格文件")
    print(f"输出目录: {output_path.absolute()}")
    
    print(f"\n生成的文件列表:")
    for file_path in generated_files:
        print(f"  - {Path(file_path).name}")
    print(f"  - {summary_file.name} (汇总文件)")
    
    # 显示数据集和模型统计
    print(f"\n数据统计:")
    print(f"数据集数量: {df['dataset'].nunique()}")
    print(f"数据集: {', '.join(sorted(df['dataset'].unique()))}")
    print(f"模型数量: {df['model'].nunique()}")
    print(f"模型: {', '.join(sorted(df['model'].unique()))}")
    
    # 显示最优值规则
    print(f"\n最优值加粗规则:")
    for config in METRIC_CONFIGS:
        best_desc = "最小值" if config['best_type'] == 'min' else "最大值"
        print(f"  - {config['name']}: {best_desc}加粗")
    
    return {
        'generated_files': generated_files,
        'summary_file': str(summary_file),
        'output_dir': str(output_path),
        'n_tables': len(generated_files),
        'datasets': sorted(df['dataset'].unique()),
        'models': sorted(df['model'].unique())
    }

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    try:
        # 可以在这里修改输入文件路径
        # INPUT_CSV_PATH = "your_custom_path.csv"
        
        result = convert_csv_to_latex_tables(INPUT_CSV_PATH, OUTPUT_DIR)
        
        print(f"\n✅ 转换成功!")
        print(f"生成了 {result['n_tables']} 个LaTeX表格文件")
        print(f"输出目录: {result['output_dir']}")
        
        print(f"\n💡 使用说明:")
        print(f"1. 单独使用表格：直接复制各个.tex文件的内容到你的LaTeX文档中")
        print(f"2. 完整文档：编译 all_tables.tex 查看所有表格")
        print(f"3. 需要安装LaTeX包：booktabs, float, geometry")
        print(f"4. 方差保留6位小数，每列最优值已加粗显示")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)