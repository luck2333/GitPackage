import os
import shutil
import csv  # 导入csv模块
import json
from package_core.PDF_Processed.PDF_Processed_main import PackageDetectionPipeline
from package_core.PackageExtract.get_pairs_data_present5_test import tee_print
from package_core.UI.ui_class import RecoThread
from package_core.Segment.Segment_function import get_type

BGA_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Body X (D)', 'Body Y (E)', 'Edge Fillet Radius',
             'Ball Diameter Normal (b)', 'Exclude Pins']
QFN_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Pull Back (p)', 'Body X (D)', 'Body Y (E)',
             'Lead style', 'Pin Length (L)', 'Lead width (b)', 'Lead Height (c)', 'Exclude Pins',
             'Thermal X (D2)', 'Thermal Y (E2)']
QFP_TABLE = ['Number of pins along X', 'Number of pins along Y', 'Package Height (A)', 'Standoff (A1)',
             'Span X (E)', 'Span Y (D)', 'Body X (D1)', 'Body Y (E1)', 'Body draft (θ)', 'Edge Fillet radius',
             'Lead Length (L)', 'Lead width (b)', 'Lead Thickness (c)', 'Lead Radius (r)', 'Thermal X (D2)', 'Thermal Y (E2)']


# 自动搜索
# 自动搜索
def auto_detect_reco(pdf_path, test_type):
    tee_print(f"正在处理文件{pdf_path}...")
    # 创建 PackageDetectionPipeline 实例
    pipeline = PackageDetectionPipeline(pdf_path)  # 添加这行

    # 阶段一：预处理页面
    page_list = pipeline.step1_preprocess_pages()  # 修改这行

    # 阶段二：运行DETR检测
    raw_detr_results = pipeline.step2_run_detr_detection(page_list)  # 修改这行

    # 阶段三：关键词匹配
    modified_detr_results = pipeline.step3_match_keywords(raw_detr_results, page_list)  # 修改这行

    # 阶段四：分组封装组件
    package_views, final_data, have_page, _ = pipeline.step4_group_package_components(modified_detr_results)  # 修改这行


    # 自动识别
    type_dict = get_type(final_data)
    for index in range(0, len(final_data)):
        pdf_page_count = final_data[index]["page"]
        current_package = final_data[index]
        # 识别函数
        if final_data[index]['package_type'] is None:
            print(f"{pdf_page_count}页封装类型为空，跳过")
            continue
        elif final_data[index]['package_type'] != test_type:
            continue
        reco = RecoThread(None,pdf_path, final_data[index]["page"], final_data[index], final_data, type_dict)
        reco.run()

        # ------------------修改后的代码---------------------------
        table_mapping = {
            'BGA': BGA_TABLE,
            'QFN': QFN_TABLE,
            'QFP': QFP_TABLE
        }
        Table_type = table_mapping.get(test_type, None)
        pdf_name = os.path.basename(pdf_path)
        # 获取当前封装类型
        package_type = final_data[index]['package_type']
        row_data = [pdf_name, pdf_page_count + 1, package_type]

        if isinstance(reco.result, dict):
            data = [reco.result.get(f, []) for f in Table_type]
        else:
            data = (reco.result + [[]] * len(Table_type))[:len(Table_type)]

        # 对每个字段的值去掉首元素，转字符串，不足补空
        row_data.extend([str(v[1:]) if isinstance(v, list) and len(v) > 1 else ""
                         for v in data])

        # 写入CSV文件
        with open('bga_result.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 如果是第一次写入，先写表头（增加package_type字段）
            if os.path.getsize('bga_result.csv') == 0:
                header = ['PDF名称', '页码', 'package_type'] + Table_type
                writer.writerow(header)

            # 提取每个字段去掉第一个元素后的列表（增加package_type）
            row = [pdf_name, pdf_page_count + 1, package_type]
            for values in reco.result:
                trimmed = values[1:]  # 去掉第一个元素
                row.append(str(trimmed))  # 转为字符串写入
            writer.writerow(row)


if __name__ == "__main__":
    completed_folder = 'completed_pdf'  # 存放已经测试过的文件
    if not os.path.exists(completed_folder):
        os.makedirs(completed_folder)

    # 从文件夹中获取所有PDF文件
    folder_path = 'BGA_pdf'
    pdf_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf') or file.endswith('.PDF')]

    for pdf_path in pdf_paths:
        auto_detect_reco(pdf_path,test_type='BGA') # 测试什么类型就写什么类型-BGA、QFP、QFN

        # 构建目标路径
        destination_path = os.path.join(completed_folder, pdf_path)
        # 获取目标目录并确保其存在
        dest_dir = os.path.dirname(destination_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # 移动文件
        shutil.move(pdf_path, destination_path)
