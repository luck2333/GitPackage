"""
PackageWizard 性能分析脚本

统计：
1. 各处理步骤的耗时
2. 模型推理时间（YOLO、DETR、DBNet、SVTR等）

使用方法:
    python run_profiler.py <pdf_path>
    或
    python run_profiler.py  (使用默认测试PDF)

输出:
    - Result/performance_report.json
    - Result/performance_report.html
"""

import sys
import os
import time

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from package_core.profiler import profiler, step_timer, model_timer


def find_test_pdf():
    """查找可用的测试PDF文件"""
    temp_dir = "Result/temp"
    if os.path.isdir(temp_dir):
        for f in os.listdir(temp_dir):
            if f.endswith('.pdf'):
                return os.path.join(temp_dir, f)
    return None


def run_auto_detect(pdf_path: str):
    """运行自动检测流程"""
    print(f"\n{'='*60}")
    print("开始自动检测流程")
    print(f"{'='*60}")

    from package_core.PDF_Processed.PDF_Processed_main import PackageDetectionPipeline

    with step_timer("初始化Pipeline"):
        pipeline = PackageDetectionPipeline(pdf_path)

    # Step 1: 预处理
    with step_timer("PDF页面预处理"):
        page_list = pipeline.step1_preprocess_pages()

    if not page_list:
        print("未找到有效页面")
        return None

    # Step 2: DETR检测
    with step_timer("DETR目标检测"):
        with model_timer("DETR", {"pages": len(page_list)}):
            detection_results = pipeline.step2_run_detr_detection(page_list)

    # Step 3: 关键词匹配
    with step_timer("关键词匹配"):
        modified_results = pipeline.step3_match_keywords(detection_results, page_list)

    # Step 4: 组件分组
    with step_timer("组件分组"):
        package_data, data2, have_page, modified_results = pipeline.step4_group_package_components(modified_results)

    return package_data


def run_param_recognition(package_type: str = "QFN"):
    """运行参数识别流程"""
    print(f"\n{'='*60}")
    print("开始参数识别流程")
    print(f"{'='*60}")

    data_path = "Result/Package_extract/data"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print("警告: 未找到待识别的图片，跳过参数识别")
        return

    try:
        from package_core.PackageExtract.common_pipeline import (
            prepare_workspace,
            get_data_location_by_yolo_dbnet,
            remove_other_annotations,
            enrich_pairs_with_lines,
            preprocess_pairs_and_text,
            run_svtr_ocr,
            DATA, DATA_COPY, DATA_BOTTOM_CROP, ONNX_OUTPUT, OPENCV_OUTPUT
        )

        # 准备工作区
        with step_timer("准备工作区"):
            prepare_workspace(DATA, DATA_COPY, DATA_BOTTOM_CROP, ONNX_OUTPUT, OPENCV_OUTPUT)

        # YOLO+DBNet检测
        with step_timer("YOLO+DBNet检测"):
            with model_timer("YOLOX", {"package_type": package_type}):
                L3 = get_data_location_by_yolo_dbnet(DATA, package_type)

        # 剔除OTHER类型
        with step_timer("数据清洗"):
            L3 = remove_other_annotations(L3)

        # 标注增强
        with step_timer("标注增强"):
            L3 = enrich_pairs_with_lines(L3, DATA, test_mode=0)

        # 预处理
        with step_timer("数据预处理"):
            L3 = preprocess_pairs_and_text(L3, key=0)

        # OCR识别
        with step_timer("SVTR-OCR识别"):
            with model_timer("SVTR", {"task": "ocr"}):
                L3 = run_svtr_ocr(L3)

        print("参数识别流程完成")

    except Exception as e:
        print(f"参数识别流程出错: {e}")
        import traceback
        traceback.print_exc()


def run_image_extraction(package_data):
    """运行图片提取流程"""
    print(f"\n{'='*60}")
    print("开始图片提取流程")
    print(f"{'='*60}")

    if not package_data:
        print("警告: 无封装数据，跳过图片提取")
        return

    try:
        from package_core.Segment.Package_pretreat import package_process

        with step_timer("图片提取"):
            package_process()
        print("图片提取完成")

    except Exception as e:
        print(f"图片提取流程出错: {e}")


def print_summary(report, total_time):
    """打印简化的性能摘要"""
    print(f"\n{'='*60}")
    print("性能分析摘要")
    print(f"{'='*60}")

    # 总运行时间
    print(f"\n总运行时间: {total_time:.2f} 秒")

    # 步骤耗时
    steps = report.get('steps', [])
    if steps:
        print(f"\n[步骤耗时统计]")
        print("-" * 50)
        for s in steps:
            print(f"  {s['name']}: {s['total_time']:.3f}秒")

    # 模型推理统计
    models = report.get('models', [])
    if models:
        print(f"\n[模型推理统计]")
        print("-" * 50)
        print(f"  {'模型名称':<20} {'调用次数':>8} {'总耗时':>10} {'平均耗时':>10}")
        print("-" * 50)
        for m in models:
            print(f"  {m['name']:<20} {m['call_count']:>8} {m['total_time']:>9.3f}s {m['avg_time']:>9.3f}s")

    print(f"\n{'='*60}\n")


def main():
    # 获取PDF路径
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = find_test_pdf()
        if not pdf_path:
            print("错误: 请提供PDF文件路径")
            print("用法: python run_profiler.py <pdf_path>")
            sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        sys.exit(1)

    print(f"PackageWizard 性能分析")
    print(f"{'='*60}")
    print(f"PDF文件: {pdf_path}")

    # 初始化profiler
    profiler.start_session({'pdf_path': pdf_path})

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行自动检测
        package_data = run_auto_detect(pdf_path)

        # 如果检测到封装，运行图片提取和参数识别
        if package_data:
            run_image_extraction(package_data)

            # 确定封装类型
            package_type = "QFN"
            for pkg in package_data:
                if 'package_type' in pkg:
                    package_type = pkg['package_type']
                    break

            run_param_recognition(package_type)

    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 计算总运行时间
        total_time = time.time() - start_time

        # 确保输出目录存在
        os.makedirs("Result", exist_ok=True)

        # 生成报告
        json_path = "Result/performance_report.json"
        report = profiler.generate_report(json_path)

        # 生成简化的HTML报告
        html_path = "Result/performance_report.html"
        generate_simple_html_report(report, html_path, total_time)

        # 打印简化摘要
        print_summary(report, total_time)

        print(f"报告已生成:")
        print(f"  - JSON: {os.path.abspath(json_path)}")
        print(f"  - HTML: {os.path.abspath(html_path)}")


def generate_simple_html_report(report, output_path, total_time):
    """生成简化的HTML报告"""
    steps = report.get('steps', [])
    models = report.get('models', [])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>性能分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .summary .time {{ font-size: 24px; color: #2e7d32; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .bar {{ background: #4CAF50; height: 20px; border-radius: 3px; }}
        .bar-container {{ background: #e0e0e0; border-radius: 3px; overflow: hidden; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PackageWizard 性能分析报告</h1>

        <div class="summary">
            <p>总运行时间: <span class="time">{total_time:.2f} 秒</span></p>
        </div>

        <h2>步骤耗时统计</h2>
        <table>
            <tr><th>步骤名称</th><th>耗时(秒)</th><th>占比</th></tr>
"""

    # 计算最大时间用于比例条
    max_step_time = max([s['total_time'] for s in steps], default=1)

    for s in steps:
        pct = (s['total_time'] / total_time * 100) if total_time > 0 else 0
        bar_width = (s['total_time'] / max_step_time * 100) if max_step_time > 0 else 0
        html += f"""
            <tr>
                <td>{s['name']}</td>
                <td>{s['total_time']:.3f}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width: {bar_width}%"></div>
                    </div>
                    {pct:.1f}%
                </td>
            </tr>
"""

    html += """
        </table>

        <h2>模型推理统计</h2>
        <table>
            <tr><th>模型名称</th><th>调用次数</th><th>总耗时(秒)</th><th>平均耗时(秒)</th></tr>
"""

    for m in models:
        html += f"""
            <tr>
                <td>{m['name']}</td>
                <td>{m['call_count']}</td>
                <td>{m['total_time']:.3f}</td>
                <td>{m['avg_time']:.3f}</td>
            </tr>
"""

    html += """
        </table>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
