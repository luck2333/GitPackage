from pathlib import Path
#F3.表格内容解析与判断 F4.表格规范化流程
from package_core.Table_Processed.Table_function.GetTable import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log_path = Path("test_data.txt")

def tee_print(*args, sep=" ", end="\n"):
    msg = sep.join(map(str, args)) + end
    print(*args, sep=sep, end=end)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg)

def _process_single_page(args):
    """
    处理单个页面的表格提取（用于并行处理）
    Args:
        args: (pageNumber, TableCoordinate, pdfPath, packageType)

    Returns:
        (table, type_result, integrity_result)
    """
    pageNumber, TableCoordinate, pdfPath, packageType = args
    if TableCoordinate == []:
        return [], False, False

    table = get_table(pdfPath, pageNumber, TableCoordinate)
    if (table is None) or (table == []) or (len(table) == 1):
        print("INFO:传统方法识别的表格有误，需要调用大模型APIkey重新识别:\n")

    type_result = judge_if_package_table(table, packageType)
    integrity_result = judge_if_complete_table(table, packageType)

    return table, type_result, integrity_result

def extract_table(pdfPath, page_Number_List, Table_Coordinate_List, packageType):
    """
    传入表格坐标，获得表格信息（支持并行处理多页面）
    :param pdfPath: pdf路径
    :param page_Number_List: 存在表格页
    :param Table_Coordinate_List: 表格坐标
    :param packageType: 封装类型
    :return: 当前表格信息
    """
    print(f'文件路径：{pdfPath}\n', f'存在表格页：{page_Number_List}\n',
          f'表格对应坐标：{Table_Coordinate_List}\n', f'封装类型：{packageType}')
    Table = []
    Type = []
    Integrity = []

    try:
        # 准备并行任务参数
        tasks = [
            (pageNumber, TableCoordinate, pdfPath, packageType)
            for pageNumber, TableCoordinate in zip(page_Number_List, Table_Coordinate_List)
        ]
        # 如果只有一个页面，直接处理（避免线程开销）
        if len(tasks) == 1:
            table, type_result, integrity_result = _process_single_page(tasks[0])
            Table.append(table)
            Type.append(type_result)
            Integrity.append(integrity_result)
        else:
            # 多页面并行处理
            # 使用字典保持顺序
            results = {}

            with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as executor:
                future_to_idx = {
                    executor.submit(_process_single_page, task): idx
                    for idx, task in enumerate(tasks)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"处理页面 {page_Number_List[idx]} 时出错: {e}")
                        results[idx] = ([], False, False)

            # 按原始顺序整理结果
            for idx in range(len(tasks)):
                table, type_result, integrity_result = results.get(idx, ([], False, False))
                Table.append(table)
                Type.append(type_result)
                Integrity.append(integrity_result)
                print("##")

    finally:
        # 清理 PDF 缓存
        PDFDocumentCache.close_doc(pdfPath)

    # 根据封装表是否完整进行合并
    table = table_Select(Table, Type, Integrity)
    table = table_checked(table)
    # 提取表内信息
    data = postProcess(table, packageType)

    return data

def save_table_image(pdfPath, pageNumber, tableCoordinate):
    """
    保存表格图片到结果目录
    :param pdfPath: PDF文件路径
    :param pageNumber: 页码
    :param tableCoordinate: 表格坐标 [x0, y0, x1, y1]
    """
    # 创建保存图片的目录
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "Result", "Table_images")
    os.makedirs(result_dir, exist_ok=True)
    
    # 打开PDF并获取页面
    doc = fitz.open(pdfPath)
    page = doc.load_page(pageNumber - 1)  # 页码从0开始
    
    # 截取表格区域
    rect = fitz.Rect(tableCoordinate)
    mat = fitz.Matrix(2.0, 2.0)  # 放大2倍以提高清晰度
    pix = page.get_pixmap(matrix=mat, clip=rect)
    
    # 生成文件名
    filename = f"{Path(pdfPath).stem}_{pageNumber}.png"
    filepath = os.path.join(result_dir, filename)
    
    # 保存图片
    pix.save(filepath)
    pix = None
    doc.close()
    
    print(f"表格图片已保存: {filepath}")