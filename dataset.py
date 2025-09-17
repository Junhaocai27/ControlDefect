import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def prepare_mvtec_single_category_dataset(source_root, output_root, category):
    """
    为单个 MVTec-AD 类别生成 Hugging Face `datasets` 兼容的格式。
    
    Args:
        source_root (str): MVTec-AD 数据集的根目录路径。
        output_root (str): 输出转换后数据集的根目录路径。
        category (str): 要处理的单个类别名称。
    """
    # 1. 创建该类别的输出目录结构
    category_output_dir = os.path.join(output_root, category)
    images_dir = os.path.join(category_output_dir, "images")
    guides_dir = os.path.join(category_output_dir, "guides")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(guides_dir, exist_ok=True)
    
    # 2. 准备写入该类别的 metadata.jsonl 文件
    metadata_path = os.path.join(category_output_dir, "metadata.jsonl")
    global_img_id = 0

    category_path = os.path.join(source_root, category)
    if not os.path.isdir(category_path):
        print(f"错误：找不到类别目录 {category_path}")
        return 0

    # 统计ground truth查找情况
    gt_found_count = 0
    gt_missing_count = 0

    with open(metadata_path, "w", encoding="utf-8") as f:
        print(f"正在处理类别: {category}")
        
        # 遍历 train 和 test 文件夹
        for split in ["train", "test"]:
            split_path = os.path.join(category_path, split)
            if not os.path.isdir(split_path):
                continue

            # 遍历子类别（如 'good', 'broken_large' 等）
            defect_types = os.listdir(split_path)
            for defect_type in tqdm(defect_types, desc=f"Processing {category} {split}", leave=False):
                defect_path = os.path.join(split_path, defect_type)
                if not os.path.isdir(defect_path):
                    continue
                
                # 3. 根据类别和缺陷类型生成新的 prompt 格式
                if defect_type == "good":
                    prompt = f"a {category} photo"
                else:
                    prompt = f"a {category} photo with {defect_type}"

                # 遍历文件夹中的所有图片
                for filename in os.listdir(defect_path):
                    if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg")):
                        continue
                    
                    img_path = os.path.join(defect_path, filename)
                    
                    # 4. 读取原图
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告：无法读取图片 {img_path}，已跳过。")
                        continue
                    
                    # 5. 生成 guide 图像
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 10, 100)
                    
                    if defect_type == "good":
                        # 对于good图片，直接使用canny边缘
                        final_guide = edges
                    else:
                        # 对于异常图片，使用ground truth遮掩canny边缘
                        # 构造ground truth路径，考虑_mask后缀
                        gt_folder = os.path.join(category_path, "ground_truth", defect_type)
                        
                        # 获取原始文件名（不含扩展名）
                        base_name = os.path.splitext(filename)[0]
                        
                        # 尝试多种可能的ground truth文件名格式
                        possible_gt_names = [
                            f"{base_name}_mask.png",    # 000_mask.png
                            f"{base_name}_mask.jpg",    # 000_mask.jpg
                            f"{filename}",              # 原始文件名
                            f"{base_name}.png",         # 000.png
                            f"{base_name}.jpg",         # 000.jpg
                        ]
                        
                        gt_path = None
                        for gt_name in possible_gt_names:
                            potential_path = os.path.join(gt_folder, gt_name)
                            if os.path.exists(potential_path):
                                gt_path = potential_path
                                break
                        
                        if gt_path is not None:
                            # 读取ground truth
                            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                            if gt_mask is not None:
                                # 确保gt_mask和edges具有相同的尺寸
                                if gt_mask.shape != edges.shape:
                                    gt_mask = cv2.resize(gt_mask, (edges.shape[1], edges.shape[0]))
                                
                                # 将ground truth二值化，并创建反向mask
                                _, gt_binary = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
                                
                                # 创建反向mask：将异常区域设为0，正常区域设为255
                                inverse_mask = cv2.bitwise_not(gt_binary)
                                
                                # 将canny边缘与反向mask相乘，遮掩异常区域的边缘
                                # 先将两个图像归一化到[0,1]范围
                                edges_norm = edges.astype(np.float32) / 255.0
                                inverse_mask_norm = inverse_mask.astype(np.float32) / 255.0
                                
                                # 相乘操作：保留正常区域的canny，遮掩异常区域
                                masked_edges = edges_norm * inverse_mask_norm
                                
                                # 转换回[0,255]范围
                                final_guide = (masked_edges * 255).astype(np.uint8)
                                
                                gt_found_count += 1
                                if gt_found_count <= 5:  # 只显示前5个成功的案例
                                    print(f"✓ 已遮掩异常区域: {os.path.basename(gt_path)} -> {filename}")
                            else:
                                print(f"警告：无法读取ground truth {gt_path}，使用原始canny")
                                final_guide = edges
                                gt_missing_count += 1
                        else:
                            # 显示所有尝试的路径（仅用于调试前几个文件）
                            if gt_missing_count < 3:
                                print(f"警告：找不到ground truth for {filename}")
                                print(f"  尝试的路径包括:")
                                for gt_name in possible_gt_names:
                                    potential_path = os.path.join(gt_folder, gt_name)
                                    print(f"    {potential_path} - {'存在' if os.path.exists(potential_path) else '不存在'}")
                            final_guide = edges
                            gt_missing_count += 1

                    # 6. 定义输出文件的绝对路径
                    image_abs_path = os.path.join(images_dir, f"img_{global_img_id:06d}.png")
                    guide_abs_path = os.path.join(guides_dir, f"img_{global_img_id:06d}.png")

                    # 获取规范化的绝对路径用于写入metadata
                    final_image_path = os.path.abspath(image_abs_path)
                    final_guide_path = os.path.abspath(guide_abs_path)

                    # 保存处理后的图片
                    cv2.imwrite(image_abs_path, img)
                    cv2.imwrite(guide_abs_path, final_guide)

                    # 7. 构造并写入 metadata 记录 (使用绝对路径)
                    record = {
                        "text": prompt,
                        "guide": final_guide_path.replace("\\", "/"),
                        "file_name": final_image_path.replace("\\", "/")
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    global_img_id += 1

    print(f"✅ 类别 {category} 处理完成，共 {global_img_id} 张图片")
    print(f"   Guide处理统计: 成功遮掩 {gt_found_count} 个异常区域，{gt_missing_count} 个使用原始canny")
    print(f"   输出目录: {os.path.abspath(category_output_dir)}")
    return global_img_id


def verify_gt_structure(source_root, category):
    """
    验证特定类别的Ground Truth结构
    """
    print(f"\n🔍 验证 {category} 的Ground Truth结构:")
    
    category_path = os.path.join(source_root, category)
    gt_path = os.path.join(category_path, "ground_truth")
    
    if not os.path.isdir(gt_path):
        print(f"❌ Ground Truth目录不存在: {gt_path}")
        return
    
    defect_types = [d for d in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, d))]
    
    for defect_type in defect_types:
        defect_gt_path = os.path.join(gt_path, defect_type)
        gt_files = os.listdir(defect_gt_path)
        
        print(f"  📁 {defect_type}:")
        print(f"    文件数量: {len(gt_files)}")
        
        # 显示前3个文件名作为示例
        sample_files = gt_files[:3]
        for sample in sample_files:
            print(f"    示例: {sample}")


def prepare_all_mvtec_categories_separately(source_root, output_root):
    """
    为所有 MVTec-AD 类别分别生成独立的数据集。
    
    Args:
        source_root (str): MVTec-AD 数据集的根目录路径。
        output_root (str): 输出转换后数据集的根目录路径。
    """
    # MVTec-AD的所有类别
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    total_images = 0
    successful_categories = []
    failed_categories = []
    
    print("开始为每个类别生成独立数据集...")
    print("Guide生成策略:")
    print("  - good图片: 使用完整的Canny边缘检测")
    print("  - 异常图片: 使用Canny边缘，但遮掩(移除)异常区域的边缘")
    print("  - 遮掩原理: Canny × (NOT Ground_Truth_Mask)")
    print("="*60)
    
    for category in categories:
        try:
            img_count = prepare_mvtec_single_category_dataset(source_root, output_root, category)
            if img_count > 0:
                total_images += img_count
                successful_categories.append(category)
            else:
                failed_categories.append(category)
        except Exception as e:
            print(f"❌ 处理类别 {category} 时出错: {e}")
            failed_categories.append(category)
    
    # 输出总结报告
    print("\n" + "="*60)
    print("🎉 所有类别处理完成！")
    print(f"✅ 成功处理的类别 ({len(successful_categories)}个): {', '.join(successful_categories)}")
    if failed_categories:
        print(f"❌ 处理失败的类别 ({len(failed_categories)}个): {', '.join(failed_categories)}")
    print(f"📊 总计处理图片数量: {total_images}")
    print(f"📁 输出根目录: {os.path.abspath(output_root)}")
    print("\n生成的prompt格式:")
    print("  - 正常图片: 'a bottle photo' + 完整canny边缘")
    print("  - 异常图片: 'a bottle photo with broken_large' + 遮掩异常区域的canny边缘")
    print("="*60)


if __name__ == "__main__":
    # ==================== 请修改这里的路径 ====================
    # 原始 MVTec-AD 数据集的根目录
    SOURCE_DATA_ROOT = "/root/SeaS/data/mvtec_anomaly_detection"
    
    # 您希望保存转换后数据集的根目录
    # 每个类别会在这个目录下创建单独的文件夹
    OUTPUT_DATA_ROOT = "/root/control-lora-v3/exps/mvtec_ad_datasets"
    # ========================================================

    # 检查源路径是否存在
    if not os.path.isdir(SOURCE_DATA_ROOT):
        print(f"错误：源数据路径不存在 -> {SOURCE_DATA_ROOT}")
        print("请确保 SOURCE_DATA_ROOT 指向您下载的 MVTec-AD 数据集根目录。")
    else:
        # 可选：验证Ground Truth结构（取消注释来查看）
        # verify_gt_structure(SOURCE_DATA_ROOT, "bottle")
        
        # 选择运行方式：
        
        # 方式1: 处理所有类别，每个类别生成独立数据集
        prepare_all_mvtec_categories_separately(SOURCE_DATA_ROOT, OUTPUT_DATA_ROOT)
        
        # 方式2: 只处理单个类别（取消下面的注释并修改类别名称）
        # category_name = "bottle"  # 可以改为任意类别名称
        # prepare_mvtec_single_category_dataset(SOURCE_DATA_ROOT, OUTPUT_DATA_ROOT, category_name)