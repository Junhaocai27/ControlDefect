import torch, cv2, numpy as np
import os, sys
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, SpinnerColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 只使用GPU 1

from PIL import Image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline

# 添加enhanced_mask_generator_my.py的路径到sys.path
sys.path.append('/root/control-lora-v3/lora')  # 添加这一行
sys.path.append('/root/control-lora-v3')       # 添加这一行
sys.path.append('/root/control-lora-v3/lora/lora_diffusion')
sys.path.append('/root/control-lora-v3/exps')  # 添加canny_mask.py路径

from enhanced_mask_generator_my import SAMAutoMaskExtractor, method1_binary_mask, initialize_model

# 导入canny_mask功能
try:
    from canny_mask import mask_canny
    print("✅ 成功导入canny_mask功能")
    CANNY_MASK_AVAILABLE = True
except ImportError:
    print("⚠️ 无法导入canny_mask模块，将使用内置实现")
    CANNY_MASK_AVAILABLE = False

# === GPU设备信息检测 ===
print("=" * 60)
print("🔍 GPU设备信息检测")
print("=" * 60)

# 检查CUDA是否可用
if torch.cuda.is_available():
    print(f"✅ CUDA可用")
    print(f"📊 CUDA版本: {torch.version.cuda}")
    print(f"🎯 PyTorch版本: {torch.__version__}")
    
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"🖥️ 可用GPU数量: {gpu_count}")
    
    # 显示每个GPU的详细信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 显示当前使用的GPU
    current_device = torch.cuda.current_device()
    current_gpu_name = torch.cuda.get_device_name(current_device)
    print(f"🎯 当前使用GPU: GPU {current_device} ({current_gpu_name})")
    
    # 显示GPU内存使用情况
    print(f"💾 GPU内存使用情况:")
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: 已分配 {allocated:.2f}GB / 已缓存 {cached:.2f}GB / 总计 {total:.1f}GB")
    
    # 恢复到原来的设备
    torch.cuda.set_device(current_device)
    
    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        print(f"🔧 CUDA_VISIBLE_DEVICES: {cuda_visible}")
    else:
        print(f"🔧 CUDA_VISIBLE_DEVICES: 未设置 (所有GPU可见)")
    
else:
    print("❌ CUDA不可用，将使用CPU运行")
    print(f"🎯 PyTorch版本: {torch.__version__}")

print("=" * 60)

# === 配置区 ===
BASE_MODEL = "/root/sdv1_5"
LORA_DIR   = "/root/control-lora-v3/out_gt1/bottle_sd15/pytorch_lora_weights.safetensors"  # 训练输出目录
INPUT_IMG  = "/root/control-lora-v3/exps/mvtec_ad_datasets/bottle/images/img_000000.png"  # 输入正常图片
PROMPT     = "a bottle photo with broken_large"
NEGATIVE   = "low quality, bad anatomy, blurry, watermark"
STEPS      = 50
SEED       = 1234
H          = 512  # 输出尺寸修改为512x512
W          = 512  # 输出尺寸修改为512x512

# Canny边缘检测参数
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 100

# 批量生成参数
BATCH_CONFIG = {
    "enable_batch": False,           # 是否启用批量生成
    "num_images": 1000,                # 生成图片数量（改为3张用于测试）
    "start_seed": 1000,             # 起始随机种子
    "seed_step": 1,                 # 种子步长
    "save_intermediate": False,     # 是否保存中间结果（约束掩码、异常掩码等）
    "save_only_final": True,        # 是否只保存最终的缺陷图像
    "batch_size": 10,               # 每批处理数量（用于显示进度）
    "random_reference": True,       # 是否随机选择reference_image
    "reference_images_dir": "/root/control-lora-v3/lora/data/mvtec_masks_only/bottle/ground_truth/broken_large",  # reference图像目录
    "quiet_mode": True,             # 批量生成时简化输出（只显示进度）
}

# Anomaly mask生成参数
ANOMALY_GENERATION_CONFIG = {
    "prompt": "a <bottle_broken_large>",
    "reference_image": "/root/control-lora-v3/lora/data/mvtec_masks_only/bottle/ground_truth/broken_large/000.png",
    "normal_image": INPUT_IMG,
    "strengths": [0.8],
    "threshold": 0.5,
    "seed": 42,
    "model_id": "/root/stable-diffusion-2-1-base",
    "lora_path": "/root/control-lora-v3/lora/lora_weight/bottle_broken_large/final_lora.safetensors",
}

# 输出目录
OUTPUT_DIR = "/root/control-lora-v3/outputs/anomaly_generation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 显示配置信息 ===
print("📋 运行配置信息:")
print(f"  输入图像: {INPUT_IMG}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  输出尺寸: {W}x{H}")
print(f"  批量生成: {'启用' if BATCH_CONFIG['enable_batch'] else '禁用'}")
if BATCH_CONFIG['enable_batch']:
    print(f"  生成数量: {BATCH_CONFIG['num_images']}")
print("=" * 60)

class ModelManager:
    """模型管理器 - 一次加载，多次使用"""
    def __init__(self):
        self.sam_extractor = None
        self.anomaly_pipe = None
        self.defective_pipe = None
        self.models_loaded = False
        
    def load_models(self, quiet_mode=False):
        """一次性加载所有需要的模型"""
        if self.models_loaded:
            if not quiet_mode:
                print("✅ 模型已经加载过了，跳过重复加载")
            return True
            
        if not quiet_mode:
            print("🔄 开始加载所有模型...")
            print("="*60)
        
        try:
            # 1. 加载SAM模型
            if not quiet_mode:
                print("📦 加载SAM模型...")
            self.sam_extractor = SAMAutoMaskExtractor()
            if not quiet_mode:
                print("✅ SAM模型加载完成")
            
            # 2. 加载异常掩码生成Pipeline
            if not quiet_mode:
                print("📦 加载异常掩码生成模型...")
            
            # 检查LoRA文件
            lora_path = ANOMALY_GENERATION_CONFIG['lora_path']
            if not os.path.exists(lora_path):
                if not quiet_mode:
                    print(f"⚠️ LoRA文件不存在: {lora_path}")
                # 尝试寻找可能的路径
                possible_paths = [
                    "/root/control-lora-v3/lora/lora_weight/bottle_broken_large/final_lora.safetensors",
                    "/root/lora/lora_weight/bottle_broken_large/final_lora.safetensors",
                    "/root/control-lora-v3/out_gt1/bottle_sd15/pytorch_lora_weights.safetensors",
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        if not quiet_mode:
                            print(f"✅ 找到LoRA文件: {path}")
                        ANOMALY_GENERATION_CONFIG['lora_path'] = path
                        lora_path = path
                        break
                else:
                    if not quiet_mode:
                        print("❌ 未找到任何LoRA文件")
                    return False
            
            self.anomaly_pipe = initialize_model(
                custom_model_id=ANOMALY_GENERATION_CONFIG["model_id"],
                custom_lora_path=lora_path
            )
            if not quiet_mode:
                print("✅ 异常掩码生成模型加载完成")
            
            # 3. 加载缺陷图像生成Pipeline
            if not quiet_mode:
                print("📦 加载缺陷图像生成模型...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            unet = UNet2DConditionModelEx.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=torch_dtype)
            unet = unet.add_extra_conditions(["canny"])

            self.defective_pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(BASE_MODEL, unet=unet, torch_dtype=torch_dtype)
            self.defective_pipe.scheduler = UniPCMultistepScheduler.from_config(self.defective_pipe.scheduler.config)

            if torch.cuda.is_available():
                self.defective_pipe.enable_model_cpu_offload()
                try:
                    self.defective_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            # 加载LoRA权重
            self.defective_pipe.load_lora_weights(LORA_DIR, weight_name=None)
            if not quiet_mode:
                print("✅ 缺陷图像生成模型加载完成")
            
            # 显示加载后的GPU内存使用情况
            if torch.cuda.is_available() and not quiet_mode:
                print("💾 模型加载后GPU内存使用情况:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    print(f"  GPU {i}: 已分配 {allocated:.2f}GB / 已缓存 {cached:.2f}GB / 总计 {total:.1f}GB")
            
            self.models_loaded = True
            if not quiet_mode:
                print("="*60)
                print("🎉 所有模型加载完成！")
            return True
            
        except Exception as e:
            if not quiet_mode:
                print(f"❌ 模型加载失败: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def generate_constraint_mask(self, normal_image_path, width=900, height=900):
        """生成约束掩码"""
        if not self.models_loaded or self.sam_extractor is None:
            raise RuntimeError("SAM模型未加载")
            
        constraint_mask, sam_debug_info = self.sam_extractor.extract_subject_mask(
            normal_image=normal_image_path,
            width=width,
            height=height
        )
        return constraint_mask, sam_debug_info
    
    def generate_anomaly_mask_with_model(self, reference_image, prompt, strength, seed, threshold):
        """使用已加载的模型生成异常掩码"""
        if not self.models_loaded or self.anomaly_pipe is None:
            raise RuntimeError("异常掩码生成模型未加载")
            
        result_rgb, result_binary, _ = method1_binary_mask(
            self.anomaly_pipe,
            prompt_text=prompt,
            reference_image=reference_image,
            strength=strength,
            seed=seed,
            threshold=threshold
        )
        return result_rgb, result_binary
    
    def generate_defective_image_with_model(self, anomaly_canny_pil, prompt, negative_prompt, steps, seed, height, width):
        """使用已加载的模型生成缺陷图像"""
        if not self.models_loaded or self.defective_pipe is None:
            raise RuntimeError("缺陷图像生成模型未加载")
            
        g = torch.Generator(device=self.defective_pipe._execution_device).manual_seed(seed)
        images = self.defective_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=g,
            height=height,
            width=width,
            image=anomaly_canny_pil,
        ).images
        return images[0]

# 全局模型管理器实例
model_manager = ModelManager()

def get_random_reference_image():
    """从指定目录随机选择一个reference图像"""
    import random
    import glob
    
    if not BATCH_CONFIG.get("random_reference", False):
        # 如果不启用随机选择，返回默认的reference_image
        return ANOMALY_GENERATION_CONFIG['reference_image']
    
    reference_dir = BATCH_CONFIG.get("reference_images_dir", "")
    if not reference_dir or not os.path.exists(reference_dir):
        print(f"⚠️ Reference目录不存在: {reference_dir}")
        print("使用默认reference_image")
        return ANOMALY_GENERATION_CONFIG['reference_image']
    
    # 支持常见的图像格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(reference_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"⚠️ 在目录中未找到图像文件: {reference_dir}")
        print("使用默认reference_image")
        return ANOMALY_GENERATION_CONFIG['reference_image']
    
    # 随机选择一个图像
    selected_image = random.choice(image_files)
    print(f"🎲 随机选择reference图像: {os.path.basename(selected_image)}")
    
    return selected_image

def generate_anomaly_mask(seed=None, save_suffix="", custom_reference_image=None):
    """第一步：使用预加载的模型生成anomaly_mask"""
    print("="*60)
    print("第一步：生成 Anomaly Mask")
    print("="*60)
    
    # 使用传入的种子或默认种子
    current_seed = seed if seed is not None else ANOMALY_GENERATION_CONFIG['seed']
    
    # 使用自定义reference_image或默认的
    reference_image_path = custom_reference_image if custom_reference_image else ANOMALY_GENERATION_CONFIG['reference_image']
    print(f"📷 使用reference图像: {reference_image_path}")
    
    try:
        # 确保模型已加载
        if not model_manager.models_loaded:
            print("⚠️ 模型未加载，开始加载...")
            if not model_manager.load_models():
                return None, None, None
        
        # 使用SAM提取主体掩码（作为约束）- 只在第一次或需要时执行
        constraint_mask_path = os.path.join(OUTPUT_DIR, "01_constraint_mask.png")
        if not os.path.exists(constraint_mask_path):
            print(f"使用SAM提取主体掩码: {ANOMALY_GENERATION_CONFIG['normal_image']}")
            constraint_mask, sam_debug_info = model_manager.generate_constraint_mask(
                ANOMALY_GENERATION_CONFIG['normal_image'],
                width=900,  # 修改为900x900
                height=900  # 修改为900x900
            )
            
            # 保存约束掩码
            Image.fromarray(constraint_mask, mode='L').save(constraint_mask_path)
            print(f"约束掩码已保存: {constraint_mask_path}")
        else:
            print(f"✅ 约束掩码已存在，跳过SAM提取: {constraint_mask_path}")
            constraint_mask = np.array(Image.open(constraint_mask_path).convert('L'))
        
        # 读取参考图像 - 注意这里仍然用512x512，因为这是用于生成anomaly mask的
        print(f"读取参考图像: {reference_image_path}")
        if not os.path.exists(reference_image_path):
            print(f"❌ 参考图像不存在: {reference_image_path}")
            return None, None, None
            
        orig_img = Image.open(reference_image_path).convert("RGB").resize((512, 512))  # 保持512x512用于异常掩码生成
        
        # 生成异常掩码
        print("生成异常掩码...")
        for strength in ANOMALY_GENERATION_CONFIG['strengths']:
            print(f"处理 strength={strength}")
            
            # 使用预加载的模型生成异常掩码
            result_rgb, result_binary = model_manager.generate_anomaly_mask_with_model(
                reference_image=orig_img,
                prompt=ANOMALY_GENERATION_CONFIG['prompt'],
                strength=strength,
                seed=current_seed,
                threshold=ANOMALY_GENERATION_CONFIG['threshold']
            )
            
            # 保存异常掩码
            if save_suffix:
                anomaly_mask_path = os.path.join(OUTPUT_DIR, f"02_anomaly_mask_{strength}_{save_suffix}.png")
            else:
                anomaly_mask_path = os.path.join(OUTPUT_DIR, f"02_anomaly_mask_{strength}.png")
            result_binary.save(anomaly_mask_path)
            print(f"异常掩码已保存: {anomaly_mask_path}")
            
            return np.array(result_binary), constraint_mask, anomaly_mask_path
            
    except Exception as e:
        print(f"生成异常掩码失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def extract_canny_from_image(image_path, low_threshold=50, high_threshold=100):
    """从输入图像提取Canny边缘图"""
    print("="*60)
    print("提取正常图像的 Canny 边缘图")
    print("="*60)
    
    try:
        # 读取并预处理图像
        image = Image.open(image_path).convert("RGB").resize((W, H))
        image_np = np.array(image)
        
        # 转换为灰度图
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 应用Canny边缘检测
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 保存Canny边缘图
        normal_canny_path = os.path.join(OUTPUT_DIR, "00_normal_canny.png")
        cv2.imwrite(normal_canny_path, edges)
        print(f"正常Canny边缘图已保存: {normal_canny_path}")
        
        return edges, normal_canny_path
        
    except Exception as e:
        print(f"提取Canny边缘图失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def apply_anomaly_mask_to_canny(anomaly_mask, normal_canny):
    """第二步：使用canny_mask.py将anomaly_mask覆盖在正常图片的canny上"""
    print("="*60)
    print("第二步：将 Anomaly Mask 应用到 Normal Canny")
    print("="*60)
    
    try:
        # 如果可以使用canny_mask模块，则使用原始功能
        if CANNY_MASK_AVAILABLE:
            print("使用canny_mask.py原始功能")
            
            # 保存临时文件用于canny_mask函数
            temp_canny_path = os.path.join(OUTPUT_DIR, "temp_normal_canny.png")
            temp_mask_path = os.path.join(OUTPUT_DIR, "temp_anomaly_mask.png")
            output_path = os.path.join(OUTPUT_DIR, "03_anomaly_canny.png")
            
            # 保存临时文件
            cv2.imwrite(temp_canny_path, normal_canny)
            
            # 将512x512的anomaly_mask放大到900x900后保存
            anomaly_mask_900 = cv2.resize(anomaly_mask, (900, 900))
            cv2.imwrite(temp_mask_path, anomaly_mask_900)
            print(f"Anomaly mask从{anomaly_mask.shape}调整到{anomaly_mask_900.shape}")
            
            # 调用canny_mask.py的核心功能
            result = mask_canny(temp_canny_path, temp_mask_path, output_path)
            
            # 清理临时文件
            try:
                os.remove(temp_canny_path)
                os.remove(temp_mask_path)
            except:
                pass
            
            if result is not None:
                print(f"✅ 使用canny_mask.py成功处理")
                print(f"Anomaly Canny已保存: {output_path}")
                
                # 转换为3通道RGB格式
                anomaly_canny_3ch = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
                anomaly_canny_pil = Image.fromarray(anomaly_canny_3ch)
                
                return anomaly_canny_pil, output_path
            else:
                print("❌ canny_mask.py处理失败，使用内置实现")
        
        # 内置实现（与canny_mask.py逻辑一致）
        print("使用内置canny_mask实现")
        
        canny = normal_canny
        print(f"Canny图像尺寸: {canny.shape}")
        print(f"Anomaly mask原始尺寸: {anomaly_mask.shape}")
        
        # 将anomaly_mask从512x512调整到900x900（与canny相同尺寸）
        if canny.shape != anomaly_mask.shape:
            anomaly_mask_resized = cv2.resize(anomaly_mask, (canny.shape[1], canny.shape[0]))
            print(f"Anomaly mask调整后尺寸: {anomaly_mask_resized.shape}")
        else:
            anomaly_mask_resized = anomaly_mask
            print("Anomaly mask尺寸与Canny相同，无需调整")
        
        # === 完全按照canny_mask.py的逻辑实现 ===
        # 反转mask：将白色区域变为黑色（用于遮罩）
        inverse_mask = 255 - anomaly_mask_resized
        
        # 相乘操作：遮罩掉mask白色区域的canny内容
        # 这是canny_mask.py中的核心逻辑
        result = (canny.astype(np.float32) / 255.0 * inverse_mask.astype(np.float32) / 255.0 * 255).astype(np.uint8)
        
        print(f"遮罩操作完成:")
        print(f"  原始Canny非零像素: {np.count_nonzero(canny)}")
        print(f"  遮罩后非零像素: {np.count_nonzero(result)}")
        print(f"  被遮盖的像素: {np.count_nonzero(canny) - np.count_nonzero(result)}")
        
        # 保存遮罩后的结果
        anomaly_canny_path = os.path.join(OUTPUT_DIR, "03_anomaly_canny.png")
        cv2.imwrite(anomaly_canny_path, result)
        print(f"Anomaly Canny已保存: {anomaly_canny_path}")
        
        # 可选：保存调整后的mask（用于调试）
        mask_debug_path = os.path.join(OUTPUT_DIR, "03_anomaly_mask_resized_900x900.png")
        cv2.imwrite(mask_debug_path, anomaly_mask_resized)
        print(f"调整到900x900的异常掩码已保存: {mask_debug_path}")
        
        # 转换为3通道RGB格式，用于后续处理
        anomaly_canny_3ch = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        anomaly_canny_pil = Image.fromarray(anomaly_canny_3ch)
        
        return anomaly_canny_pil, anomaly_canny_path
        
    except Exception as e:
        print(f"应用anomaly mask到canny失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_defective_image(anomaly_canny_pil, seed=None):
    """第三步：使用预加载的模型生成带缺陷的图片"""
    print("="*60)
    print("第三步：生成带缺陷的图片")
    print("="*60)
    
    # 使用传入的种子或默认种子
    current_seed = seed if seed is not None else SEED
    
    try:
        # 确保模型已加载
        if not model_manager.models_loaded:
            print("⚠️ 模型未加载，开始加载...")
            if not model_manager.load_models():
                return None, None
        
        print("🚀 使用预加载的模型生成图像...")
        
        # 使用预加载的模型生成图像
        gen_img = model_manager.generate_defective_image_with_model(
            anomaly_canny_pil=anomaly_canny_pil,
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            steps=STEPS,
            seed=current_seed,
            height=H,
            width=W
        )
        
        # 保存生成的图像
        defective_image_path = os.path.join(OUTPUT_DIR, "04_defective_image.png")
        gen_img.save(defective_image_path)
        print(f"带缺陷图像已保存: {defective_image_path}")
        
        return gen_img, defective_image_path
        
    except Exception as e:
        print(f"生成带缺陷图像失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_single_anomaly_image(image_index, anomaly_seed, defective_seed, normal_canny, save_intermediate=False, quiet_mode=False):
    """生成单张异常图像
    
    Args:
        image_index: 图像索引
        anomaly_seed: 用于异常掩码生成的种子
        defective_seed: 用于缺陷图像生成的种子（保持固定以维持风格一致性）
        normal_canny: 正常图像的canny边缘
        save_intermediate: 是否保存中间结果
        quiet_mode: 是否静默模式
    """
    import time
    start_time = time.time()
    
    if not quiet_mode:
        print(f"正在生成第 {image_index + 1} 张图像 (异常种子={anomaly_seed}, 风格种子={defective_seed})...")
    
    try:
        # 随机选择reference_image（如果启用）
        reference_image = get_random_reference_image()
        
        # 第一步：生成异常掩码（使用变化的异常种子）
        step1_start = time.time()
        anomaly_mask, constraint_mask, anomaly_mask_path = generate_anomaly_mask(
            seed=anomaly_seed, 
            save_suffix=f"idx_{image_index:06d}_aseed_{anomaly_seed}",
            custom_reference_image=reference_image
        )
        step1_time = time.time() - step1_start
        
        if anomaly_mask is None:
            if not quiet_mode:
                print(f"❌ 第 {image_index + 1} 张图像的异常掩码生成失败")
            return None
        
        # 第二步：将异常掩码应用到正常canny图像
        step2_start = time.time()
        anomaly_canny_pil, anomaly_canny_path = apply_anomaly_mask_to_canny(anomaly_mask, normal_canny)
        step2_time = time.time() - step2_start
        
        if anomaly_canny_pil is None:
            if not quiet_mode:
                print(f"❌ 第 {image_index + 1} 张图像的异常canny生成失败")
            return None
        
        # 第三步：使用异常canny生成带缺陷的图像（使用固定的风格种子）
        step3_start = time.time()
        defective_img, defective_image_path = generate_defective_image(anomaly_canny_pil, seed=defective_seed)
        step3_time = time.time() - step3_start
        
        if defective_img is None:
            if not quiet_mode:
                print(f"❌ 第 {image_index + 1} 张图像的缺陷图像生成失败")
            return None
        
        # 重命名最终图像文件（文件名包含两个种子信息）
        final_image_path = os.path.join(OUTPUT_DIR, f"defective_image_{image_index:06d}_aseed_{anomaly_seed}_dseed_{defective_seed}.png")
        defective_img.save(final_image_path)
        
        # 如果不保存中间结果，删除临时文件
        if not save_intermediate:
            try:
                if os.path.exists(anomaly_mask_path):
                    os.remove(anomaly_mask_path)
                if os.path.exists(anomaly_canny_path):
                    os.remove(anomaly_canny_path)
                if os.path.exists(defective_image_path):
                    os.remove(defective_image_path)
            except Exception as e:
                if not quiet_mode:
                    print(f"⚠️ 清理临时文件失败: {e}")
        
        total_time = time.time() - start_time
        
        if not quiet_mode:
            print(f"✅ 第 {image_index + 1} 张图像生成完成: {final_image_path}")
            print(f"⏱️ 用时: 总计{total_time:.1f}s (掩码{step1_time:.1f}s + 应用{step2_time:.1f}s + 生成{step3_time:.1f}s)")
        
        return final_image_path
        
    except Exception as e:
        if not quiet_mode:
            print(f"❌ 第 {image_index + 1} 张图像生成失败: {e}")
            import traceback
            traceback.print_exc()
        return None

def batch_generate_anomaly_images():
    """批量生成异常图像"""
    quiet_mode = BATCH_CONFIG.get("quiet_mode", False)
    
    if not quiet_mode:
        print("🚀 开始批量生成异常图像")
        print("="*80)
    
    # 配置参数
    num_images = BATCH_CONFIG["num_images"]
    start_seed = BATCH_CONFIG["start_seed"]
    seed_step = BATCH_CONFIG["seed_step"]
    save_intermediate = BATCH_CONFIG["save_intermediate"]
    batch_size = BATCH_CONFIG["batch_size"]
    
    # 初始化rich console
    console = Console()
    
    if not quiet_mode:
        # 创建漂亮的配置表格
        config_table = Table(title="� 批量生成配置", show_header=True, header_style="bold blue")
        config_table.add_column("配置项", style="cyan", no_wrap=True)
        config_table.add_column("值", style="magenta")
        
        config_table.add_row("图像数量", str(num_images))
        config_table.add_row("起始种子", str(start_seed))
        config_table.add_row("种子步长", str(seed_step))
        config_table.add_row("风格种子(固定)", str(start_seed))
        config_table.add_row("异常种子范围", f"42 - {42 + num_images * seed_step - 1}")
        config_table.add_row("保存中间结果", "是" if save_intermediate else "否")
        config_table.add_row("随机Reference", "是" if BATCH_CONFIG.get('random_reference', False) else "否")
        if BATCH_CONFIG.get('random_reference', False):
            config_table.add_row("Reference目录", BATCH_CONFIG.get('reference_images_dir', 'N/A'))
        config_table.add_row("输出目录", OUTPUT_DIR)
        
        console.print(config_table)
        console.print()
    else:
        # 简化模式只显示基本信息
        console.print(f"🚀 批量生成 {num_images} 张异常图像 (风格种子固定: {start_seed}, 异常种子: 42-{42 + num_images * seed_step - 1})")
    
    # 预加载所有模型（关键优化！）
    if not quiet_mode:
        console.print("⏰ 预加载所有模型以提升批量生成速度...")
    if not model_manager.load_models(quiet_mode=quiet_mode):
        console.print("❌ 模型加载失败，停止批量生成")
        return []
    
    # 如果启用随机reference，检查并列出可用图像
    if BATCH_CONFIG.get('random_reference', False):
        import glob
        reference_dir = BATCH_CONFIG.get('reference_images_dir', '')
        if os.path.exists(reference_dir):
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            all_images = []
            for ext in image_extensions:
                pattern = os.path.join(reference_dir, ext)
                all_images.extend(glob.glob(pattern))
            
            if not quiet_mode:
                console.print(f"🎲 找到 {len(all_images)} 个可用的reference图像:")
                for i, img_path in enumerate(all_images[:5]):  # 只显示前5个
                    console.print(f"   {i+1}. {os.path.basename(img_path)}")
                if len(all_images) > 5:
                    console.print(f"   ... 还有 {len(all_images)-5} 个图像")
                console.print()
            else:
                console.print(f"📂 找到 {len(all_images)} 个reference图像")
        else:
            if not quiet_mode:
                console.print(f"⚠️ Reference目录不存在，将使用默认图像: {reference_dir}")
    
    # 第零步：提取正常图像的Canny边缘图（只需要做一次）
    if not quiet_mode:
        console.print("📸 提取正常图像的Canny边缘...")
    normal_canny, normal_canny_path = extract_canny_from_image(
        INPUT_IMG, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
    )
    if normal_canny is None:
        console.print("❌ Canny边缘图提取失败，停止批量生成")
        return
    
    # 统计变量
    success_count = 0
    failed_count = 0
    generated_files = []
    
    # 批量生成
    import time
    start_time = time.time()
    
    # 使用rich进度条
    if quiet_mode:
        # 简化模式：只显示基本进度条
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("🎨 生成异常图像", total=num_images)
            
            for i in range(num_images):
                # 异常种子变化：从42开始递增
                anomaly_seed = 42 + i * seed_step
                # 风格种子固定：使用start_seed保持一致风格
                defective_seed = start_seed
                
                # 生成单张图像
                result_path = generate_single_anomaly_image(
                    i, anomaly_seed, defective_seed, normal_canny, save_intermediate, quiet_mode=True
                )
                
                # 更新统计
                if result_path:
                    success_count += 1
                    generated_files.append(result_path)
                else:
                    failed_count += 1
                
                # 更新进度条描述
                progress.update(task, advance=1, description=f"🎨 生成异常图像 [green]✅{success_count}[/green] [red]❌{failed_count}[/red]")
    else:
        # 详细模式：显示丰富的进度信息
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("•"),
            TextColumn("[green]✅ {task.fields[success]}"),
            TextColumn("[red]❌ {task.fields[failed]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("🎨 批量生成异常图像", total=num_images, success=0, failed=0)
            
            for i in range(num_images):
                # 异常种子变化：从42开始递增
                anomaly_seed = 42 + i * seed_step
                # 风格种子固定：使用start_seed保持一致风格
                defective_seed = start_seed
                
                # 生成单张图像
                result_path = generate_single_anomaly_image(
                    i, anomaly_seed, defective_seed, normal_canny, save_intermediate, quiet_mode=True
                )
                
                # 更新统计
                if result_path:
                    success_count += 1
                    generated_files.append(result_path)
                else:
                    failed_count += 1
                
                # 更新进度条
                progress.update(task, advance=1, success=success_count, failed=failed_count)
    
    # 最终统计
    total_time = time.time() - start_time
    
    # 显示最终结果
    console.print()
    if quiet_mode:
        console.print(f"🎉 批量生成完成! [green]✅{success_count}[/green]/{num_images} ({success_count/num_images:.1%}) 用时:{total_time/60:.1f}min")
    else:
        # 创建漂亮的结果表格
        table = Table(title="📊 批量生成结果", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan", no_wrap=True)
        table.add_column("数值", style="green")
        
        table.add_row("总计划数量", str(num_images))
        table.add_row("成功生成", f"[green]{success_count}[/green]")
        table.add_row("失败数量", f"[red]{failed_count}[/red]" if failed_count > 0 else "0")
        table.add_row("成功率", f"{success_count/num_images:.1%}")
        table.add_row("总用时", f"{total_time/60:.1f} 分钟")
        table.add_row("平均用时", f"{total_time/num_images:.1f} 秒/张")
        
        console.print(table)
        console.print(f"📁 输出目录: [blue]{OUTPUT_DIR}[/blue]")
        console.print(f"📋 文件命名格式: [yellow]defective_image_XXXXXX_aseed_YYYY_dseed_ZZZZ.png[/yellow]")
        console.print(f"   其中: XXXXXX=图像索引, YYYY=异常种子, ZZZZ=风格种子(固定{start_seed})")
        
        if not save_intermediate:
            console.print("💾 已清理中间文件，仅保留最终的缺陷图像")
    
    return generated_files

def create_final_comparison(original_img_path, anomaly_mask_path, anomaly_canny_path, defective_image_path):
    """创建最终的对比图像"""
    print("="*60)
    print("创建最终对比图像")
    print("="*60)
    
    try:
        # 读取所有图像并调整到900x900
        orig_img = Image.open(original_img_path).convert("RGB").resize((900, 900))  # 修改为900
        anomaly_mask = Image.open(anomaly_mask_path).convert("RGB").resize((900, 900))  # 修改为900
        anomaly_canny = Image.open(anomaly_canny_path).convert("RGB").resize((900, 900))  # 修改为900
        defective_img = Image.open(defective_image_path).convert("RGB").resize((900, 900))  # 修改为900
        
        # 创建2x2的对比图
        combined = Image.new("RGB", (900*2, 900*2))  # 修改为900
        combined.paste(orig_img, (0, 0))           # 左上：原图
        combined.paste(anomaly_mask, (900, 0))     # 右上：异常掩码
        combined.paste(anomaly_canny, (0, 900))    # 左下：异常canny
        combined.paste(defective_img, (900, 900))  # 右下：生成的缺陷图
        
        # 保存对比图
        comparison_path = os.path.join(OUTPUT_DIR, "05_final_comparison.png")
        combined.save(comparison_path)
        print(f"最终对比图已保存: {comparison_path}")
        
        # 创建单行对比图
        single_row = Image.new("RGB", (900*4, 900))  # 修改为900
        single_row.paste(orig_img, (0, 0))
        single_row.paste(anomaly_mask, (900, 0))
        single_row.paste(anomaly_canny, (900*2, 0))
        single_row.paste(defective_img, (900*3, 0))
        
        single_row_path = os.path.join(OUTPUT_DIR, "06_single_row_comparison.png")
        single_row.save(single_row_path)
        print(f"单行对比图已保存: {single_row_path}")
        
        return comparison_path, single_row_path
        
    except Exception as e:
        print(f"创建对比图失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """主函数：完整的推理流程"""
    print("🚀 开始异常图像生成流程")
    print("="*80)
    
    # 检查是否启用批量生成
    if BATCH_CONFIG["enable_batch"]:
        # 批量生成模式
        if BATCH_CONFIG["save_only_final"]:
            print("📝 批量生成模式: 仅保存最终缺陷图像")
            BATCH_CONFIG["save_intermediate"] = False
        
        generated_files = batch_generate_anomaly_images()
        
        print("="*80)
        print("🎉 批量异常图像生成流程执行完成！")
        print(f"📁 输出目录: {OUTPUT_DIR}")
        print(f"📋 生成了 {len(generated_files)} 张缺陷图像")
        print("="*80)
        
    else:
        # 单张生成模式（原有逻辑）
        print("📝 单张生成模式")
        
        # 预加载模型
        print("⏰ 预加载所有模型...")
        if not model_manager.load_models():
            print("❌ 模型加载失败，停止执行")
            return
        
        # 第零步：从输入图像提取正常的Canny边缘图
        normal_canny, normal_canny_path = extract_canny_from_image(
            INPUT_IMG, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
        )
        if normal_canny is None:
            print("❌ Canny边缘图提取失败，停止执行")
            return
        
        # 第一步：生成异常掩码
        anomaly_mask, constraint_mask, anomaly_mask_path = generate_anomaly_mask()
        if anomaly_mask is None:
            print("❌ 异常掩码生成失败，停止执行")
            return
        
        # 第二步：将异常掩码应用到正常canny图像
        anomaly_canny_pil, anomaly_canny_path = apply_anomaly_mask_to_canny(anomaly_mask, normal_canny)
        if anomaly_canny_pil is None:
            print("❌ 异常canny生成失败，停止执行")
            return
        
        # 第三步：使用异常canny生成带缺陷的图像
        defective_img, defective_image_path = generate_defective_image(anomaly_canny_pil)
        if defective_img is None:
            print("❌ 缺陷图像生成失败，停止执行")
            return
        
        # 第四步：创建最终对比图
        comparison_path, single_row_path = create_final_comparison(
            INPUT_IMG, anomaly_mask_path, anomaly_canny_path, defective_image_path
        )
        
        print("="*80)
        print("🎉 单张异常图像生成流程执行完成！")
        print("="*80)
        print(f"📁 输出目录: {OUTPUT_DIR}")
        print("📋 生成的文件:")
        print(f"  0. 正常Canny: 00_normal_canny.png")
        print(f"  1. 约束掩码: 01_constraint_mask.png")
        print(f"  2. 异常掩码: 02_anomaly_mask_0.8.png")  
        print(f"  3. 异常Canny: 03_anomaly_canny.png")
        print(f"  4. 缺陷图像: 04_defective_image.png")
        print(f"  5. 2x2对比图: 05_final_comparison.png")
        print(f"  6. 1x4对比图: 06_single_row_comparison.png")
        print("="*80)

if __name__ == "__main__":
    main()