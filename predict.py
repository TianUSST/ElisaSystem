import time

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import json
import argparse
from pathlib import Path
import numpy as np

def load_model(model_path, model_name, num_classes, device):
    """加载训练好的模型"""
    # 创建模型
    try:
        import timm
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
        raise RuntimeError(f"无法创建模型 {model_name}: {e}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint.get('model_state', checkpoint)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    model = model.to(device)
    model.eval()
    return model

def load_class_names(classes_json_path):
    """加载类别名称"""
    with open(classes_json_path, 'r', encoding='utf-8') as f:
        classes_data = json.load(f)
    return classes_data['classes']

def preprocess_image(image_path, image_size=224):
    """预处理单张图片"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    return input_tensor, image

def predict_single_image(model, image_tensor, class_names, device):
    """对单张图片进行推理"""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities.cpu().numpy()[0]

def add_text_to_image(image, text, position=(10, 10)):
    """在图片上添加文本"""
    draw = ImageDraw.Draw(image)
    
    # 尝试使用默认字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # 添加文本背景框
    # text_width, text_height = draw.textsize(text, font=font)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.rectangle([position, (position[0] + text_width, position[1] + text_height)], fill="black")
    draw.text(position, text, fill="white", font=font)
    
    return image

def predict_and_visualize_single(model, image_path, class_names, device, output_path=None, image_size=224, confidence_threshold=0.0):
    """对单张图片进行推理并可视化结果"""
    # 预处理图片
    input_tensor, original_image = preprocess_image(image_path, image_size)
    
    # 推理
    predicted_class, confidence, probabilities = predict_single_image(model, input_tensor, class_names, device)
    
    # 根据置信度阈值判断是否接受预测结果
    if confidence < confidence_threshold:
        result_text = f"Prediction below threshold: {predicted_class} ({confidence:.2f})"
    else:
        result_text = f"Predicted: {predicted_class} ({confidence:.2f})"
    
    result_image = add_text_to_image(original_image.copy(), result_text)
    
    # 保存或显示结果
    if output_path:
        result_image.save(output_path)
        print(f"结果已保存到: {output_path}")
    else:
        result_image.show()
    
    return predicted_class, confidence, probabilities

def predict_batch_images(model, image_paths, class_names, device, image_size=224, confidence_threshold=0.0):
    """批量推理"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    
    results = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
            
            # 根据置信度阈值判断是否接受预测结果
            if confidence_score < confidence_threshold:
                final_class = "Below Threshold"
            else:
                final_class = predicted_class
                
            results.append({
                'image_path': image_path,
                'predicted_class': final_class,
                'original_prediction': predicted_class,
                'confidence': confidence_score,
                'probabilities': probabilities.cpu().numpy()[0]
            })
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results

def visualize_batch_results(image_paths, results, output_dir):
    """可视化批量推理结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        if 'error' in result:
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
            if result['predicted_class'] == "Below Threshold":
                result_text = f"Prediction below threshold: {result['original_prediction']} ({result['confidence']:.2f})"
            else:
                result_text = f"Predicted: {result['predicted_class']} ({result['confidence']:.2f})"
            result_image = add_text_to_image(image.copy(), result_text)
            
            # 保存结果图片
            filename = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{filename}_result.jpg")
            result_image.save(output_path)
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存图片 {image_path} 的结果时出错: {e}")

if __name__ == '__main__':
    # 在此处配置运行参数
    class Args:
        pass
    
    args = Args()
    # 模型相关参数
    args.model_path = './output/run_resnet18_20250929_103249/best.pth'  # 模型路径
    args.classes_json = './output/run_resnet18_20250929_103249/classes.json'  # 类别名称JSON文件路径
    args.model_name = 'resnet18'  # 模型名称
    args.image_size = 224  # 输入图像尺寸
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备类型
    
    # 推理参数
    args.confidence_threshold = 0.5  # 置信度阈值
    
    # 输入输出参数
    args.single_image = None  # 单张图片路径
    args.batch_images = None  # 批量图片路径列表
    args.batch_dir = "example"  # 包含批量图片的目录
    args.output_path = None  # 单张图片输出路径
    args.batch_output_dir = './96batch_results'  # 批量图片输出目录
    
    # 加载类别名称
    class_names = load_class_names(args.classes_json)
    num_classes = len(class_names)
    
    # 加载模型
    device = torch.device(args.device)
    model = load_model(args.model_path, args.model_name, num_classes, device)
    print(f"模型已加载，使用设备: {device}")
    
    # 单张图片推理
    if args.single_image:
        if not os.path.exists(args.single_image):
            print(f"错误: 图片文件 {args.single_image} 不存在")
        else:
            predicted_class, confidence, probabilities = predict_and_visualize_single(
                model, args.single_image, class_names, device, args.output_path, args.image_size, args.confidence_threshold
            )
            print(f"预测结果: {predicted_class} (置信度: {confidence:.4f})")
            
            # 打印所有类别的概率
            print("\n各类别概率:")
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                print(f"  {class_name}: {prob:.4f}")
    start_time = time.time()
    # 批量图片推理
    image_paths = []
    if args.batch_images:
        image_paths.extend(args.batch_images)
    
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend([str(p) for p in batch_dir.glob(ext)])
    
    if image_paths:
        print(f"开始批量推理 {len(image_paths)} 张图片...")
        results = predict_batch_images(model, image_paths, class_names, device, args.image_size, args.confidence_threshold)
        end_time = time.time()
        print(f"批量推理完成，耗时: {end_time - start_time:.2f} 秒")

        # 输出结果
        for result in results:
            if 'error' in result:
                print(f"处理 {result['image_path']} 时出错: {result['error']}")
            else:
                if result['predicted_class'] == "Below Threshold":
                    print(f"{result['image_path']}: 低于阈值 - {result['original_prediction']} ({result['confidence']:.4f})")
                else:
                    print(f"{result['image_path']}: {result['predicted_class']} ({result['confidence']:.4f})")
        
        # 可视化结果
        if args.batch_output_dir:
            visualize_batch_results(image_paths, results, args.batch_output_dir)
        
        # 保存结果到CSV
        try:
            import pandas as pd
            df_data = []
            for result in results:
                if 'error' not in result:
                    df_data.append({
                        'image_path': result['image_path'],
                        'predicted_class': result['predicted_class'],
                        'original_prediction': result['original_prediction'],
                        'confidence': result['confidence']
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                csv_path = os.path.join(args.batch_output_dir or '.', 'batch_results.csv')
                df.to_csv(csv_path, index=False)
                print(f"批量推理结果已保存到: {csv_path}")
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")
