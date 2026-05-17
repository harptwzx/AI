#!/usr/bin/env python3
"""
人脸杂交主程序
"""
import argparse
import os
import json
import sys
import cv2
import numpy as np

from data_collector import FaceDataCollector
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_hybridizer import FaceHybridizer
from image_utils import save_image, create_collage


def parse_args():
    parser = argparse.ArgumentParser(description='人脸杂交生成器')
    parser.add_argument('--num-sources', type=int, default=50, help='采集源图片数量')
    parser.add_argument('--num-hybrids', type=int, default=10, help='生成杂交数量')
    parser.add_argument('--generations', type=int, default=3, help='杂交代数')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='变异率')
    parser.add_argument('--source-type', type=str, default='random', help='人脸来源类型')
    parser.add_argument('--min-face-size', type=int, default=50, help='最小人脸尺寸')
    parser.add_argument('--output-dir', type=str, default='hybrid_results', help='输出目录')
    parser.add_argument('--output-json', action='store_true', help='输出JSON格式')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("🎭 人脸杂交生成器")
    print("="*60)
    
    # 创建输出目录（使用相对路径，不要用../）
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("extracted_faces", exist_ok=True)
    os.makedirs("source_images", exist_ok=True)
    
    result = {
        'success': False,
        'source_faces': 0,
        'hybrid_faces': [],
        'error': None
    }
    
    try:
        # 步骤1: 采集人脸图片
        print("\n📸 步骤 1/5: 采集人脸图片...")
        collector = FaceDataCollector(cache_dir="source_images")
        source_images = collector.collect(
            num_images=args.num_sources,
            source_type=args.source_type
        )
        print(f"✅ 采集到 {len(source_images)} 张图片")
        
        # 步骤2: 检测并提取人脸
        print("\n🔍 步骤 2/5: 检测并提取人脸...")
        detector = FaceDetector(min_face_size=args.min_face_size)
        faces = []
        for i, img_path in enumerate(source_images):
            face = detector.detect_and_extract(img_path)
            if face is not None:
                faces.append(face)
            if (i+1) % 10 == 0:
                print(f"   进度: {i+1}/{len(source_images)}")
        
        print(f"\n✅ 成功提取 {len(faces)} 个人脸")
        
        if len(faces) < 2:
            raise Exception("提取的人脸不足2个，无法进行杂交")
        
        # 步骤3: 编码人脸特征
        print("\n🧬 步骤 3/5: 编码人脸特征（基因）...")
        encoder = FaceEncoder()
        face_genes = []
        for face in faces:
            gene = encoder.encode(face)
            if gene is not None:
                face_genes.append(gene)
        
        print(f"✅ 成功编码 {len(face_genes)} 个人脸基因")
        
        # 步骤4: 人脸杂交
        print("\n🧬 步骤 4/5: 进行人脸杂交...")
        hybridizer = FaceHybridizer(
            mutation_rate=args.mutation_rate,
            generations=args.generations
        )
        
        hybrids = hybridizer.hybridize(
            face_genes=face_genes,
            num_hybrids=args.num_hybrids
        )
        
        print(f"✅ 生成了 {len(hybrids)} 个杂交人脸")
        
        # 步骤5: 保存结果
        print("\n💾 步骤 5/5: 保存杂交结果...")
        saved_paths = []
        for i, hybrid in enumerate(hybrids):
            img = hybridizer.decode_to_image(hybrid)
            filename = f"{args.output_dir}/hybrid_{i+1:03d}_gen{hybrid['generation']}.png"
            save_image(img, filename)
            saved_paths.append(filename)
            
            # 保存元数据
            metadata = {
                'id': i+1,
                'generation': hybrid['generation'],
                'parents': hybrid['parents'],
                'mutation_applied': hybrid.get('mutation', False),
                'features': hybrid['features'][:10]
            }
            
            meta_file = f"{args.output_dir}/hybrid_{i+1:03d}_meta.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # 创建拼接图
        if len(saved_paths) > 0:
            collage_path = f"{args.output_dir}/all_hybrids_collage.png"
            create_collage(saved_paths, collage_path, cols=5)
        
        # 保存摘要
        summary = f"""
人脸杂交摘要
============

参数配置:
- 源图片数量: {args.num_sources}
- 成功提取人脸: {len(faces)}
- 成功编码基因: {len(face_genes)}
- 生成杂交数量: {len(hybrids)}
- 杂交代数: {args.generations}
- 变异率: {args.mutation_rate}

输出文件:
- 杂交人脸: {len(saved_paths)} 张
- 元数据: {len(saved_paths)} 个JSON文件

所有文件保存在: {args.output_dir}
        """
        
        with open(f"{args.output_dir}/summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
        
        result['success'] = True
        result['source_faces'] = len(faces)
        result['hybrid_faces'] = saved_paths
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        result['error'] = str(e)
    
    if args.output_json:
        print(json.dumps(result, ensure_ascii=False))
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())