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


def save_image(image, filepath):
    """保存图像"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, image)


def main():
    parser = argparse.ArgumentParser(description='人脸杂交生成器')
    parser.add_argument('--num-sources', type=int, default=50)
    parser.add_argument('--num-hybrids', type=int, default=10)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--mutation-rate', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str, default='hybrid_results')
    args = parser.parse_args()
    
    print("="*60)
    print("🎭 人脸杂交生成器")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("extracted_faces", exist_ok=True)
    os.makedirs("source_images", exist_ok=True)
    
    try:
        # 1. 采集人脸
        print("\n📸 采集人脸图片...")
        collector = FaceDataCollector(cache_dir="source_images")
        source_images = collector.collect(num_images=args.num_sources)
        print(f"✅ 采集到 {len(source_images)} 张图片")
        
        # 2. 检测人脸并提取
        print("\n🔍 检测并提取人脸...")
        detector = FaceDetector(min_face_size=80)
        faces = []
        face_images = []  # 保存原始人脸图像用于融合
        
        for i, img_path in enumerate(source_images):
            face = detector.detect_and_extract(img_path)
            if face is not None:
                faces.append(face)
                # 保存提取的人脸
                face_img_path = f"extracted_faces/face_{len(faces):03d}.png"
                cv2.imwrite(face_img_path, face)
                face_images.append(face)
            if (i+1) % 20 == 0:
                print(f"   进度: {i+1}/{len(source_images)}")
        
        print(f"✅ 成功提取 {len(faces)} 个人脸")
        
        if len(faces) < 2:
            raise Exception("提取的人脸不足2个")
        
        # 3. 编码特征
        print("\n🧬 编码人脸特征...")
        encoder = FaceEncoder()
        face_genes = []
        
        for i, face in enumerate(faces):
            gene = encoder.encode(face)
            if gene is not None:
                # 保存原始图像用于后续融合
                gene['image'] = face
                face_genes.append(gene)
        
        print(f"✅ 成功编码 {len(face_genes)} 个人脸")
        
        # 4. 杂交
        print("\n🧬 进行人脸杂交...")
        hybridizer = FaceHybridizer(
            mutation_rate=args.mutation_rate,
            generations=args.generations
        )
        # 设置参考人脸用于融合
        hybridizer.set_reference_faces(face_genes)
        
        hybrids = hybridizer.hybridize(
            face_genes=face_genes,
            num_hybrids=args.num_hybrids
        )
        
        print(f"✅ 生成了 {len(hybrids)} 个杂交人脸")
        
        # 5. 保存结果
        print("\n💾 保存杂交结果...")
        
        for i, hybrid in enumerate(hybrids):
            img = hybridizer.decode_to_image(hybrid)
            filename = f"{args.output_dir}/hybrid_{i+1:03d}_gen{hybrid['generation']}.png"
            save_image(img, filename)
            
            # 保存元数据
            metadata = {
                'id': i+1,
                'generation': hybrid['generation'],
                'parents': hybrid['parents'],
                'mutation': hybrid.get('mutation', False)
            }
            with open(f"{args.output_dir}/hybrid_{i+1:03d}_meta.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"\n✅ 完成！生成了 {len(hybrids)} 张杂交人脸")
        print(f"📁 保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()