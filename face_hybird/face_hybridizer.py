"""
人脸杂交核心算法 - 真实的图像融合
"""
import numpy as np
import random
import cv2
from scipy import interpolate


class FaceHybridizer:
    """人脸杂交器 - 使用图像融合技术"""
    
    def __init__(self, mutation_rate=0.1, generations=3):
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.reference_faces = []  # 存储原始人脸图像用于融合
    
    def set_reference_faces(self, faces):
        """设置参考人脸图像（用于图像融合）"""
        self.reference_faces = faces
    
    def hybridize(self, face_genes, num_hybrids=10):
        """进行人脸杂交"""
        hybrids = []
        
        for gen in range(self.generations):
            print(f"   第 {gen+1} 代杂交...")
            
            if gen == 0:
                parents_pool = face_genes
            else:
                parents_pool = [h['gene'] for h in hybrids[-num_hybrids:]]
            
            gen_hybrids = []
            
            for i in range(num_hybrids):
                # 选择父母
                parent1, parent2 = random.sample(parents_pool, 2)
                
                # 特征交叉
                child_gene = self._crossover(parent1['vector'], parent2['vector'])
                
                # 变异
                mutated = False
                if random.random() < self.mutation_rate:
                    child_gene = self._mutate(child_gene)
                    mutated = True
                
                # 记录杂交信息
                hybrid = {
                    'gene': {'vector': child_gene, 'dimension': len(child_gene)},
                    'generation': gen + 1,
                    'parents': [parent1.get('hash', 'unknown'), parent2.get('hash', 'unknown')],
                    'mutation': mutated,
                    'features': child_gene[:20].tolist()
                }
                
                gen_hybrids.append(hybrid)
            
            hybrids.extend(gen_hybrids)
        
        return hybrids
    
    def _crossover(self, gene1, gene2):
        """基因交叉"""
        child = gene1.copy()
        crossover_point = random.randint(0, len(gene1))
        child[crossover_point:] = gene2[crossover_point:]
        return child
    
    def _mutate(self, gene, mutation_strength=0.15):
        """基因变异"""
        mutated = gene.copy()
        mutation_indices = random.sample(range(len(gene)), 
                                        k=max(1, int(len(gene) * self.mutation_rate)))
        
        for idx in mutation_indices:
            mutated[idx] += np.random.normal(0, mutation_strength)
        
        # 重新归一化
        norm = np.linalg.norm(mutated) + 1e-8
        mutated = mutated / norm * np.sqrt(len(mutated))
        mutated = np.clip(mutated, -1, 1)
        
        return mutated
    
    def decode_to_image(self, hybrid, target_size=(128, 128)):
        """将基因解码为真实人脸图像 - 使用特征重构和图像融合"""
        
        if len(self.reference_faces) >= 2:
            # 方法1: 使用参考人脸进行特征插值
            return self._blend_faces(hybrid['gene']['vector'], target_size)
        else:
            # 方法2: 从特征重建
            return self._reconstruct_from_features(hybrid['gene']['vector'], target_size)
    
    def _blend_faces(self, gene, target_size):
        """基于特征向量混合多张人脸"""
        if len(self.reference_faces) < 2:
            return self._reconstruct_from_features(gene, target_size)
        
        # 计算每张人脸的权重（基于基因相似度）
        weights = []
        for ref_gene in self.reference_faces[:10]:  # 最多混合10张
            # 计算相似度
            similarity = np.dot(gene, ref_gene['vector'])
            weights.append(max(0, similarity))
        
        if sum(weights) == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = np.array(weights) / sum(weights)
        
        # 加权融合图像
        blended = None
        for i, ref_face in enumerate(self.reference_faces[:10]):
            if 'image' not in ref_face:
                continue
            img = ref_face['image']
            img = cv2.resize(img, target_size)
            
            if blended is None:
                blended = img.astype(np.float32) * weights[i]
            else:
                blended += img.astype(np.float32) * weights[i]
        
        if blended is not None:
            result = blended.astype(np.uint8)
            # 添加微小的随机变化（模拟变异）
            noise = np.random.randint(-5, 5, result.shape, dtype=np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return result
        
        return self._reconstruct_from_features(gene, target_size)
    
    def _reconstruct_from_features(self, gene, target_size):
        """从特征向量重建人脸图像"""
        # 创建基础图像
        img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # 使用基因特征生成纹理和颜色
        # 将基因值映射到0-255范围
        gene_normalized = ((gene[:128] + 1) / 2 * 255).astype(np.uint8)
        
        # 创建噪声纹理
        texture = np.zeros(target_size, dtype=np.uint8)
        for i in range(min(64, len(gene_normalized))):
            freq = (gene_normalized[i] % 10) + 1
            phase = gene_normalized[i] % 360
            for y in range(target_size[0]):
                for x in range(target_size[1]):
                    val = (np.sin(x * freq * 0.1 + phase) + 
                           np.cos(y * freq * 0.1 + phase)) * 128 + 127
                    texture[y, x] = (texture[y, x] + val) // 2
        
        # 生成肤色
        skin_r = 200 + (gene[0] * 55)
        skin_g = 160 + (gene[1] * 60)
        skin_b = 140 + (gene[2] * 50)
        
        skin_color = np.array([skin_b, skin_g, skin_r], dtype=np.uint8)
        
        # 创建面部区域（椭圆形）
        mask = np.zeros(target_size, dtype=np.uint8)
        center = (target_size[1]//2, target_size[0]//2)
        axes = (int(target_size[1] * (0.3 + gene[3] * 0.1)),
                int(target_size[0] * (0.4 + gene[4] * 0.1)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # 应用肤色
        for c in range(3):
            channel = img[:,:,c]
            channel[mask > 0] = skin_color[c]
            img[:,:,c] = channel
        
        # 添加纹理
        img = cv2.addWeighted(img, 0.7, cv2.cvtColor(cv2.merge([texture]*3), cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # 高斯模糊使图像平滑
        img = cv2.GaussianBlur(img, (3, 3), 1)
        
        return img