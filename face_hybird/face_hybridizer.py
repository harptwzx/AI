"""
人脸杂交核心算法
"""
import numpy as np
import random
import cv2


class FaceHybridizer:
    """人脸杂交器"""
    
    def __init__(self, mutation_rate=0.1, generations=3):
        self.mutation_rate = mutation_rate
        self.generations = generations
    
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
                
                # 交叉繁殖
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
    
    def _mutate(self, gene, mutation_strength=0.1):
        """基因变异"""
        mutated = gene.copy()
        mutation_indices = random.sample(range(len(gene)), 
                                        k=max(1, int(len(gene) * self.mutation_rate)))
        
        for idx in mutation_indices:
            mutated[idx] += np.random.normal(0, mutation_strength)
        
        # 重新归一化
        norm = np.linalg.norm(mutated) + 1e-8
        mutated = mutated / norm
        
        return mutated
    
    def decode_to_image(self, hybrid, target_size=(128, 128)):
        """将基因解码为人脸图像"""
        gene = hybrid['gene']['vector']
        
        # 创建基础图像
        img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # 基础颜色
        base_color = (int((gene[0] + 1) * 127.5),
                     int((gene[1] + 1) * 127.5),
                     int((gene[2] + 1) * 127.5))
        
        img[:] = base_color
        
        # 人脸区域
        face_center_y = int((gene[3] + 1) * target_size[0] * 0.3 + target_size[0] * 0.2)
        face_center_x = int((gene[4] + 1) * target_size[1] * 0.5 + target_size[1] * 0.25)
        face_size = int(((gene[5] + 1) / 2) * min(target_size) * 0.6 + min(target_size) * 0.2)
        
        # 脸型
        cv2.ellipse(img, (face_center_x, face_center_y), 
                   (face_size, int(face_size*1.2)), 0, 0, 360,
                   (220, 180, 140), -1)
        
        # 眼睛
        eye_y = face_center_y - int(face_size * 0.2)
        eye_size = int(face_size * 0.15)
        eye_distance = int(face_size * 0.3)
        left_eye_x = face_center_x - eye_distance
        right_eye_x = face_center_x + eye_distance
        
        cv2.circle(img, (left_eye_x, eye_y), eye_size, (255, 255, 255), -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_size, (255, 255, 255), -1)
        
        pupil_size = int(eye_size * 0.4)
        cv2.circle(img, (left_eye_x, eye_y), pupil_size, (0, 0, 0), -1)
        cv2.circle(img, (right_eye_x, eye_y), pupil_size, (0, 0, 0), -1)
        
        # 鼻子
        nose_y = face_center_y + int(face_size * 0.1)
        nose_size = int(face_size * 0.15)
        cv2.ellipse(img, (face_center_x, nose_y), 
                   (nose_size, nose_size//2), 0, 0, 360,
                   (200, 150, 100), -1)
        
        # 嘴巴
        mouth_y = face_center_y + int(face_size * 0.35)
        mouth_width = int(face_size * 0.4)
        mouth_height = int(face_size * 0.15)
        
        smile_curve = (gene[12] + 1) / 2
        start_angle = 180
        end_angle = 360 - int(180 * smile_curve)
        
        cv2.ellipse(img, (face_center_x, mouth_y), 
                   (mouth_width//2, mouth_height//2), 0, 
                   start_angle, end_angle, (100, 50, 50), -1)
        
        return img