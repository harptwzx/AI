"""
人脸杂交核心算法
实现特征的交叉、变异和重建
"""
import numpy as np
import random
import cv2
from scipy.interpolate import griddata


class FaceHybridizer:
    """人脸杂交器"""

    def __init__(self, mutation_rate=0.1, generations=3):
        self.mutation_rate = mutation_rate
        self.generations = generations

    def hybridize(self, face_genes, num_hybrids=10):
        """
        进行人脸杂交
        
        Args:
            face_genes: 人脸基因列表
            num_hybrids: 需要生成的杂交数量
            
        Returns:
            杂交后代列表
        """
        hybrids = []

        for gen in range(self.generations):
            print(f"   第 {gen+1} 代杂交...")

            if gen == 0:
                # 第一代：使用原始基因
                parents_pool = face_genes
            else:
                # 后续：使用上一代杂交结果
                parents_pool = [h['gene'] for h in hybrids[-num_hybrids:]]

            gen_hybrids = []

            for i in range(num_hybrids):
                # 选择父母
                parent1, parent2 = random.sample(parents_pool, 2)

                # 交叉繁殖
                child_gene = self._crossover(parent1['vector'], parent2['vector'])

                # 变异
                if random.random() < self.mutation_rate:
                    child_gene = self._mutate(child_gene)

                # 记录杂交信息
                hybrid = {
                    'gene': {'vector': child_gene, 'dimension': len(child_gene)},
                    'generation': gen + 1,
                    'parents': [parent1['hash'], parent2['hash']],
                    'mutation': random.random() < self.mutation_rate,
                    'features': child_gene[:20].tolist()  # 保存部分特征
                }

                gen_hybrids.append(hybrid)

            hybrids.extend(gen_hybrids)

        return hybrids

    def _crossover(self, gene1, gene2):
        """
        基因交叉（单点交叉）
        """
        child = gene1.copy()

        # 随机选择交叉点
        crossover_point = random.randint(0, len(gene1))

        # 交换基因片段
        child[crossover_point:] = gene2[crossover_point:]

        return child

    def _mutate(self, gene, mutation_strength=0.1):
        """
        基因变异
        """
        mutated = gene.copy()

        # 随机选择变异位置
        mutation_indices = random.sample(range(len(gene)), 
                                        k=int(len(gene) * self.mutation_rate))

        for idx in mutation_indices:
            # 添加高斯噪声
            mutated[idx] += np.random.normal(0, mutation_strength)

        # 重新归一化
        mutated = mutated / (np.linalg.norm(mutated) + 1e-8)

        return mutated

    def decode_to_image(self, hybrid, target_size=(128, 128)):
        """
        将基因解码为人脸图像
        
        这是一个简化的解码器，实际应用中使用生成对抗网络(GAN)
        这里使用特征重建的方法
        """
        gene = hybrid['gene']['vector']

        # 创建基础图像
        img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        # 使用基因特征来影响图像生成
        # 这里简化为根据基因值生成颜色和纹理

        # 1. 基础颜色（由基因前3个值决定）
        base_color = (int((gene[0] + 1) * 127.5),
                     int((gene[1] + 1) * 127.5),
                     int((gene[2] + 1) * 127.5))

        # 填充背景
        img[:] = base_color

        # 2. 生成人脸区域（根据基因的4-6个值决定位置）
        face_center_y = int((gene[3] + 1) * target_size[0] * 0.3 + target_size[0] * 0.2)
        face_center_x = int((gene[4] + 1) * target_size[1] * 0.5 + target_size[1] * 0.25)
        face_size = int(((gene[5] + 1) / 2) * min(target_size) * 0.6 + min(target_size) * 0.2)

        # 绘制椭圆脸型
        cv2.ellipse(img, (face_center_x, face_center_y), 
                   (face_size, int(face_size*1.2)), 0, 0, 360,
                   (220, 180, 140), -1)

        # 3. 眼睛（基因7-10）
        eye_y = face_center_y - int(face_size * 0.2)
        eye_size = int(face_size * 0.15)

        eye_distance = int(face_size * 0.3)
        left_eye_x = face_center_x - eye_distance
        right_eye_x = face_center_x + eye_distance

        # 眼睛颜色
        eye_color = (255, 255, 255)
        pupil_color = (0, 0, 0)

        cv2.circle(img, (left_eye_x, eye_y), eye_size, eye_color, -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_size, eye_color, -1)

        # 瞳孔
        pupil_size = int(eye_size * 0.4)
        cv2.circle(img, (left_eye_x, eye_y), pupil_size, pupil_color, -1)
        cv2.circle(img, (right_eye_x, eye_y), pupil_size, pupil_color, -1)

        # 4. 鼻子（基因11）
        nose_y = face_center_y + int(face_size * 0.1)
        nose_size = int(face_size * 0.15)
        cv2.ellipse(img, (face_center_x, nose_y), 
                   (nose_size, nose_size//2), 0, 0, 360,
                   (200, 150, 100), -1)

        # 5. 嘴巴（基因12）
        mouth_y = face_center_y + int(face_size * 0.35)
        mouth_width = int(face_size * 0.4)
        mouth_height = int(face_size * 0.15)

        # 嘴巴形状由基因12决定
        smile_curve = (gene[12] + 1) / 2  # 0-1之间
        start_angle = 180
        end_angle = 360 - int(180 * smile_curve)

        cv2.ellipse(img, (face_center_x, mouth_y), 
                   (mouth_width//2, mouth_height//2), 0, 
                   start_angle, end_angle, (100, 50, 50), -1)

        # 6. 添加纹理噪声（基因其他值影响）
        for y in range(target_size[0]):
            for x in range(target_size[1]):
                # 使用基因值作为噪声权重
                noise_idx = (y * target_size[1] + x) % min(50, len(gene))
                noise_strength = int((gene[noise_idx] + 1) * 20)

                if noise_strength > 0:
                    r = min(255, img[y, x, 0] + random.randint(-noise_strength, noise_strength))
                    g = min(255, img[y, x, 1] + random.randint(-noise_strength, noise_strength))
                    b = min(255, img[y, x, 2] + random.randint(-noise_strength, noise_strength))
                    img[y, x] = [max(0, r), max(0, g), max(0, b)]

        return img