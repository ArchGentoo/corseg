import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
# ------------------------------
# 2. 血管重连阶段
# ------------------------------
# ---------------------------------------------
# 1. 辅助函数和数据结构定义
# ---------------------------------------------

def get_neighbors(point):
    """
    获取当前点的 26 个邻居点。
    :param point: 当前点坐标，numpy 数组，形状为 (3,)
    :return: 邻居点列表，每个元素是形状为 (3,) 的 numpy 数组
    """
    neighbors = []
    x, y, z = point
    # 遍历相邻的偏移量
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # 跳过自身
                neighbor = np.array([x + dx, y + dy, z + dz])
                neighbors.append(neighbor)
    return neighbors

def extract_patch(image, center, l):
    """
    从图像中提取以 center 为中心，大小为 l*l*l 的立方体 patch。
    :param image: 3D 图像，numpy 数组
    :param center: patch 中心点坐标，numpy 数组，形状为 (3,)
    :param l: patch 的边长（必须为奇数）
    :return: 提取的 patch，numpy 数组，形状为 (l, l, l)
    """
    assert l % 2 == 1, "Patch size l must be odd."
    r = l // 2
    x, y, z = center.astype(int)
    # 计算切片范围
    x_min, x_max = x - r, x + r + 1
    y_min, y_max = y - r, y + r + 1
    z_min, z_max = z - r, z + r + 1
    # 检查边界条件，超出图像范围则填充 0
    pad_width = [
        (max(0, -x_min), max(0, x_max - image.shape[0])),
        (max(0, -y_min), max(0, y_max - image.shape[1])),
        (max(0, -z_min), max(0, z_max - image.shape[2]))
    ]
    image_padded = np.pad(image, pad_width, mode='constant', constant_values=0)
    x_min_p, x_max_p = x_min + pad_width[0][0], x_max + pad_width[0][0]
    y_min_p, y_max_p = y_min + pad_width[1][0], y_max + pad_width[1][0]
    z_min_p, z_max_p = z_min + pad_width[2][0], z_max + pad_width[2][0]
    patch = image_padded[x_min_p:x_max_p, y_min_p:y_max_p, z_min_p:z_max_p]
    return patch

def cosine_similarity(u, v):
    """
    计算两个向量的余弦相似度。
    :param u: 向量 u，numpy 数组
    :param v: 向量 v，numpy 数组
    :return: 余弦相似度，标量
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    cos_sim = np.dot(u, v) / (norm_u * norm_v)
    return cos_sim

# ---------------------------------------------
# 2. DPC Walk 算法实现
# ---------------------------------------------

class DPCWalk:
    def __init__(self, image, CFC, omega=5.0, patch_size=9, low_P_threshold=0.1):
        """
        初始化 DPCWalk 类。
        :param image: 3D 图像数据，numpy 数组
        :param CFC: 预训练的中心线分类器，支持 predict_proba 方法
        :param omega: P 项的权重 ω
        :param patch_size: 提取 patch 的大小 l（必须为奇数）
        :param low_P_threshold: P 的低阈值，当环境中 P 极低时放大 ω
        """
        self.image = image
        self.CFC = CFC
        self.omega = omega
        self.patch_size = patch_size
        self.low_P_threshold = low_P_threshold

    def dpc_walk(self, CLj, Vi, max_steps=1000):
        """
        对单个断裂的中心线 CLj，使用 DPC 算法连接到 Vi。
        :param CLj: 断裂的中心线，numpy 数组，形状为 (n, 3)
        :param Vi: 主要的中心线，numpy 数组，形状为 (m, 3)
        :param max_steps: 最大步数，防止无限循环
        :return: 连接后的路径，如果连接失败则返回 None
        """
        # 初始化
        q1 = CLj[0]  # CLj 的头部
        pm = Vi[-1]  # Vi 的尾部
        A = q1.astype(int)  # 当前点 A
        path = [A.copy()]  # 存储路径
        o_minus_1 = None  # o_{-1}
        o_minus_2 = None  # o_{-2}
        v = pm - q1  # 全局方向向量

        # 记录连续低概率的次数
        low_P_count = 0
        max_low_P_count = 10  # 连续低概率的最大次数

        # 主循环
        for step in range(max_steps):
            # 获取当前点的 26 个邻居
            neighbors = get_neighbors(A)
            DPC_values = []
            candidates = []

            for Ak in neighbors:
                Ak = Ak.astype(int)
                # 检查 Ak 是否在图像范围内
                if not (0 <= Ak[0] < self.image.shape[0] and
                        0 <= Ak[1] < self.image.shape[1] and
                        0 <= Ak[2] < self.image.shape[2]):
                    continue

                # 计算 D(Ak)
                D_Ak = -np.linalg.norm(Ak - pm) ** 2

                # 计算 P(Ak)
                patch = extract_patch(self.image, Ak, self.patch_size)
                patch_flatten = patch.flatten()
                # 预测概率，需要确保输入的形状正确
                P_Ak = self.CFC.predict_proba([patch_flatten])[0][1]

                # 检查是否处于极低概率环境
                omega = self.omega
                if P_Ak < self.low_P_threshold:
                    omega *= 10  # 放大 ω

                # 计算 ok
                ok = Ak - A

                # 计算 C(Ak)
                if o_minus_1 is not None and o_minus_2 is not None:
                    cos_o = cosine_similarity(o_minus_1, o_minus_2)
                    if cos_o <= 0.5:
                        C_Ak = cosine_similarity(ok, o_minus_1) + cosine_similarity(ok, o_minus_2)
                    else:
                        C_Ak = 0  # 避免直线，忽略 C(Ak)
                else:
                    C_Ak = 0  # 初始化阶段，无法计算 C(Ak)

                # 计算 DPC(Ak)
                DPC_Ak = D_Ak + omega * P_Ak + C_Ak

                # 检查 cos(ok, v) ≥ 0
                cos_ok_v = cosine_similarity(ok, v)
                if cos_ok_v >= 0:
                    DPC_values.append(DPC_Ak)
                    candidates.append(Ak)

            # 如果没有候选点，终止
            if not candidates:
                print("No candidates available, stopping.")
                return None

            # 选择具有最大 DPC 值的候选点
            max_idx = np.argmax(DPC_values)
            Anext = candidates[max_idx]

            # 更新低概率计数器
            max_P_Ak = self.CFC.predict_proba([extract_patch(self.image, Anext, self.patch_size).flatten()])[0][1]
            if max_P_Ak < self.low_P_threshold:
                low_P_count += 1
            else:
                low_P_count = 0  # 重置计数器

            # 检查连续低概率终止条件
            if low_P_count >= max_low_P_count:
                print("Continuous low probabilities, stopping.")
                return None

            # 更新路径和方向向量
            path.append(Anext.copy())
            if o_minus_1 is not None:
                o_minus_2 = o_minus_1.copy()
            o_minus_1 = Anext - A
            A = Anext  # 更新当前点

            # 检查是否到达 pm
            if np.array_equal(A, pm):
                print("Reached the target point pm.")
                return np.array(path)

        print("Exceeded maximum steps, stopping.")
        return None

# ---------------------------------------------
# 3. 示例使用 DPCWalk 进行中心线重连
# ---------------------------------------------

def example_usage():
    # 假设已经有 3D 图像数据 image，以及中心线数据 Vi 和 CLj
    image = np.random.rand(100, 100, 100)  # 示例图像数据
    Vi = np.array([[50, 50, i] for i in range(50, 100)])  # 示例主要中心线 Vi
    CLj = np.array([[50 + i, 50, 49 - i] for i in range(10)])  # 示例断裂的中心线 CLj

    # 初始化中心线分类器 CFC，这里使用随机森林作为示例
    # 实际应用中，需要使用训练好的模型
    CFC = RandomForestClassifier()
    # 为了演示，使用随机数据训练分类器
    X_train = np.random.rand(100, 27)  # 假设 patch 展平后有 27 个特征
    y_train = np.random.randint(0, 2, 100)
    CFC.fit(X_train, y_train)

    # 初始化 DPCWalk 对象
    dpc_walker = DPCWalk(image=image, CFC=CFC, omega=5.0, patch_size=3, low_P_threshold=0.1)

    # 执行 DPC Walk
    connected_path = dpc_walker.dpc_walk(CLj=CLj, Vi=Vi)

    if connected_path is not None:
        print("Successfully connected the centerlines.")
        # 可以在此处将连接后的路径与 Vi 合并
        # 例如：
        Vi_connected = np.vstack((Vi, connected_path))
    else:
        print("Failed to connect the centerlines.")


# 使用示例
def reconnect_centerlines(broken_centerlines, main_centerline):
    dpc_walk = DPCWalk(omega=1.0)
    connected_centerlines = []

    for CLj in broken_centerlines:
        CLj_head = CLj[0]
        pm = main_centerline[-1]  # 假设目标点为主中心线的末尾点

        path = dpc_walk.walk(CLj_head, pm)
        connected_centerlines.append(path)

    # 合并所有中心线
    complete_centerline = main_centerline + connected_centerlines
    return complete_centerline

# ------------------------------
# 3. 血管重建阶段
# ------------------------------

class LevelSetReconstruction:
    def __init__(self, alpha=1.0, beta=1.0, sigma=1.0):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def level_set_energy(self, I, phi):
        """
        I: 图像
        phi: 水平集函数
        """
        H_phi = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / self.sigma))
        delta_phi = (1 / (np.pi * self.sigma)) / (1 + (phi / self.sigma) ** 2)

        # 计算局部平均值 μ(x, φ)
        K_sigma = gaussian_filter(np.ones_like(I), sigma=self.sigma)
        numerator = gaussian_filter(I * H_phi, sigma=self.sigma)
        denominator = gaussian_filter(H_phi, sigma=self.sigma)
        mu = numerator / (denominator + 1e-6)

        # 图像梯度
        grad_I = np.gradient(I)
        grad_I_norm = np.sqrt(grad_I[0] ** 2 + grad_I[1] ** 2 + grad_I[2] ** 2)
        g = 1 / (1 + grad_I_norm ** 2)

        # 能量函数
        E = self.alpha * ((I - mu) ** 2 * H_phi).sum() + self.beta * (g * delta_phi * np.abs(np.gradient(H_phi))).sum()
        return E

    def reconstruct(self, centerlines, image):
        """
        centerlines: 重连后的中心线列表
        image: 原始图像
        """
        # 初始化水平集函数 phi
        phi = np.ones_like(image) * -1  # 外部为 -1
        for cl in centerlines:
            for point in cl:
                phi[tuple(point)] = 1  # 内部为 1

        # 迭代优化水平集函数
        for iter in range(num_iterations):
            E = self.level_set_energy(image, phi)
            # 更新 phi（梯度下降，示例实现）
            phi -= learning_rate * E

        # 提取等值面，生成 3D 模型（可使用 marching cubes 算法）
        vertices, faces, normals, values = measure.marching_cubes(phi, level=0)
        return vertices, faces

# 使用示例
def reconstruct_vessels(centerlines, image):
    level_set = LevelSetReconstruction(alpha=1.0, beta=1.0, sigma=1.0)
    vertices, faces = level_set.reconstruct(centerlines, image)
    # 可视化或保存模型
    return vertices, faces

# ------------------------------
# 整体流程整合
# ------------------------------

def main():
    # 加载 CCTA 图像
    ccta_image = load_ccta_image()

    # 第一步：血管分割
    model = SimpleUNet().to(device)
    segmentation = model(ccta_image)
    # 使用 NSDT Soft-ClDice 损失函数训练模型（在训练过程中完成）

    # 第二步：血管重连
    broken_centerlines = extract_broken_centerlines(segmentation)
    main_centerline = extract_main_centerline(segmentation)
    complete_centerlines = reconnect_centerlines(broken_centerlines, main_centerline)

    # 第三步：血管重建
    vertices, faces = reconstruct_vessels(complete_centerlines, ccta_image)

    # 输出最终的冠状动脉树模型
    save_model(vertices, faces)

if __name__ == "__main__":
    main()

# 运行示例
