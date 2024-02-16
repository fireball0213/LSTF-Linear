import numpy as np

class SPIRITModel:
    def __init__(self,args):
        """
        初始化SPIRIT算法参数
        :param window_size: 窗口大小，即每次处理的数据维度
        :param rank: 目标维度，即数据降维后的维度
        :param spirit_alpha: 遗忘因子，用于调整特征向量更新的速度,默认0.98
        """
        self.window_size = args.channels
        self.rank = args.rank
        self.alpha = args.spirit_alpha
        # 初始化矩阵P和权重向量w
        self.P = np.eye(self.window_size) * 0.001  # 小的正值初始化
        self.w = np.random.rand(self.window_size, self.rank)
        self.update_threshold = 0.1  # 更新的最大阈值，可以根据需要调整

    def update(self, x):
        """
        根据新的观测向量x更新模型参数。
        :param x: 新的观测向量。
        """

        x = x.reshape(-1, 1)  # 确保x是列向量
        p_x = self.P @ x
        alpha_inv = 1.0 / self.alpha
        denominator = 1 + (x.T @ p_x) * alpha_inv
        self.P = alpha_inv * self.P - (p_x @ p_x.T) * (alpha_inv ** 2 / denominator)

        # 更新权重w，加入数值稳定性措施
        projection = self.w.T @ x
        err = x - self.w @ projection
        update_step = self.P @ err @ projection.T

        # 限制更新幅度
        update_step = np.clip(update_step, -self.update_threshold, self.update_threshold)
        self.w += update_step

    def fit_transform(self, X):
        """
        X：训练数据，形状为(n_samples, channel)
        """
        self.original_dim = X.shape[1]
        for x in X:
            self.update(x)
        return self.transform(X)

    def transform(self, X):
        """
        使用训练好的SPIRIT模型对数据X进行降维。
        """
        # 确保X是二维数组
        if X.ndim == 1:
            X = X.reshape(1, -1)
        transformed = X @ self.w
        return transformed

    def inverse_transform(self, X_transformed):
        """
        将降维后的数据逆变换回原始数据空间。
        """
        if self.original_dim is None:
            raise ValueError("The SPIRIT model must be fit before calling inverse_transform.")
        # 使用伪逆进行逆变换
        w_pinv = np.linalg.pinv(self.w)
        return X_transformed @ w_pinv