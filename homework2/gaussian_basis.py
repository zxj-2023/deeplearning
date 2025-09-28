import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(list(map(float, line.strip().split())))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)

def gaussian_basis(x, feature_num=10):
    """
    高斯基函数（径向基函数 RBF）
    
    使用高斯函数作为基函数，形式为 exp(-((x - μ_i) / σ)^2)。
    其中 μ_i 是第i个高斯函数的中心，σ 是标准差。
    高斯基函数能够很好地处理局部特征。
    
    Args:
        x (numpy.ndarray): 输入数据，形状为 (N,)，其中N是样本数量
        feature_num (int): 高斯基函数的数量，默认为10
                          即使用 feature_num 个不同中心的高斯函数
        
    Returns:
        numpy.ndarray: 高斯特征矩阵，形状为 (N, feature_num)
                      每列对应一个高斯基函数的输出
    """
    # 根据训练集x的范围(0-25)，均匀设置feature_num个高斯中心
    centers = np.linspace(0, 25, feature_num)
    
    # 设置标准差，影响高斯函数的宽度
    # 选择合适的σ值，使相邻高斯函数有适当的重叠
    sigma = (25 - 0) / (feature_num - 1) * 1.5  # 1.5倍的中心间距
    
    # 初始化结果矩阵，形状为 (N, feature_num)
    ret = np.zeros((len(x), feature_num))
    
    # 计算每个样本在各个高斯函数下的值
    for i in range(feature_num):
        # 高斯函数公式：exp(-((x - center_i) / sigma)^2)
        ret[:, i] = np.exp(-((x - centers[i]) / sigma) ** 2)
    
    return ret

def train_gaussian_regression(x_train, y_train, feature_num=15):
    """
    使用高斯基函数训练线性回归模型
    
    Args:
        x_train: 训练输入数据
        y_train: 训练目标数据
        feature_num: 高斯基函数的数量
    
    Returns:
        w: 训练好的权重参数
        feature_num: 高斯基函数数量
    """
    print(f"训练数据: {len(x_train)} 个样本")
    print(f"x范围: [{np.min(x_train):.2f}, {np.max(x_train):.2f}]")
    print(f"y范围: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    print(f"使用 {feature_num} 个高斯基函数")
    
    # 创建高斯基函数特征矩阵
    phi_gauss = gaussian_basis(x_train, feature_num=feature_num)
    # 添加偏置项
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi = np.concatenate([phi0, phi_gauss], axis=1)
    
    print(f"特征矩阵形状: {phi.shape}")
    print(f"特征矩阵数值范围: [{np.min(phi):.2e}, {np.max(phi):.2e}]")
    
    # 使用伪逆求解（最稳定的方法）
    try:
        w = np.linalg.pinv(phi) @ y_train
        print("成功使用伪逆求解")
        print(f"权重形状: {w.shape}")
    except Exception as e:
        print(f"伪逆求解失败: {e}")
        # 备用方案：正常最小二乘法
        try:
            w = np.linalg.inv(phi.T @ phi) @ phi.T @ y_train
            print("使用正常最小二乘法")
        except np.linalg.LinAlgError:
            raise RuntimeError("所有求解方法都失败了")
    
    return w, feature_num

def predict(x_test, w, feature_num):
    """
    使用训练好的模型进行预测
    
    Args:
        x_test: 测试输入数据
        w: 训练好的权重
        feature_num: 高斯基函数数量
    
    Returns:
        y_pred: 预测结果
    """
    # 创建相同的特征矩阵
    phi_gauss = gaussian_basis(x_test, feature_num=feature_num)
    phi0 = np.expand_dims(np.ones_like(x_test), axis=1)
    phi = np.concatenate([phi0, phi_gauss], axis=1)
    
    # 预测
    y_pred = np.dot(phi, w)
    return y_pred

def evaluate(y_true, y_pred):
    """评估模型性能"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def plot_results(x_train, y_train, x_test, y_test, y_pred):
    """绘制结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制训练数据
    plt.scatter(x_train, y_train, c='red', s=20, alpha=0.6, label='Training Data')
    
    # 绘制测试数据和预测结果
    # 按x排序以便绘制平滑曲线
    sorted_indices = np.argsort(x_test)
    x_test_sorted = x_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    plt.plot(x_test_sorted, y_test_sorted, 'b-', linewidth=2, label='True Values')
    plt.plot(x_test_sorted, y_pred_sorted, 'g--', linewidth=2, label='Predictions')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Basis Function Regression Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """主函数"""
    # 数据文件路径
    train_file = r'd:\代码\python\deeplearning\homework2\train.txt'
    test_file = r'd:\代码\python\deeplearning\homework2\test.txt'
    
    # 载入数据
    print("=== 载入数据 ===")
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    
    print(f"训练集: {x_train.shape[0]} 个样本")
    print(f"测试集: {x_test.shape[0]} 个样本")
    
    # 设置高斯基函数数量
    feature_num = 3
    
    print(f"\n=== 训练高斯基函数回归模型 ===")
    
    try:
        # 训练模型
        w, num_features = train_gaussian_regression(x_train, y_train, feature_num)
        
        # 预测
        y_train_pred = predict(x_train, w, num_features)
        y_test_pred = predict(x_test, w, num_features)
        
        # 评估
        train_rmse = evaluate(y_train, y_train_pred)
        test_rmse = evaluate(y_test, y_test_pred)
        
        print(f"\n=== 模型性能 ===")
        print(f"训练RMSE: {train_rmse:.4f}")
        print(f"测试RMSE: {test_rmse:.4f}")
        print(f"过拟合程度: {test_rmse - train_rmse:.4f}")
        
        # 绘制回归结果
        plot_results(x_train, y_train, x_test, y_test, y_test_pred)
        
        # 保存预测结果
        output_file = r'd:\代码\python\deeplearning\homework2\gaussian_predictions.txt'
        with open(output_file, 'w') as f:
            f.write("x_test\ty_true\ty_pred\n")
            for i in range(len(x_test)):
                f.write(f"{x_test[i]:.5f}\t{y_test[i]:.5f}\t{y_test_pred[i]:.5f}\n")
        print(f"预测结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"模型训练失败: {e}")

if __name__ == "__main__":
    main()