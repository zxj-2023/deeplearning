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

def multinomial_basis(x, feature_num=10):
    """
    多项式基函数
    
    将输入特征x扩展为多项式特征 [x^0, x^1, x^2, ..., x^(feature_num-1)]。
    
    Args:
        x (numpy.ndarray): 输入数据，形状为 (N,)，其中N是样本数量
        feature_num (int): 多项式的阶数，默认为10
                          生成的特征为 x^0, x^1, ..., x^(feature_num-1)
        
    Returns:
        numpy.ndarray: 多项式特征矩阵，形状为 (N, feature_num)
    """
    # 将输入扩展为列向量，形状从 (N,) 变为 (N, 1)
    x = np.expand_dims(x, axis=1)  # shape(N, 1)

    # 初始化结果矩阵，形状为 (N, feature_num)
    ret = np.zeros((x.shape[0], feature_num))
    
    # 生成多项式特征：x^0, x^1, x^2, ..., x^(feature_num-1)
    for i in range(feature_num):
        ret[:, i] = (x[:, 0] ** i)
    return ret

def train_polynomial_regression(x_train, y_train, polynomial_order=25):
    """
    使用多项式基函数训练线性回归模型
    
    Args:
        x_train: 训练输入数据
        y_train: 训练目标数据
        polynomial_order: 多项式阶数
    
    Returns:
        w: 训练好的权重参数
        x_mean, x_std: 用于标准化的参数
    """
    print(f"训练数据: {len(x_train)} 个样本")
    print(f"x范围: [{np.min(x_train):.2f}, {np.max(x_train):.2f}]")
    print(f"y范围: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    print(f"使用 {polynomial_order} 阶多项式")
    
    # 特征标准化 - 关键步骤，避免数值问题
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    x_train_normalized = (x_train - x_mean) / x_std
    
    print(f"标准化后x范围: [{np.min(x_train_normalized):.2f}, {np.max(x_train_normalized):.2f}]")
    
    # 创建多项式特征矩阵
    phi_poly = multinomial_basis(x_train_normalized, feature_num=polynomial_order+1)
    # 添加偏置项（实际上x^0已经是偏置项了，但为了与原代码保持一致）
    phi0 = np.expand_dims(np.ones_like(x_train_normalized), axis=1)
    phi = np.concatenate([phi0, phi_poly], axis=1)
    
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
    
    return w, x_mean, x_std, polynomial_order

def predict(x_test, w, x_mean, x_std, polynomial_order):
    """
    使用训练好的模型进行预测
    
    Args:
        x_test: 测试输入数据
        w: 训练好的权重
        x_mean, x_std: 标准化参数
        polynomial_order: 多项式阶数
    
    Returns:
        y_pred: 预测结果
    """
    # 使用训练时相同的标准化
    x_test_normalized = (x_test - x_mean) / x_std
    
    # 创建相同的特征矩阵
    phi_poly = multinomial_basis(x_test_normalized, feature_num=polynomial_order+1)
    phi0 = np.expand_dims(np.ones_like(x_test_normalized), axis=1)
    phi = np.concatenate([phi0, phi_poly], axis=1)
    
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
    plt.title('Polynomial Regression Results')
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
    
    # 设置多项式阶数
    polynomial_order = 12
    
    print(f"\n=== 训练 {polynomial_order} 阶多项式回归模型 ===")
    
    try:
        # 训练模型
        w, x_mean, x_std, poly_order = train_polynomial_regression(x_train, y_train, polynomial_order)
        
        # 预测
        y_train_pred = predict(x_train, w, x_mean, x_std, poly_order)
        y_test_pred = predict(x_test, w, x_mean, x_std, poly_order)
        
        # 评估
        train_rmse = evaluate(y_train, y_train_pred)
        test_rmse = evaluate(y_test, y_test_pred)
        
        print(f"\n=== 模型性能 ===")
        print(f"训练RMSE: {train_rmse:.4f}")
        print(f"测试RMSE: {test_rmse:.4f}")
        print(f"过拟合程度: {test_rmse - train_rmse:.4f}")
        
        # 绘制结果
        plot_results(x_train, y_train, x_test, y_test, y_test_pred)
        
        # 保存预测结果
        output_file = r'd:\代码\python\deeplearning\homework2\predictions.txt'
        with open(output_file, 'w') as f:
            f.write("x_test\ty_true\ty_pred\n")
            for i in range(len(x_test)):
                f.write(f"{x_test[i]:.5f}\t{y_test[i]:.5f}\t{y_test_pred[i]:.5f}\n")
        print(f"预测结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"模型训练失败: {e}")

if __name__ == "__main__":
    main()