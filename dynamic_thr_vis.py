# import matplotlib.pyplot as plt
# import torch
#
#
# def visualize_dynamic_threshold(mask_threshold, control_value, control_point, distance_range=(0, 3)):
#     relative_distance = torch.linspace(distance_range[0], distance_range[1], 300)
#     max_thr = 1.0
#     control_point = torch.tensor(control_point, dtype=torch.float32)
#     distance_end = torch.tensor(distance_range[1], dtype=torch.float32)
#     value = torch.tensor((control_value - mask_threshold) / (max_thr - mask_threshold), dtype=torch.float32)
#     # Recalculate scale based on control_value and control_point
#     scale = torch.log(value) / (control_point-distance_end)
#
#     # Calculate dynamic threshold using the recalculated scale
#     dynamic_thr = mask_threshold + (max_thr - mask_threshold) * torch.exp(relative_distance * scale - 1) / torch.exp(distance_end * scale - 1)
#
#     dynamic_thr = torch.clamp(dynamic_thr, 0, 1)  # Ensure dynamic_thr is within 0 to 1
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(relative_distance.numpy(), dynamic_thr.numpy(),
#              label=f'Control Value: {control_value} at Distance: {control_point.item()}', color='blue')
#     plt.xlabel('Relative Distance')
#     plt.ylabel('Dynamic Threshold')
#     plt.title('Dynamic Threshold vs Relative Distance')
#
#     # Customizing the x-axis to show evenly spaced ticks with one decimal place
#     plt.xticks(ticks=[round(x, 1) for x in torch.linspace(distance_range[0], distance_range[1], 10).numpy()])
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#
# # 示例使用
# visualize_dynamic_threshold(mask_threshold=0.1, control_value=0.15, control_point=1, distance_range=(0, 3))
import torch

class MaskProcessor:
    def generate_dynamic_threshold_function(self, mask_threshold, control_value, control_point, distance_range=(0, 3), device='cpu'):
        max_thr = 1.0
        control_point = torch.tensor(control_point, dtype=torch.float32, device=device)
        distance_end = torch.tensor(distance_range[1], dtype=torch.float32, device=device)
        value = torch.tensor((control_value - mask_threshold) / (max_thr - mask_threshold), dtype=torch.float32, device=device)

        # Recalculate scale based on control_value and control_point
        scale = torch.log(value) / (control_point - distance_end)

        # Return the dynamic threshold function
        def dynamic_threshold_function(relative_distance):
            relative_distance = relative_distance.clone().detach().to(device=device, dtype=torch.float32)
            return mask_threshold + (max_thr - mask_threshold) * torch.exp(relative_distance * scale - 1) / torch.exp(distance_end * scale - 1)

        return dynamic_threshold_function

# 示例使用
processor = MaskProcessor()

# 生成一个虚拟的 distance_map 进行测试
distance_map_example = torch.rand((512, 512), device='cuda')

# 归一化 distance_map
max_distance = distance_map_example.max()
distance_map_example_normalized = distance_map_example / max_distance

# 获取 distance_map 的设备
device = distance_map_example.device

# 生成动态阈值函数
dynamic_threshold_fn = processor.generate_dynamic_threshold_function(mask_threshold=0.1, control_value=0.15, control_point=1, distance_range=(0, 3), device=device)

# 计算动态阈值
dynamic_thr = dynamic_threshold_fn(distance_map_example_normalized)

# 打印动态阈值的形状和设备信息
print("Dynamic Threshold Shape:", dynamic_thr.shape)
print("Dynamic Threshold Device:", dynamic_thr.device)

