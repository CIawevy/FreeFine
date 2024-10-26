import pyautogui
import time

print("移动鼠标到你想要的坐标位置，5秒后程序会显示坐标。")
time.sleep(5)  # 等待5秒
print(pyautogui.position())  # 输出当前鼠标位置



# # 指定点击位置的坐标
# positions = [(100, 200), (300, 400)]  # 替换为实际坐标
#
# try:
#     while True:
#         for pos in positions:
#             pyautogui.click(pos[0], pos[1])  # 点击指定位置
#             time.sleep(0.5)  # 可选：在两个点击之间稍作停顿
#         time.sleep(30)  # 每30秒执行一次
# except KeyboardInterrupt:
#     print("脚本停止")
