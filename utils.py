import numpy as np
import pygame.draw
import math
from numba import njit

@njit
def clip_scalar(value, min_val, max_val):
    """Scalar clipping function compatible with Numba."""
    return max(min_val, min(max_val, value))

# https://stackoverflow.com/a/73855696
def draw_rectangle(screen, x, y, width, height, color, rx, ry):
    """Draw a rectangle, centered at x, y.

    Arguments:
    x (int/float):
        The x coordinate of the center of the shape.
    y (int/float):
        The y coordinate of the center of the shape.
    width (int/float):
        The width of the rectangle.
    height (int/float):
        The height of the rectangle.
    color (str):
        Name of the fill color, in HTML format.
    """

    points = []

    # The distance from the center of the rectangle to
    # one of the corners is the same for each corner.
    radius = math.sqrt((height / 2) ** 2 + (width / 2) ** 2)

    # Get the angle to one of the corners with respect
    # to the x-axis.
    angle = math.atan2(height / 2, width / 2)

    # Transform that angle to reach each corner of the rectangle.
    angles = [angle, -angle + math.pi, angle + math.pi, -angle]

    # Calculate rotation.
    rot_radians = math.atan2(ry, rx)

    # Calculate the coordinates of each point.
    for angle in angles:
        y_offset = -1 * radius * math.sin(angle + rot_radians)
        x_offset = radius * math.cos(angle + rot_radians)
        points.append((x + x_offset, y + y_offset))

    pygame.draw.polygon(screen, color, points)

def check_obs_in_space(obs, obs_space):
    # 確保 obs 是 NumPy 陣列
    obs = np.asarray(obs)

    # 檢查形狀是否一致
    if obs.shape != obs_space.low.shape or obs.shape != obs_space.high.shape:
        raise ValueError(f"觀測值的形狀 {obs.shape} 與觀測空間的 low/high 形狀 {obs_space.low.shape} 不一致")

    # 逐元素檢查 obs 是否在 [low, high] 範圍內
    within_bounds = (obs >= obs_space.low) & (obs <= obs_space.high)

    # 如果所有元素都在範圍內，直接返回 True
    if np.all(within_bounds):
        return True

    # 找出超出範圍的索引
    out_of_bounds_indices = np.where(~within_bounds)
    error_messages = []

    # 遍歷所有超出範圍的索引
    for idx in zip(*out_of_bounds_indices):
        idx_str = str(idx)  # 將索引轉為字串
        obs_value = obs[idx]  # 當前觀測值
        low_value = obs_space.low[idx]  # 對應的下界
        high_value = obs_space.high[idx]  # 對應的上界

        # 判斷超出的是下界還是上界
        if obs_value < low_value:
            error_messages.append(f"索引 {idx_str}: 觀測值 {obs_value} 小於下界 {low_value}")
        elif obs_value > high_value:
            error_messages.append(f"索引 {idx_str}: 觀測值 {obs_value} 大於上界 {high_value}")

    # 抛出詳細的錯誤訊息
    raise ValueError("以下觀測值超出範圍:\n" + "\n".join(error_messages))