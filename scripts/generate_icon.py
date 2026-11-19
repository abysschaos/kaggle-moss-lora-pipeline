"""
生成 OpenMOSS MOSS-Image LoRA 训练器专属多分辨率 Windows 图标 (assets/moss_trainer.ico)。
采用 Catppuccin Mocha 深色主题配色，通过 1024x1024 超采样渲染确保 16px~256px 全尺寸清晰抗锯齿。
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ICO = PROJECT_ROOT / "assets" / "moss_trainer.ico"


def _lerp_color(c1, c2, t: float):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def build_master_icon(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # 1. 圆角渐变底座 (Catppuccin Mocha 深蓝紫)
    margin = int(size * 0.06)
    radius = int(size * 0.22)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=255,
    )

    bg_grad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = bg_grad.load()
    c_top = (24, 24, 37, 255)      # #181825
    c_mid = (30, 30, 46, 255)      # #1e1e2e
    c_bot = (39, 36, 62, 255)      # deep violet-slate
    for y in range(size):
        t = y / (size - 1)
        if t < 0.5:
            col = _lerp_color(c_top, c_mid, t * 2.0)
        else:
            col = _lerp_color(c_mid, c_bot, (t - 0.5) * 2.0)
        for x in range(size):
            px[x, y] = col

    img.paste(bg_grad, (0, 0), mask)

    # 2. 顶部内柔和蓝紫环境光晕
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow)
    g_draw.ellipse(
        [int(size * 0.14), int(size * 0.04), int(size * 0.86), int(size * 0.62)],
        fill=(137, 180, 250, 38),
    )
    g_draw.ellipse(
        [int(size * 0.34), int(size * 0.38), int(size * 0.94), int(size * 0.96)],
        fill=(203, 166, 247, 34),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=int(size * 0.08)))
    glow_clipped = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_clipped.paste(glow, (0, 0), mask)
    img = Image.alpha_composite(img, glow_clipped)

    # 3. 精致蓝紫微光边框
    border_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(border_layer)
    b_width = max(4, int(size * 0.022))
    b_draw.rounded_rectangle(
        [margin + b_width // 2, margin + b_width // 2, size - margin - b_width // 2, size - margin - b_width // 2],
        radius=radius - b_width // 3,
        outline=(137, 180, 250, 155),
        width=b_width,
    )
    img = Image.alpha_composite(img, border_layer)

    # 4. 极简科技网格底纹（低调装饰，提升质感）
    grid_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gr_draw = ImageDraw.Draw(grid_layer)
    step = int(size * 0.14)
    for pos in range(margin + step, size - margin, step):
        gr_draw.line([(pos, margin), (pos, size - margin)], fill=(166, 173, 200, 16), width=2)
        gr_draw.line([(margin, pos), (size - margin, pos)], fill=(166, 173, 200, 16), width=2)
    grid_clipped = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grid_clipped.paste(grid_layer, (0, 0), mask)
    img = Image.alpha_composite(img, grid_clipped)

    # 5. 核心图标：几何 "M" 折线（融合 MOSS 首字母与神经网络 Loss 趋势意象）
    pts = [
        (int(size * 0.24), int(size * 0.72)),
        (int(size * 0.24), int(size * 0.30)),
        (int(size * 0.50), int(size * 0.56)),
        (int(size * 0.76), int(size * 0.30)),
        (int(size * 0.76), int(size * 0.72)),
    ]

    # 折线外发光层
    neon_glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ng_draw = ImageDraw.Draw(neon_glow)
    stroke_w = int(size * 0.088)
    ng_draw.line(pts, fill=(137, 180, 250, 110), width=int(stroke_w * 1.45), joint="curve")
    for p in pts:
        r = int(stroke_w * 0.72)
        ng_draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(203, 166, 247, 100))
    neon_glow = neon_glow.filter(ImageFilter.GaussianBlur(radius=int(size * 0.032)))
    img = Image.alpha_composite(img, neon_glow)

    # 主体分段渐变折线
    main_line = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ml_draw = ImageDraw.Draw(main_line)
    seg_colors = [
        (137, 180, 250, 255),  # #89b4fa 蓝
        (166, 175, 249, 255),  # 过渡蓝紫
        (203, 166, 247, 255),  # #cba6f7 紫
        (166, 227, 161, 255),  # #a6e3a1 绿（收敛向好意象）
    ]
    for i in range(len(pts) - 1):
        ml_draw.line([pts[i], pts[i + 1]], fill=seg_colors[i], width=stroke_w)
    # 圆头拐点平滑衔接
    cap_r = stroke_w // 2
    for i, p in enumerate(pts):
        col = seg_colors[min(i, len(seg_colors) - 1)]
        ml_draw.ellipse([p[0] - cap_r, p[1] - cap_r, p[0] + cap_r, p[1] + cap_r], fill=col)

    # 关键神经节点高亮圆点
    node_specs = [
        (pts[1], (137, 180, 250, 255)),
        (pts[2], (203, 166, 247, 255)),
        (pts[3], (166, 227, 161, 255)),
    ]
    node_outer = int(size * 0.056)
    node_inner = int(size * 0.026)
    for (nx, ny), ncol in node_specs:
        ml_draw.ellipse([nx - node_outer, ny - node_outer, nx + node_outer, ny + node_outer], fill=ncol)
        ml_draw.ellipse([nx - node_inner, ny - node_inner, nx + node_inner, ny + node_inner], fill=(24, 24, 37, 255))

    # 底部细长能量进度条点缀（呼应训练进度）
    bar_y1 = int(size * 0.81)
    bar_y2 = int(size * 0.855)
    ml_draw.rounded_rectangle(
        [int(size * 0.24), bar_y1, int(size * 0.76), bar_y2],
        radius=int(size * 0.022),
        fill=(49, 50, 68, 220),
    )
    ml_draw.rounded_rectangle(
        [int(size * 0.24), bar_y1, int(size * 0.62), bar_y2],
        radius=int(size * 0.022),
        fill=(166, 227, 161, 245),
    )

    img = Image.alpha_composite(img, main_line)
    return img


def main():
    OUT_ICO.parent.mkdir(parents=True, exist_ok=True)
    master = build_master_icon(1024)
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    master.save(OUT_ICO, format="ICO", sizes=sizes)
    print(f"Icon generated: {OUT_ICO} ({OUT_ICO.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
