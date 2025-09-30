# Local Stable Diffusion MCP

一个基于本地Stable Diffusion WebUI的MCP（Model Context Protocol）服务器，提供AI图片生成功能，支持透明背景图片生成。

## 🎨 功能特性

- **文本生成图片**：支持多种风格和参数设置
- **透明背景生成**：专门的透明背景图片生成功能
- **图生图功能**：基于现有图片生成新图片
- **多种采样器**：Euler a、DPM++ 2M、DPM++ SDE等
- **预设风格模板**：动漫角色、写实肖像、幻想艺术、现代风格
- **智能提示词优化**：自动生成高质量提示词

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Stable Diffusion WebUI（已安装并运行）
- MCP客户端（Claude Desktop、Cursor等）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置MCP服务器

#### 配置文件示例（`config.json`）

```json
{
  "sd_webui_url": "http://127.0.0.1:7860",
  "models_path": "<YOUR_SD_MODELS_PATH>",
  "default_model": "<YOUR_DEFAULT_MODEL>",
  "output_path": "./output",
  "max_retries": 3,
  "timeout": 60
}
```

### 4. 启动MCP服务器

```bash
python server.py
```

## 🛠️ 在各种AI编辑器中配置MCP

### Trae&Trae CN

```
{
  "mcpServers": {
    "local_stable_diffusion_mcp": {
      "command": "python",
      "args": [
        "<YOUR_PROJECT_PATH>\\server.py"
      ]
    }
  }
}
```

### Claude Desktop

1. 打开Claude Desktop设置
2. 找到MCP配置选项
3. 添加新的MCP服务器：

```json
{
  "mcpServers": {
    "local-stable-diffusion": {
      "command": "python",
      "args": ["<YOUR_PROJECT_PATH>/server.py"],
      "env": {
        "PYTHONPATH": "<YOUR_PROJECT_PATH>"
      }
    }
  }
}
```

### Cursor

1. 打开Cursor设置（Settings）
2. 搜索"MCP"或"Model Context Protocol"
3. 在MCP配置中添加：

```json
{
  "servers": {
    "local-stable-diffusion": {
      "command": "python",
      "args": ["<YOUR_PROJECT_PATH>/server.py"],
      "cwd": "<YOUR_PROJECT_PATH>"
    }
  }
}
```

### VS Code + Continue插件

1. 安装Continue插件
2. 打开`.continue/config.json`
3. 添加MCP服务器配置：

```json
{
  "models": [
    {
      "title": "Local Stable Diffusion MCP",
      "provider": "mcp",
      "server": {
        "command": "python",
        "args": ["server.py"],
        "cwd": "<YOUR_PROJECT_PATH>"
      }
    }
  ]
}
```

### Windsurf

1. 打开Windsurf设置
2. 导航到MCP配置
3. 添加服务器配置：

```json
{
  "mcpServers": {
    "stable-diffusion": {
      "command": "python",
      "args": ["<YOUR_PROJECT_PATH>/server.py"],
      "env": {
        "MCP_CONFIG_PATH": "<YOUR_PROJECT_PATH>/config.json"
      }
    }
  }
}
```

## 🎨 使用示例

### 生成透明背景图片

```python
# 生成透明背景的动漫角色
generate_transparent_image(
    prompt="beautiful anime girl, long silver hair, detailed eyes, school uniform",
    output_path="./transparent_character.png",
    width=512,
    height=768,
    style="anime_character"
)
```

### 图生图功能

```python
# 基于现有图片生成新图片
generate_image_img2img(
    input_image_path="./input.jpg",
    prompt="beautiful anime style, detailed eyes, masterpiece",
    output_path="./output.png",
    denoising_strength=0.75
)
```

## ⚙️ 参数说明

### 基础参数

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `width` | 图片宽度 | 512 | 64-2048 |
| `height` | 图片高度 | 512 | 64-2048 |
| `steps` | 生成步数 | 20 | 1-150 |
| `cfg_scale` | CFG引导强度 | 7.5 | 1-30 |
| `sampler` | 采样器 | "Euler a" | 多种可选 |

### 风格模板

- `none`：无特定风格
- `anime_character`：动漫角色风格
- `realistic_portrait`：写实肖像风格
- `fantasy_art`：幻想艺术风格
- `modern_style`：现代风格

### 采样器推荐

- **快速生成**：Euler a
- **平衡质量**：DPM++ 2M
- **高质量**：DPM++ SDE

## 💡 使用技巧

### 透明背景生成技巧

1. **使用专门的透明背景功能**：使用`generate_transparent_image`而不是普通生成功能
2. **选择合适的提示词**：包含"transparent background", "no background", "isolated object"
3. **避免背景相关词汇**：如"background", "scene", "landscape"
4. **使用适当的风格模板**：根据需求选择`anime_character`、`fantasy_art`等

### 提示词优化

系统会自动添加以下优化：
- 质量增强：`best quality, amazing quality, very aesthetic, absurdres`
- 透明背景：`transparent background, alpha channel, no background, isolated object`
- 负面提示词：`background, white background, black background, colored background`

## 🔧 故障排除

### 常见问题

1. **连接失败**：检查Stable Diffusion WebUI是否运行
2. **模型加载失败**：确认模型文件路径正确
3. **生成质量差**：调整`steps`和`cfg_scale`参数
4. **透明效果不佳**：使用专门的透明生成功能

### 调试模式

启动服务器时添加调试参数：

```bash
python server.py --debug
```

## 📄 文件结构

```
local_stable_diffusion_mcp/
├── config.json              # 配置文件
├── requirements.txt         # Python依赖
├── server.py               # MCP服务器主文件
└── README.md              # 本文件
```

## 🤝 贡献

此项目为测试使用项目不一定保证都能正常使用，欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可证

MIT License
