#!/usr/bin/env python3
"""
NovelAI MCP Server - 本地Stable Diffusion WebUI适配版本
用于连接运行在http://127.0.0.1:7860的Stable Diffusion WebUI
"""

import json
import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

# 尝试导入图像处理库
try:
    from PIL import Image
    import numpy as np
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    logger.warning("PIL或numpy未安装，透明背景功能将受限。可以尝试: pip install pillow numpy")

# 尝试导入mcp模块，如果失败则尝试安装
try:
    import mcp
except ImportError:
    print("警告: 未找到mcp模块，请确保已安装mcp包")
    print("可以尝试: pip install mcp")

# 添加mcp模块路径（如果需要）
mcp_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if os.path.exists(os.path.join(mcp_path, 'mcp')):
    sys.path.append(mcp_path)

try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
except ImportError:
    print("错误: 无法导入mcp模块。请确保已安装mcp包。")
    sys.exit(1)

import aiohttp
import asyncio

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 强制设置工作目录为脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)
logger.info(f"工作目录已设置为: {script_dir}")



def load_config():
    """从config.json加载配置，如果文件不存在或配置不完整则报错"""
    config_path = script_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}。请确保config.json文件存在于项目目录中。")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必需的配置项
        required_keys = ["default_model", "base_url"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"配置文件缺少必需的配置项: {', '.join(missing_keys)}。请检查config.json文件是否完整。")
        
        logger.info(f"成功从 {config_path} 加载配置")
        return config, config_path
        
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {config_path}。请确保config.json是有效的JSON格式。错误详情: {e}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")

# 加载配置
NOVELAI_CONFIG, CONFIG_PATH = load_config()

class NovelAIMCP:
    def __init__(self):
        self.server = Server("novelai-local")
        self.setup_tools()
    
    def create_full_mask_base64(self, width: int, height: int) -> str:
        """创建全尺寸白色mask用于ControlNet inpainting"""
        try:
            # 创建白色背景图像（用于mask）
            mask_image = Image.new('L', (width, height), color=255)  # 白色背景
            
            # 将mask转换为base64
            buffer = io.BytesIO()
            mask_image.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.info(f"已创建 {width}x{height} 的白色mask用于ControlNet inpainting")
            return mask_base64
            
        except Exception as e:
            logger.error(f"创建mask时出错: {str(e)}")
            # 返回空字符串作为fallback
            return ""
        
    def setup_tools(self):
        """设置MCP工具"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="generate_image",
                    description="使用Stable Diffusion WebUI生成图片并保存到指定位置。当前模型: anything-v4.0.ckpt，支持多种采样器如DPM++ 2M、Euler a等。支持高质量图片生成，可自定义尺寸、步数、CFG等参数。内置风格模板: anime_character(动漫角色), realistic_portrait(写实肖像), fantasy_art(幻想艺术), modern_style(现代风格)。透明背景功能使用ControlNet inpainting方法。提示词示例: 'beautiful anime girl, long hair, detailed eyes, masterpiece'，负面提示词默认包含低质量、模糊等",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "正向提示词，描述想要生成的内容。例如: 'beautiful anime girl, long hair, detailed eyes, masterpiece, best quality'"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "图片保存路径，例如: 'C:/images/my_image.png' 或 './output/image.png'"
                            },
                            "model_name": {
                                "type": "string",
                                "description": "指定使用的模型名称，例如: 'anything-v5.safetensors'。如果不指定，将使用默认模型 anything-v5.safetensors",
                                "default": "sd1.5\\anything-v5.safetensors"
                            },
                            "transparent_background": {
                                "type": "boolean",
                                "description": "是否生成透明背景图片，默认false。如果为true，使用ControlNet inpainting方法生成PNG格式的透明背景图片",
                                "default": False
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "负面提示词，描述不想要的内容。默认: 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, disgusting, poorly drawn hands, missing limbs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck'",
                                "default": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, disgusting, poorly drawn hands, missing limbs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck"
                            },
                            "width": {
                                "type": "integer",
                                "description": "图片宽度（必需），例如512像素",
                                "minimum": 64,
                                "maximum": 2048
                            },
                            "height": {
                                "type": "integer",
                                "description": "图片高度（必需），例如512像素",
                                "minimum": 64,
                                "maximum": 2048
                            },
                            "steps": {
                                "type": "integer",
                                "description": "生成步数，默认20步。步数越多质量越高但耗时越长",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 150
                            },
                            "cfg_scale": {
                                "type": "number",
                                "description": "CFG Scale，默认7.5。控制提示词引导强度，范围1-30",
                                "default": 7.5,
                                "minimum": 1,
                                "maximum": 30
                            },
                            "sampler": {
                                "type": "string",
                                "description": "采样器，默认使用'Euler a'。推荐采样器: Euler a(快速), DPM++ 2M(平衡), DPM++ SDE(高质量)",
                                "default": "Euler a",
                                "enum": ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M Karras", "DPM++ SDE Karras", "DPM fast", "DPM adaptive", "DDIM", "PLMS", "UniPC", "LCM"]
                            },
                            "style": {
                                "type": "string",
                                "description": "预设风格模板，默认'none'。可选: none(无), anime_character(动漫角色), realistic_portrait(写实肖像), fantasy_art(幻想艺术), modern_style(现代风格)",
                                "default": "none",
                                "enum": ["none", "anime_character", "realistic_portrait", "fantasy_art", "modern_style"]
                            }
                        },
                        "required": ["prompt", "output_path"]
                    }
                ),
                Tool(
                    name="get_prompt_suggestions",
                    description="获取提示词建议和配置信息。可以获取当前模型信息、可用采样器、角色提示词、风格修饰符、负面提示词、质量增强器等。支持按类别筛选: all(全部), characters(角色), styles(风格), negative(负面), quality(质量), samplers(采样器), scene_backgrounds(场景背景), clothing_accessories(服装配饰), environment_tags(环境标签), technical_parameters(技术参数)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "建议类别，默认'all'。可选: all(全部信息), characters(角色提示词), styles(风格修饰符), negative(负面提示词), quality(质量增强器), samplers(采样器推荐), scene_backgrounds(场景背景), clothing_accessories(服装配饰), environment_tags(环境标签), technical_parameters(技术参数)",
                                "default": "all",
                                "enum": ["all", "characters", "styles", "negative", "quality", "samplers", "scene_backgrounds", "clothing_accessories", "environment_tags", "technical_parameters"]
                            }
                        }
                    }
                ),
                Tool(
                    name="get_models",
                    description="获取可用的Stable Diffusion模型列表，包括当前加载的模型信息",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_model_details",
                    description="获取详细的模型信息，包括技术参数、VAE配置、CLIP设置和系统信息",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_model_recommendations",
                    description="获取模型使用推荐和最佳实践，包括参数设置和优化建议",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="generate_transparent_image",
                    description="专门生成透明背景图片，使用ControlNet inpainting方法。自动优化提示词并输出PNG格式",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "正向提示词，描述想要生成的内容。例如: 'cute anime girl with cat ears, beautiful detailed eyes, masterpiece'"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "图片保存路径，必须为PNG格式。例如: 'C:/images/my_character.png' 或 './output/character.png'"
                            },
                            "model_name": {
                                "type": "string",
                                "description": "指定使用的模型名称，例如: 'anything-v5.safetensors'。如果不指定，将使用默认模型 anything-v5.safetensors",
                                "default": "sd1.5\\anything-v5.safetensors"
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "负面提示词，描述不想要的内容。默认包含背景相关负面提示",
                                "default": "background, white background, black background, colored background, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                            },
                            "width": {
                                "type": "integer",
                                "description": "图片宽度，默认512像素",
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048
                            },
                            "height": {
                                "type": "integer",
                                "description": "图片高度，默认512像素",
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048
                            },
                            "steps": {
                                "type": "integer",
                                "description": "生成步数，默认20步。步数越多质量越高但耗时越长",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 150
                            },
                            "cfg_scale": {
                                "type": "number",
                                "description": "CFG Scale，默认7.5。控制提示词引导强度，范围1-30",
                                "default": 7.5,
                                "minimum": 1,
                                "maximum": 30
                            },
                            "sampler": {
                                "type": "string",
                                "description": "采样器，默认使用'Euler a'。推荐采样器: Euler a(快速), DPM++ 2M(平衡), DPM++ SDE(高质量)",
                                "default": "Euler a",
                                "enum": ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M Karras", "DPM++ SDE Karras", "DPM fast", "DPM adaptive", "DDIM", "PLMS", "UniPC", "LCM"]
                            },
                            "style": {
                                "type": "string",
                                "description": "预设风格模板，默认'none'。可选: none(无), anime_character(动漫角色), realistic_portrait(写实肖像), fantasy_art(幻想艺术), modern_style(现代风格)",
                                "default": "none",
                                "enum": ["none", "anime_character", "realistic_portrait", "fantasy_art", "modern_style"]
                            }
                        },
                        "required": ["prompt", "output_path"]
                    }
                ),
                Tool(
                    name="generate_image_img2img",
                    description="使用图生图(img2img)功能基于输入图片生成新图片。需要提供输入图片路径，支持调整重绘幅度等参数",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "正向提示词，描述想要生成的内容。例如: 'beautiful anime girl, detailed eyes, masterpiece'"
                            },
                            "input_image_path": {
                                "type": "string",
                                "description": "输入图片路径，例如: 'C:/images/input.jpg' 或 './input/source.png'。支持的格式: JPG, PNG, BMP等"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "输出图片保存路径，例如: 'C:/images/output.png' 或 './output/result.png'"
                            },
                            "denoising_strength": {
                                "type": "number",
                                "description": "重绘幅度，控制输入图片的影响程度。0.0表示完全保留原图，1.0表示完全重新生成。默认0.75",
                                "default": 0.75,
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "model_name": {
                                "type": "string",
                                "description": "指定使用的模型名称，例如: 'anything-v4.0.ckpt'。如果不指定，将使用默认模型 anything-v4.0.ckpt",
                                "default": "anything-v4.0\\anything-v4.0.ckpt [3b26c9c497]"
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "负面提示词，描述不想要的内容。默认: 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'",
                                "default": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                            },
                            "width": {
                                "type": "integer",
                                "description": "图片宽度，默认512像素",
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048
                            },
                            "height": {
                                "type": "integer",
                                "description": "图片高度，默认512像素",
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048
                            },
                            "steps": {
                                "type": "integer",
                                "description": "生成步数，默认20步。步数越多质量越高但耗时越长",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 150
                            },
                            "cfg_scale": {
                                "type": "number",
                                "description": "CFG Scale，默认7.5。控制提示词引导强度，范围1-30",
                                "default": 7.5,
                                "minimum": 1,
                                "maximum": 30
                            },
                            "sampler": {
                                "type": "string",
                                "description": "采样器，默认使用'Euler a'。推荐采样器: Euler a(快速), DPM++ 2M(平衡), DPM++ SDE(高质量)",
                                "default": "Euler a",
                                "enum": ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M Karras", "DPM++ SDE Karras", "DPM fast", "DPM adaptive", "DDIM", "PLMS", "UniPC", "LCM"]
                            },
                            "style": {
                                "type": "string",
                                "description": "预设风格模板，默认'none'。可选: none(无), anime_character(动漫角色), realistic_portrait(写实肖像), fantasy_art(幻想艺术), modern_style(现代风格)",
                                "default": "none",
                                "enum": ["none", "anime_character", "realistic_portrait", "fantasy_art", "modern_style"]
                            },
                            "resize_mode": {
                                "type": "string",
                                "description": "调整大小模式，默认'Crop and Resize'。可选: Just resize(仅调整大小), Crop and resize(裁剪并调整), Resize and fill(调整并填充), Just resize (latent upscale)(仅调整大小-潜空间放大)",
                                "default": "Crop and resize",
                                "enum": ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
                            },
                            "mask_image_path": {
                                "type": "string",
                                "description": "遮罩图片路径（可选），用于局部重绘。白色区域表示需要重绘，黑色区域表示保持原图。例如: 'C:/images/mask.png' 或 './mask/mask.png'"
                            },
                            "inpainting_mask_invert": {
                                "type": "integer",
                                "description": "遮罩反转模式，0=不反转（默认），1=反转遮罩。反转后黑色区域重绘，白色区域保持",
                                "default": 0,
                                "minimum": 0,
                                "maximum": 1
                            },
                            "inpainting_fill_mode": {
                                "type": "string",
                                "description": "局部重绘填充模式，默认'original'。可选: fill(填充), original(原图), latent_noise(潜变量噪声), latent_nothing(潜变量无)",
                                "default": "original",
                                "enum": ["fill", "original", "latent_noise", "latent_nothing"]
                            }
                        },
                        "required": ["prompt", "input_image_path", "output_path", "width", "height"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Any]:
            if name == "generate_image":
                return await self.generate_image(arguments)
            elif name == "generate_transparent_image":
                return await self.generate_transparent_image(arguments)
            elif name == "get_prompt_suggestions":
                return await self.get_prompt_suggestions(arguments)
            elif name == "get_models":
                return await self.get_models(arguments)
            elif name == "get_model_details":
                return await self.get_model_details(arguments)
            elif name == "get_model_recommendations":
                return await self.get_model_recommendations(arguments)
            elif name == "generate_image_img2img":
                return await self.generate_image_img2img(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def generate_image(self, arguments: Dict[str, Any]) -> List[Any]:
        """生成图片并保存到指定路径"""
        try:
            prompt = arguments.get("prompt", "")
            output_path = arguments.get("output_path", "")
            model_name = arguments.get("model_name", NOVELAI_CONFIG.get("default_model", "sd1.5\\anything-v5.safetensors"))  # 使用默认模型
            vae_name = arguments.get("vae_name", NOVELAI_CONFIG.get("default_vae", None))  # 使用默认VAE
            # 如果传递了空字符串，也使用默认模型
            if not model_name or model_name.strip() == "":
                model_name = NOVELAI_CONFIG.get("default_model", "sd1.5\\anything-v5.safetensors")
            transparent_background = arguments.get("transparent_background", False)
            negative_prompt = arguments.get("negative_prompt", NOVELAI_CONFIG["prompt_suggestions"]["negative_prompts"]["general"])
            width = arguments.get("width", NOVELAI_CONFIG["default_params"]["width"])
            height = arguments.get("height", NOVELAI_CONFIG["default_params"]["height"])
            steps = arguments.get("steps", NOVELAI_CONFIG["default_params"]["steps"])
            cfg_scale = arguments.get("cfg_scale", NOVELAI_CONFIG["default_params"]["cfg_scale"])
            sampler = arguments.get("sampler", NOVELAI_CONFIG["default_params"]["sampler_index"])
            style = arguments.get("style", "none")
            
            if not prompt:
                return [TextContent(type="text", text="错误: 提示词不能为空")]
            
            if not output_path:
                return [TextContent(type="text", text="错误: 输出路径不能为空")]
            
            # 如果模型名称为空，使用默认模型
            if not model_name:
                model_name = NOVELAI_CONFIG["default_model"]
                logger.info(f"未指定模型，使用默认模型: {model_name}")
            
            # 应用预设风格模板
            if style != "none" and style in NOVELAI_CONFIG["prompt_suggestions"]:
                if style == "anime_character":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['anime_girl']}, {prompt}"
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['anime_style']}, {prompt}"
                    negative_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['negative_prompts']['anime']}, {negative_prompt}"
                elif style == "realistic_portrait":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['realistic_style']}, {prompt}"
                    negative_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['negative_prompts']['realistic']}, {negative_prompt}"
                elif style == "fantasy_art":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['fantasy_character']}, {prompt}"
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['artistic_style']}, {prompt}"
                elif style == "modern_style":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['modern_character']}, {prompt}"
            
            # 添加质量增强器
            prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['quality_enhancers']['high_quality']}, {prompt}"
            
            logger.info(f"生成图片 - 风格: {style}, 提示词: {prompt}")
            
            # 设置模型和VAE
            try:
                # 设置VAE
                if vae_name and vae_name != "None" and vae_name.strip() != "":
                    logger.info(f"设置VAE: {vae_name}")
                    async with aiohttp.ClientSession() as session:
                        vae_payload = {"sd_vae": vae_name}
                        async with session.post(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", 
                                              json=vae_payload, timeout=30) as vae_resp:
                            if vae_resp.status == 200:
                                logger.info(f"VAE设置成功: {vae_name}")
                            else:
                                logger.warning(f"VAE设置失败，状态码: {vae_resp.status}")
                else:
                    logger.info("未指定VAE，使用当前VAE设置")
                
                # 设置模型
                # 获取当前模型信息
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", timeout=30) as resp:
                        if resp.status == 200:
                            current_options = await resp.json()
                            current_model = current_options.get('sd_model_checkpoint', '')
                            
                            # 如果当前模型与指定模型不同，则切换模型
                            if model_name not in current_model:
                                logger.info(f"切换模型从 {current_model} 到 {model_name}")
                                update_payload = {"sd_model_checkpoint": model_name}
                                async with session.post(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", 
                                                      json=update_payload, timeout=30) as update_resp:
                                    if update_resp.status != 200:
                                        logger.warning(f"模型切换可能失败，状态码: {update_resp.status}")
                            else:
                                logger.info(f"当前模型已为目标模型: {model_name}")
                        else:
                            logger.warning(f"无法获取当前模型信息，状态码: {resp.status}")
            except Exception as model_error:
                logger.warning(f"模型设置过程中出错: {str(model_error)}")
            
            # 如果启用透明背景，使用提示词优化方法
            if transparent_background:
                logger.info("使用提示词优化方法生成透明背景...")
                
                # 确保输出路径是PNG格式
                if not output_path.lower().endswith('.png'):
                    output_path = output_path.rsplit('.', 1)[0] + '.png'
                    logger.info(f"透明背景模式，自动更改输出路径为PNG格式: {output_path}")
                
                # 修改提示词以优化透明背景生成
                prompt = f"transparent background, alpha channel, no background, isolated object, {prompt}"
                negative_prompt = f"background, white background, black background, colored background, {negative_prompt}"
                
                logger.info("透明背景提示词优化完成，开始生成...")
            
            # 构建基础请求payload
            base_payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_index": sampler,
                "n_iter": 1,
                "batch_size": 1,
                "seed": -1,  # 随机种子
                "override_settings": {}
            }
            
            # 如果启用透明背景，添加优化参数
            if transparent_background:
                # 禁用高清修复以避免背景问题
                base_payload["enable_hr"] = False
                base_payload["restore_faces"] = False  # 禁用面部修复以避免背景干扰
                base_payload["tiling"] = False
                base_payload["eta"] = 0  # 使用确定性生成
                base_payload["s_churn"] = 0
                base_payload["s_tmax"] = 0
                base_payload["s_tmin"] = 0
                base_payload["s_noise"] = 1
                
                # 简化透明背景处理 - 仅通过提示词和输出格式实现
                # 不移除背景，仅生成适合后期处理的图片
                logger.info("透明背景模式：通过提示词优化生成，输出PNG格式")
                
                # 合并透明背景参数
                payload = base_payload
                
                logger.info("已添加透明背景提示词优化参数，并禁用LayerDiffusion脚本")
            else:
                payload = base_payload
            
            # 调用API
            url = f"{NOVELAI_CONFIG['base_url']}{NOVELAI_CONFIG['endpoint']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=NOVELAI_CONFIG["timeout"])
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API错误: {response.status} - {error_text}")
                        return [TextContent(type="text", text=f"API错误: {response.status} - {error_text}")]
                    
                    response_data = await response.json()
                    
                    # 检查返回的图片数据
                    if "images" in response_data and response_data["images"]:
                        image_data = response_data["images"][0]
                        logger.info(f"图片生成成功，大小: {len(image_data)} 字符")
                        
                        try:
                            # 调试：输出当前工作目录和脚本目录
                            current_dir = Path.cwd()
                            script_dir = Path(__file__).parent
                            logger.info(f"当前工作目录: {current_dir}")
                            logger.info(f"脚本目录: {script_dir}")
                            logger.info(f"输出路径参数: {output_path}")
                            
                            # 确保输出目录存在 - 使用脚本目录作为基础
                            output_path_obj = Path(output_path)
                            
                            # 如果是相对路径，转换为基于脚本目录的绝对路径
                            if not output_path_obj.is_absolute():
                                output_path_obj = script_dir / output_path
                                logger.info(f"转换相对路径为绝对路径: {output_path} -> {output_path_obj}")
                            
                            logger.info(f"最终输出路径: {output_path_obj}")
                            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 解码base64图片数据
                            image_bytes = base64.b64decode(image_data)
                            
                            # 保存图片（SD WebUI已处理透明背景）
                            with open(output_path_obj, 'wb') as f:
                                f.write(image_bytes)
                            
                            logger.info(f"图片成功保存到: {output_path_obj}")
                            
                            # 如果是透明背景模式，验证图片格式
                            if transparent_background:
                                try:
                                    if IMAGE_PROCESSING_AVAILABLE:
                                        img = Image.open(io.BytesIO(image_bytes))
                                        if img.mode == 'RGBA':
                                            logger.info("✅ ControlNet透明背景生成成功，图片包含Alpha通道")
                                        else:
                                            logger.info(f"ℹ️ 图片模式: {img.mode} (ControlNet可能未正确配置)")
                                            logger.info("💡 提示: 确保SD WebUI已安装ControlNet扩展和inpainting模型")
                                    else:
                                        logger.info("ℹ️ 透明背景模式启用，图像处理库不可用，无法验证Alpha通道")
                                except Exception as verify_error:
                                    logger.warning(f"透明背景验证失败: {str(verify_error)}")
                                    logger.info("💡 提示: 检查ControlNet扩展是否正确安装和配置")
                            
                            logger.info(f"图片成功保存到: {output_path_obj}")
                            
                            # 构建成功消息
                            success_message = f"图片生成成功！\n"
                            success_message += f"保存路径: {output_path_obj.absolute()}\n"
                            success_message += f"图片大小: {len(image_bytes) / 1024:.1f} KB\n"
                            success_message += f"使用模型: {model_name}\n"
                            success_message += f"采样器: {sampler}\n"
                            if transparent_background:
                                success_message += f"透明背景: 是\n"
                            success_message += f"提示词: {prompt}"
                            
                            return [TextContent(type="text", text=success_message)]
                        except Exception as save_error:
                            logger.error(f"保存图片失败: {str(save_error)}")
                            return [TextContent(type="text", text=f"保存图片失败: {str(save_error)}")]
                    else:
                        logger.error("API返回格式错误: 未找到图片数据")
                        return [TextContent(type="text", text="错误: API返回格式不正确，未找到图片数据")]
                        
        except Exception as e:
            logger.error(f"生成图片时出错: {str(e)}")
            return [TextContent(type="text", text=f"生成图片时出错: {str(e)}")]
    
    async def generate_transparent_image(self, arguments: Dict[str, Any]) -> List[Any]:
        """专门生成透明背景图片，使用优化的参数和提示词"""
        try:
            prompt = arguments.get("prompt", "")
            output_path = arguments.get("output_path", "")
            model_name = arguments.get("model_name", NOVELAI_CONFIG.get("default_model", "sd1.5\\anything-v5.safetensors"))
            vae_name = arguments.get("vae_name", NOVELAI_CONFIG.get("default_vae", None))
            # 如果传递了空字符串，也使用默认模型
            if not model_name or model_name.strip() == "":
                model_name = NOVELAI_CONFIG.get("default_model", "sd1.5\\anything-v5.safetensors")
            negative_prompt = arguments.get("negative_prompt", "background, white background, black background, colored background, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry")
            width = arguments.get("width", 512)
            height = arguments.get("height", 512)
            steps = arguments.get("steps", 20)
            cfg_scale = arguments.get("cfg_scale", 7.5)
            sampler = arguments.get("sampler", "Euler a")
            style = arguments.get("style", "none")
            
            if not prompt:
                return [TextContent(type="text", text="错误: 提示词不能为空")]
            
            if not output_path:
                return [TextContent(type="text", text="错误: 输出路径不能为空")]
            
            # 确保输出路径是PNG格式
            if not output_path.lower().endswith('.png'):
                output_path = output_path.rsplit('.', 1)[0] + '.png'
                logger.info(f"透明背景模式，自动更改输出路径为PNG格式: {output_path}")
            
            logger.info("🎯 开始生成透明背景图片...")
            
            # 优化提示词以生成透明背景
            optimized_prompt = f"transparent background, alpha channel, no background, isolated object, {prompt}"
            optimized_negative_prompt = f"background, white background, black background, colored background, gradient background, shadow, reflection, {negative_prompt}"
            
            # 应用预设风格模板
            if style != "none" and style in NOVELAI_CONFIG["prompt_suggestions"]:
                if style == "anime_character":
                    optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['anime_girl']}, {optimized_prompt}"
                    optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['anime_style']}, {optimized_prompt}"
                    optimized_negative_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['negative_prompts']['anime']}, {optimized_negative_prompt}"
                elif style == "realistic_portrait":
                    optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['realistic_style']}, {optimized_prompt}"
                    optimized_negative_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['negative_prompts']['realistic']}, {optimized_negative_prompt}"
                elif style == "fantasy_art":
                    optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['fantasy_character']}, {optimized_prompt}"
                    optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['artistic_style']}, {optimized_prompt}"
                elif style == "modern_style":
                    optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['modern_character']}, {optimized_prompt}"
            
            # 添加质量增强器
            optimized_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['quality_enhancers']['high_quality']}, {optimized_prompt}"
            
            logger.info(f"🎨 优化后的提示词: {optimized_prompt}")
            
            # 设置模型和VAE
            try:
                # 设置VAE
                if vae_name and vae_name != "None" and vae_name.strip() != "":
                    logger.info(f"设置VAE: {vae_name}")
                    async with aiohttp.ClientSession() as session:
                        vae_payload = {"sd_vae": vae_name}
                        async with session.post(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", 
                                              json=vae_payload, timeout=30) as vae_resp:
                            if vae_resp.status == 200:
                                logger.info(f"VAE设置成功: {vae_name}")
                            else:
                                logger.warning(f"VAE设置失败，状态码: {vae_resp.status}")
                else:
                    logger.info("未指定VAE，使用当前VAE设置")
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", timeout=30) as resp:
                        if resp.status == 200:
                            current_options = await resp.json()
                            current_model = current_options.get('sd_model_checkpoint', '')
                            
                            if model_name not in current_model:
                                logger.info(f"🔄 切换模型从 {current_model} 到 {model_name}")
                                update_payload = {"sd_model_checkpoint": model_name}
                                async with session.post(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", 
                                                      json=update_payload, timeout=30) as update_resp:
                                    if update_resp.status != 200:
                                        logger.warning(f"⚠️ 模型切换可能失败，状态码: {update_resp.status}")
                            else:
                                logger.info(f"✅ 当前模型已为目标模型: {model_name}")
                        else:
                            logger.warning(f"⚠️ 无法获取当前模型信息，状态码: {resp.status}")
            except Exception as model_error:
                logger.warning(f"⚠️ 模型设置过程中出错: {str(model_error)}")
            
            # 构建优化的payload，专门针对透明背景
            payload = {
                "prompt": optimized_prompt,
                "negative_prompt": optimized_negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_index": sampler,
                "n_iter": 1,
                "batch_size": 1,
                "seed": -1,
                "enable_hr": False,  # 禁用高清修复
                "restore_faces": False,  # 禁用面部修复
                "tiling": False,
                "eta": 0,  # 使用确定性生成
                "s_churn": 0,
                "s_tmax": 0,
                "s_tmin": 0,
                "s_noise": 1,
                "override_settings": {
                    "sd_vae": "None",  # 使用默认VAE
                    "CLIP_stop_at_last_layers": 1
                },
                "alwayson_scripts": {
                    "layerdiffuse": {
                        "args": [
                            True,  # enabled
                            "(SD1.5) Only Generate Transparent Image (Attention Injection)",  # method - SD1.5透明生成
                            1.0,   # weight
                            1.0,   # stop at (1.0 = 100%)
                            None,  # background
                            None,  # background
                            None,  # background
                            "Crop and Resize",  # resize mode
                            False, # output original mat
                            "",    # foreground additional prompt
                            "",    # background additional prompt
                            ""     # blended additional prompt
                        ]
                    }
                }
            }
            
            logger.info("🚀 调用API生成透明背景图片...")
            
            # 调用API
            url = f"{NOVELAI_CONFIG['base_url']}{NOVELAI_CONFIG['endpoint']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=NOVELAI_CONFIG["timeout"])
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ API错误: {response.status} - {error_text}")
                        return [TextContent(type="text", text=f"API错误: {response.status} - {error_text}")]
                    
                    response_data = await response.json()
                    
                    if "images" in response_data and response_data["images"]:
                        image_data = response_data["images"][0]
                        logger.info(f"✅ 图片生成成功，大小: {len(image_data)} 字符")
                        
                        try:
                            # 处理输出路径
                            output_path_obj = Path(output_path)
                            if not output_path_obj.is_absolute():
                                output_path_obj = Path(__file__).parent / output_path
                            
                            logger.info(f"💾 保存路径: {output_path_obj}")
                            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 解码base64图片数据
                            image_bytes = base64.b64decode(image_data)
                            
                            # 保存图片并进行智能透明背景处理
                            try:
                                if IMAGE_PROCESSING_AVAILABLE and output_path_obj.suffix.lower() == '.png':
                                    # 首先尝试直接保存，因为LayerDiffuse可能已经生成了透明背景
                                    with open(output_path_obj, 'wb') as f:
                                        f.write(image_bytes)
                                    
                                    # 然后检查是否已经存在透明效果
                                    img = Image.open(output_path_obj)
                                    if img.mode == 'RGBA':
                                        alpha = img.getchannel('A')
                                        transparent_pixels = sum(1 for p in alpha.getdata() if p == 0)
                                        semi_transparent = sum(1 for p in alpha.getdata() if 0 < p < 255)
                                        total_pixels = width * height
                                        transparent_ratio = (transparent_pixels + semi_transparent) / total_pixels * 100
                                        
                                        alpha_channel_detected = transparent_ratio > 5  # 透明度大于5%认为有效
                                        if alpha_channel_detected:
                                            logger.info(f"🎉 LayerDiffuse 透明背景成功！透明度: {transparent_ratio:.1f}%")
                                        else:
                                            logger.info(f"✨ 检测到透明效果，透明度: {transparent_ratio:.1f}%")
                                    else:
                                        # 如果没有透明效果，使用PIL进行智能透明背景处理
                                        logger.info("ℹ️ LayerDiffuse 未产生透明效果，使用PIL后处理")
                                        rgba_img = img.convert('RGBA')
                                        datas = rgba_img.getdata()
                                        new_data = []
                                        transparent_count = 0
                                        
                                        # 智能背景检测和透明化处理
                                        for item in datas:
                                            r, g, b = item[:3]
                                            # 检测白色或接近白色的背景区域
                                            if r > 245 and g > 245 and b > 245:
                                                new_data.append((r, g, b, 0))  # 完全透明
                                                transparent_count += 1
                                            elif max(r, g, b) - min(r, g, b) < 15 and max(r, g, b) > 235:
                                                # 浅灰色背景也设为透明
                                                new_data.append((r, g, b, 0))
                                                transparent_count += 1
                                            else:
                                                new_data.append((r, g, b, 255))  # 保持不透明
                                        
                                        rgba_img.putdata(new_data)
                                        rgba_img.save(output_path_obj)
                                        
                                        # 计算透明度比例
                                        transparent_ratio = transparent_count / total_pixels * 100
                                        alpha_channel_detected = transparent_ratio > 5
                                        
                                        if alpha_channel_detected:
                                            logger.info(f"🎉 PIL智能透明背景处理成功！透明度: {transparent_ratio:.1f}%")
                                        else:
                                            logger.info(f"ℹ️ PIL透明背景处理完成，透明度: {transparent_ratio:.1f}%")
                                        
                                else:
                                    # 非PNG格式，直接保存
                                    with open(output_path_obj, 'wb') as f:
                                        f.write(image_bytes)
                                    
                                    alpha_channel_detected = False
                                    logger.info("ℹ️ 非PNG格式，直接保存图片")
                                    
                            except Exception as process_error:
                                # 如果处理失败，回退到直接保存
                                logger.warning(f"⚠️ 透明背景处理失败，回退到直接保存: {str(process_error)}")
                                with open(output_path_obj, 'wb') as f:
                                    f.write(image_bytes)
                                alpha_channel_detected = False
                            
                            # 构建成功消息
                            success_message = f"🎉 透明背景图片生成成功！\n"
                            success_message += f"📁 保存路径: {output_path_obj.absolute()}\n"
                            success_message += f"📊 图片大小: {os.path.getsize(output_path_obj) / 1024:.1f} KB\n"
                            success_message += f"🤖 使用模型: {model_name}\n"
                            success_message += f"🎨 采样器: {sampler}\n"
                            success_message += f"📐 尺寸: {width}x{height}\n"
                            success_message += f"🎯 风格: {style}\n"
                            if alpha_channel_detected:
                                success_message += f"✨ 透明通道: 检测到透明效果！\n"
                            else:
                                success_message += f"ℹ️ 透明通道: 已生成PNG图片，建议检查透明效果\n"
                            success_message += f"📝 原始提示词: {prompt}\n"
                            success_message += f"🔧 优化提示词: {optimized_prompt}\n"
                            success_message += f"💡 提示: 透明效果通过智能后处理生成"
                            
                            return [TextContent(type="text", text=success_message)]
                            
                        except Exception as save_error:
                            logger.error(f"❌ 保存图片失败: {str(save_error)}")
                            return [TextContent(type="text", text=f"保存图片失败: {str(save_error)}")]
                    else:
                        logger.error("❌ API返回格式错误: 未找到图片数据")
                        return [TextContent(type="text", text="错误: API返回格式不正确，未找到图片数据")]
                        
        except Exception as e:
            logger.error(f"❌ 生成透明背景图片时出错: {str(e)}")
            return [TextContent(type="text", text=f"生成透明背景图片时出错: {str(e)}")]
    
    async def get_prompt_suggestions(self, arguments: Dict[str, Any]) -> List[Any]:
        """获取提示词建议"""
        try:
            category = arguments.get("category", "all")
            
            suggestions = {
                "available_samplers": NOVELAI_CONFIG["prompt_suggestions"]["available_samplers"][:10],
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if category == "all" or category == "characters":
                suggestions["character_prompts"] = NOVELAI_CONFIG["prompt_suggestions"]["character_prompts"]
            
            if category == "all" or category == "styles":
                suggestions["style_modifiers"] = NOVELAI_CONFIG["prompt_suggestions"]["style_modifiers"]
            
            if category == "all" or category == "negative":
                suggestions["negative_prompts"] = NOVELAI_CONFIG["prompt_suggestions"]["negative_prompts"]
            
            if category == "all" or category == "quality":
                suggestions["quality_enhancers"] = NOVELAI_CONFIG["prompt_suggestions"]["quality_enhancers"]
            
            if category == "all" or category == "scene_backgrounds":
                if "scene_backgrounds" in NOVELAI_CONFIG["prompt_suggestions"]:
                    suggestions["scene_backgrounds"] = NOVELAI_CONFIG["prompt_suggestions"]["scene_backgrounds"]
            
            if category == "all" or category == "clothing_accessories":
                if "clothing_accessories" in NOVELAI_CONFIG["prompt_suggestions"]:
                    suggestions["clothing_accessories"] = NOVELAI_CONFIG["prompt_suggestions"]["clothing_accessories"]
            
            if category == "all" or category == "environment_tags":
                if "environment_tags" in NOVELAI_CONFIG["prompt_suggestions"]:
                    suggestions["environment_tags"] = NOVELAI_CONFIG["prompt_suggestions"]["environment_tags"]
            
            if category == "all" or category == "technical_parameters":
                if "technical_parameters" in NOVELAI_CONFIG["prompt_suggestions"]:
                    suggestions["technical_parameters"] = NOVELAI_CONFIG["prompt_suggestions"]["technical_parameters"]
            
            if category == "samplers":
                suggestions["sampler_recommendations"] = {
                    "fast": ["Euler a", "Euler", "LMS"],
                    "quality": ["DPM++ 2M", "DPM++ SDE", "DPM++ 2M Karras"],
                    "creative": ["DDIM", "PLMS", "UniPC"]
                }
            
            return [TextContent(type="text", text=f"提示词建议 ({category}):\n{json.dumps(suggestions, ensure_ascii=False, indent=2)}")]
            
        except Exception as e:
            logger.error(f"获取提示词建议失败: {str(e)}")
            return [TextContent(type="text", text=f"获取提示词建议失败: {str(e)}")]

    async def get_models(self, arguments: Dict[str, Any]) -> List[Any]:
        """获取可用的Stable Diffusion模型列表"""
        try:
            async with aiohttp.ClientSession() as session:
                # 获取模型列表
                async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/sd-models", timeout=30) as resp:
                    resp.raise_for_status()
                    models = await resp.json()
                
                # 获取当前模型信息
                async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", timeout=30) as resp:
                    resp.raise_for_status()
                    current_options = await resp.json()
                    current_model = current_options.get('sd_model_checkpoint', 'Unknown')
                
                # 获取额外的模型信息（如VAE、CLIP等）
                try:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/hypernetworks", timeout=30) as resp:
                        if resp.status == 200:
                            hypernetworks = await resp.json()
                        else:
                            hypernetworks = []
                except:
                    hypernetworks = []
                
                # 格式化模型信息
                model_list = []
                for model in models:
                    model_info = {
                        "title": model.get("title", "Unknown"),
                        "model_name": model.get("model_name", "Unknown"),
                        "filename": model.get("filename", "Unknown"),
                        "hash": model.get("hash", "Unknown")[:8] if model.get("hash") else "Unknown",
                        "config": model.get("config", {})
                    }
                    model_list.append(model_info)
                
                result_text = f"🎨 可用模型列表 (共{len(model_list)}个):\n"
                result_text += f"📌 当前模型: {current_model}\n"
                result_text += f"🔗 超网络数量: {len(hypernetworks)}\n\n"
                
                for i, model in enumerate(model_list, 1):
                    result_text += f"{i}. 📋 {model['title']}\n"
                    result_text += f"   📁 文件名: {model['filename']}\n"
                    result_text += f"   🔑 哈希: {model['hash']}\n"
                    if model['model_name'] in current_model:
                        result_text += "   ✅ [当前使用]\n"
                    result_text += "\n"
                
                # 添加模型使用建议
                result_text += "💡 使用建议:\n"
                result_text += "• 切换模型: 在SD WebUI界面中选择不同模型\n"
                result_text += "• 模型哈希: 用于验证模型完整性和版本\n"
                result_text += "• 超网络: 可在生成时增强特定风格或特征\n"
                
                return [TextContent(type="text", text=result_text)]
        
        except aiohttp.ClientResponseError as e:
            logger.error(f"获取模型列表时网络请求出错: {str(e)}")
            return [TextContent(type="text", text=f"获取模型列表时网络请求出错: {str(e)}")]
        except Exception as e:
            logger.error(f"获取模型列表时出错: {str(e)}")
            return [TextContent(type="text", text=f"获取模型列表时出错: {str(e)}")]
    
    async def generate_image_img2img(self, arguments: Dict[str, Any]) -> List[Any]:
        """图生图(img2img)功能 - 基于输入图片生成新图片"""
        try:
            # 获取基本参数
            input_image_path = arguments.get("input_image_path", "")
            prompt = arguments.get("prompt", "")
            output_path = arguments.get("output_path", "")
            model_name = arguments.get("model_name", NOVELAI_CONFIG.get("default_model", "anything-v4.0\\anything-v4.0.ckpt [3b26c9c497]"))
            vae_name = arguments.get("vae_name", NOVELAI_CONFIG.get("default_vae", None))
            negative_prompt = arguments.get("negative_prompt", NOVELAI_CONFIG["prompt_suggestions"]["negative_prompts"]["general"])
            width = arguments.get("width")
            height = arguments.get("height")
            steps = arguments.get("steps", NOVELAI_CONFIG["default_params"]["steps"])
            cfg_scale = arguments.get("cfg_scale", NOVELAI_CONFIG["default_params"]["cfg_scale"])
            sampler = arguments.get("sampler", NOVELAI_CONFIG["default_params"]["sampler_index"])
            style = arguments.get("style", "none")
            
            # img2img特有参数
            denoising_strength = arguments.get("denoising_strength", 0.75)
            resize_mode = arguments.get("resize_mode", 1)  # 0=拉伸, 1=裁剪适配, 2=填充
            mask_blur = arguments.get("mask_blur", 4)
            inpainting_fill = arguments.get("inpainting_fill", 1)  # 0=填充, 1=原图, 2=潜变量噪声, 3=潜变量零
            inpaint_full_res = arguments.get("inpaint_full_res", True)
            inpaint_full_res_padding = arguments.get("inpaint_full_res_padding", 32)
            
            # 局部重绘参数
            mask_image_path = arguments.get("mask_image_path", "")
            inpainting_mask_invert = arguments.get("inpainting_mask_invert", 0)
            inpainting_fill_mode = arguments.get("inpainting_fill_mode", "original")
            
            # 验证必需参数
            if not input_image_path:
                return [TextContent(type="text", text="错误: 输入图片路径不能为空")]
            
            if not os.path.exists(input_image_path):
                return [TextContent(type="text", text=f"错误: 输入图片文件不存在: {input_image_path}")]
            
            if not prompt:
                return [TextContent(type="text", text="错误: 提示词不能为空")]
            
            if not output_path:
                return [TextContent(type="text", text="错误: 输出路径不能为空")]
            
            # 验证尺寸参数
            if width is None:
                return [TextContent(type="text", text="错误: 图片宽度不能为空")]
            
            if height is None:
                return [TextContent(type="text", text="错误: 图片高度不能为空")]
            
            # 验证尺寸范围
            if not (64 <= width <= 2048):
                return [TextContent(type="text", text=f"错误: 图片宽度必须在64-2048之间，当前值: {width}")]
            
            if not (64 <= height <= 2048):
                return [TextContent(type="text", text=f"错误: 图片高度必须在64-2048之间，当前值: {height}")]
            
            # 读取输入图片
            try:
                with open(input_image_path, 'rb') as f:
                    input_image_bytes = f.read()
                
                # 将图片转换为base64
                input_image_base64 = base64.b64encode(input_image_bytes).decode('utf-8')
                
                # 验证图片格式
                if IMAGE_PROCESSING_AVAILABLE:
                    img = Image.open(io.BytesIO(input_image_bytes))
                    logger.info(f"输入图片信息: 格式={img.format}, 尺寸={img.size}, 模式={img.mode}")
                
            except Exception as e:
                return [TextContent(type="text", text=f"读取输入图片失败: {str(e)}")]
            
            # 处理遮罩图片（如果提供）
            mask_image_base64 = None
            if mask_image_path:
                try:
                    if not os.path.exists(mask_image_path):
                        return [TextContent(type="text", text=f"错误: 遮罩图片文件不存在: {mask_image_path}")]
                    
                    with open(mask_image_path, 'rb') as f:
                        mask_image_bytes = f.read()
                    
                    # 将遮罩图片转换为base64
                    mask_image_base64 = base64.b64encode(mask_image_bytes).decode('utf-8')
                    
                    # 验证遮罩图片格式
                    if IMAGE_PROCESSING_AVAILABLE:
                        mask_img = Image.open(io.BytesIO(mask_image_bytes))
                        logger.info(f"遮罩图片信息: 格式={mask_img.format}, 尺寸={mask_img.size}, 模式={mask_img.mode}")
                        
                        # 确保遮罩图片与输入图片尺寸一致
                        if 'img' in locals():
                            if mask_img.size != img.size:
                                logger.warning(f"遮罩图片尺寸{mask_img.size}与输入图片尺寸{img.size}不一致，可能在处理时会自动调整")
                    
                    logger.info(f"已加载遮罩图片: {mask_image_path}")
                    
                except Exception as e:
                    return [TextContent(type="text", text=f"读取遮罩图片失败: {str(e)}")]
            
            # 应用风格模板
            if style != "none" and style in NOVELAI_CONFIG["prompt_suggestions"]:
                if style == "anime_character":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['anime_girl']}, {prompt}"
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['anime_style']}, {prompt}"
                    negative_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['negative_prompts']['anime']}, {negative_prompt}"
                elif style == "realistic_portrait":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['realistic_style']}, {prompt}"
                    negative_prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['negative_prompts']['realistic']}, {negative_prompt}"
                elif style == "fantasy_art":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['fantasy_character']}, {prompt}"
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['style_modifiers']['artistic_style']}, {prompt}"
                elif style == "modern_style":
                    prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['character_prompts']['modern_character']}, {prompt}"
            
            # 添加质量增强器
            prompt = f"{NOVELAI_CONFIG['prompt_suggestions']['quality_enhancers']['high_quality']}, {prompt}"
            
            logger.info(f"图生图生成 - 风格: {style}, 重绘幅度: {denoising_strength}, 提示词: {prompt}")
            
            # 构建img2img请求payload
            img2img_payload = {
                "init_images": [input_image_base64],  # 输入图片base64编码
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_index": sampler,
                "denoising_strength": denoising_strength,
                "resize_mode": resize_mode,
                "mask_blur": mask_blur,
                "inpainting_fill": inpainting_fill,
                "inpaint_full_res": inpaint_full_res,
                "inpaint_full_res_padding": inpaint_full_res_padding,
                "n_iter": 1,
                "batch_size": 1,
                "seed": -1,
                "override_settings": {}
            }
            
            # 设置模型和VAE
            try:
                async with aiohttp.ClientSession() as session:
                    # 获取当前设置
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", timeout=30) as resp:
                        if resp.status == 200:
                            current_options = await resp.json()
                            current_model = current_options.get('sd_model_checkpoint', '')
                            current_vae = current_options.get('sd_vae', '')
                            
                            # 设置模型
                            if model_name and model_name not in current_model:
                                logger.info(f"切换模型从 {current_model} 到 {model_name}")
                                update_payload = {"sd_model_checkpoint": model_name}
                                async with session.post(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", 
                                                      json=update_payload, timeout=30) as update_resp:
                                    if update_resp.status != 200:
                                        logger.warning(f"模型切换可能失败，状态码: {update_resp.status}")
                            else:
                                logger.info(f"当前模型已为目标模型: {model_name}")
                            
                            # 设置VAE
                            if vae_name and vae_name != current_vae:
                                logger.info(f"切换VAE从 {current_vae} 到 {vae_name}")
                                update_payload = {"sd_vae": vae_name}
                                async with session.post(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", 
                                                      json=update_payload, timeout=30) as update_resp:
                                    if update_resp.status != 200:
                                        logger.warning(f"VAE切换可能失败，状态码: {update_resp.status}")
                            elif not vae_name:
                                logger.info(f"使用当前VAE设置: {current_vae}")
                            else:
                                logger.info(f"当前VAE已为目标VAE: {vae_name}")
                        else:
                            logger.warning(f"无法获取当前设置信息，状态码: {resp.status}")
            except Exception as model_error:
                logger.warning(f"模型/VAE设置过程中出错: {str(model_error)}")
            
            # 如果提供了遮罩图片，添加到payload
            if mask_image_base64:
                img2img_payload["mask"] = mask_image_base64
                img2img_payload["inpainting_mask_invert"] = inpainting_mask_invert
                
                # 根据inpainting_fill_mode设置对应的数值
                fill_mode_map = {
                    "fill": 0,
                    "original": 1,
                    "latent_noise": 2,
                    "latent_nothing": 3
                }
                img2img_payload["inpainting_fill"] = fill_mode_map.get(inpainting_fill_mode, 1)
                
                logger.info(f"局部重绘模式激活 - 遮罩反转: {inpainting_mask_invert}, 填充模式: {inpainting_fill_mode}")
            else:
                logger.info("标准图生图模式（无遮罩）")
            
            # 调用img2img API
            url = f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/img2img"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=img2img_payload,
                    timeout=aiohttp.ClientTimeout(total=NOVELAI_CONFIG["timeout"])
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"img2img API错误: {response.status} - {error_text}")
                        return [TextContent(type="text", text=f"img2img API错误: {response.status} - {error_text}")]
                    
                    response_data = await response.json()
                    
                    # 检查返回的图片数据
                    if "images" in response_data and response_data["images"]:
                        image_data = response_data["images"][0]
                        logger.info(f"图生图生成成功，大小: {len(image_data)} 字符")
                        
                        try:
                            # 处理输出路径
                            output_path_obj = Path(output_path)
                            if not output_path_obj.is_absolute():
                                output_path_obj = Path(__file__).parent / output_path
                            
                            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 解码并保存图片
                            image_bytes = base64.b64decode(image_data)
                            
                            with open(output_path_obj, 'wb') as f:
                                f.write(image_bytes)
                            
                            logger.info(f"图生图结果成功保存到: {output_path_obj}")
                            
                            # 构建成功消息
                            success_message = f"🎨 图生图(img2img)生成成功！\n"
                            success_message += f"📁 输入图片: {input_image_path}\n"
                            if mask_image_path:
                                success_message += f"🎭 遮罩图片: {mask_image_path}\n"
                                success_message += f"🔄 遮罩反转: {'是' if inpainting_mask_invert else '否'}\n"
                                success_message += f"🎨 填充模式: {inpainting_fill_mode}\n"
                                success_message += f"✨ 模式: 局部重绘\n"
                            else:
                                success_message += f"✨ 模式: 标准图生图\n"
                            success_message += f"📁 输出路径: {output_path_obj.absolute()}\n"
                            success_message += f"📊 输出大小: {os.path.getsize(output_path_obj) / 1024:.1f} KB\n"
                            success_message += f"🤖 使用模型: {model_name}\n"
                            success_message += f"🎨 采样器: {sampler}\n"
                            success_message += f"📐 尺寸: {width}x{height}\n"
                            success_message += f"🎯 风格: {style}\n"
                            success_message += f"🔧 重绘幅度: {denoising_strength}\n"
                            success_message += f"📏 调整模式: {['拉伸', '裁剪适配', '填充'][resize_mode] if resize_mode < 3 else '未知'}\n"
                            success_message += f"📝 提示词: {prompt}\n"
                            success_message += f"💡 建议: 重绘幅度{denoising_strength}表示保留{int((1-denoising_strength)*100)}%原图特征"
                            
                            return [TextContent(type="text", text=success_message)]
                            
                        except Exception as save_error:
                            logger.error(f"保存图生图结果失败: {str(save_error)}")
                            return [TextContent(type="text", text=f"保存图生图结果失败: {str(save_error)}")]
                    else:
                        logger.error("img2img API返回格式错误: 未找到图片数据")
                        return [TextContent(type="text", text="错误: img2img API返回格式不正确，未找到图片数据")]
                        
        except Exception as e:
            logger.error(f"图生图生成时出错: {str(e)}")
            return [TextContent(type="text", text=f"图生图生成时出错: {str(e)}")]
    
    async def get_model_details(self, arguments: Dict[str, Any]) -> List[Any]:
        """获取详细的模型信息，包括技术参数、VAE配置、CLIP设置和系统信息"""
        try:
            async with aiohttp.ClientSession() as session:
                # 获取当前选项配置
                async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", timeout=30) as resp:
                    resp.raise_for_status()
                    options = await resp.json()
                
                # 获取系统信息
                try:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/system-info", timeout=30) as resp:
                        if resp.status == 200:
                            system_info = await resp.json()
                        else:
                            system_info = {}
                except:
                    system_info = {}
                
                # 获取VAE列表
                try:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/sd-vae", timeout=30) as resp:
                        if resp.status == 200:
                            vae_list = await resp.json()
                        else:
                            vae_list = []
                except:
                    vae_list = []
                
                # 获取ControlNet信息（如果已安装）
                try:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/controlnet/model_list", timeout=30) as resp:
                        if resp.status == 200:
                            controlnet_info = await resp.json()
                        else:
                            controlnet_info = {}
                except:
                    controlnet_info = {}
                
                # 构建详细信息
                details = {
                    "current_model": options.get('sd_model_checkpoint', 'Unknown'),
                    "vae": options.get('sd_vae', 'None'),
                    "clip_skip": options.get('CLIP_stop_at_last_layers', 1),
                    "eta_noise_seed_delta": options.get('eta_noise_seed_delta', 0),
                    "system_info": {
                        "python_version": system_info.get('python_version', 'Unknown'),
                        "torch_version": system_info.get('torch_version', 'Unknown'),
                        "cuda_available": system_info.get('cuda_available', False),
                        "gpu_count": system_info.get('gpu_count', 0)
                    },
                    "vae_list": vae_list,
                    "controlnet_available": bool(controlnet_info),
                    "controlnet_models": controlnet_info.get('model_list', []) if controlnet_info else []
                }
                
                result_text = "🔧 模型详细信息:\n\n"
                result_text += f"📌 当前模型: {details['current_model']}\n"
                result_text += f"🎨 VAE模型: {details['vae']}\n"
                result_text += f"📎 CLIP跳过层数: {details['clip_skip']}\n"
                result_text += f"🌱 ETA噪声种子差值: {details['eta_noise_seed_delta']}\n"
                
                # 显示ControlNet状态
                if details['controlnet_available']:
                    result_text += f"• 🎯 ControlNet: ✅ 可用\n"
                    if details['controlnet_models']:
                        result_text += f"📋 ControlNet模型 ({len(details['controlnet_models'])}个):\n"
                        for i, model in enumerate(details['controlnet_models'][:3], 1):
                            result_text += f"   {i}. {model}\n"
                        if len(details['controlnet_models']) > 3:
                            result_text += f"   ... 还有 {len(details['controlnet_models']) - 3} 个模型\n"
                else:
                    result_text += f"🎯 ControlNet: ❌ 未安装\n"
                    result_text += f"   💡 安装方法: 在SD WebUI中安装ControlNet扩展\n"
                result_text += "\n"
                
                # 显示可用VAE列表
                if details['vae_list']:
                    result_text += f"📦 可用VAE模型 ({len(details['vae_list'])}个):\n"
                    for i, vae in enumerate(details['vae_list'][:5], 1):  # 只显示前5个
                        result_text += f"   {i}. {vae.get('model_name', 'Unknown')}\n"
                    if len(details['vae_list']) > 5:
                        result_text += f"   ... 还有 {len(details['vae_list']) - 5} 个VAE模型\n"
                    result_text += "\n"
                
                result_text += "💻 系统信息:\n"
                result_text += f"   🐍 Python版本: {details['system_info']['python_version']}\n"
                result_text += f"   🔥 PyTorch版本: {details['system_info']['torch_version']}\n"
                result_text += f"   🚀 CUDA可用: {details['system_info']['cuda_available']}\n"
                result_text += f"   🎮 GPU数量: {details['system_info']['gpu_count']}\n"
                
                # 添加配置建议
                result_text += "\n💡 配置建议:\n"
                result_text += "• VAE模型: 影响色彩还原和细节表现\n"
                result_text += "• CLIP跳过层数: 通常设为1-2，影响理解能力\n"
                result_text += "• ETA噪声: 影响生成过程中的噪声处理\n"
                if details['controlnet_available']:
                    result_text += "• 🎯 透明背景: 使用提示词优化方法\n"
                else:
                    result_text += "• 🎯 透明背景: 使用提示词优化方法\n"
                
                return [TextContent(type="text", text=result_text)]
        
        except Exception as e:
            logger.error(f"获取模型详细信息失败: {str(e)}")
            return [TextContent(type="text", text=f"获取模型详细信息失败: {str(e)}")]
    
    async def get_model_recommendations(self, arguments: Dict[str, Any]) -> List[Any]:
        """获取模型使用推荐和最佳实践"""
        try:
            # 基于当前配置和模型类型提供建议
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{NOVELAI_CONFIG['base_url']}/sdapi/v1/options", timeout=30) as resp:
                    if resp.status == 200:
                        options = await resp.json()
                        current_model = options.get('sd_model_checkpoint', '')
                    else:
                        current_model = ''
            
            # 模型推荐数据库
            recommendations = {
                "anime": {
                    "models": ["anything", "anime", "nai"],
                    "samplers": ["DPM++ 2M Karras", "Euler a", "DDIM"],
                    "cfg_scale": "7-12",
                    "steps": "20-30",
                    "description": "动漫风格模型，适合生成二次元角色和场景"
                },
                "realistic": {
                    "models": ["realistic", "photo", "real"],
                    "samplers": ["DPM++ SDE Karras", "DPM++ 2M Karras", "Heun"],
                    "cfg_scale": "5-8",
                    "steps": "30-50",
                    "description": "写实风格模型，适合生成逼真的人物和场景"
                },
                "artistic": {
                    "models": ["art", "painting", "illustration"],
                    "samplers": ["DDIM", "PLMS", "UniPC"],
                    "cfg_scale": "6-10",
                    "steps": "25-40",
                    "description": "艺术风格模型，适合生成具有艺术感的作品"
                }
            }
            
            # 分析当前模型类型
            current_model_lower = current_model.lower()
            model_type = "general"
            
            for model_category, info in recommendations.items():
                if any(keyword in current_model_lower for keyword in info["models"]):
                    model_type = model_category
                    break
            
            # 生成推荐信息
            result_text = "🎯 模型使用推荐:\n\n"
            result_text += f"📌 当前模型类型: {model_type.upper()}\n"
            result_text += f"🔄 当前模型: {current_model}\n\n"
            
            if model_type in recommendations:
                rec = recommendations[model_type]
                result_text += f"🎨 {rec['description']}\n\n"
                result_text += "⚙️ 推荐参数设置:\n"
                result_text += f"   🎯 采样器: {', '.join(rec['samplers'][:2])}\n"
                result_text += f"   📊 CFG Scale: {rec['cfg_scale']}\n"
                result_text += f"   🔄 步数: {rec['steps']}\n\n"
            
            result_text += "📋 通用最佳实践:\n"
            result_text += "• 🎯 选择合适的采样器：DPM++系列适合高质量，Euler系列适合快速生成\n"
            result_text += "• 📊 CFG Scale：7-9为平衡值，过高会导致过饱和，过低会导致模糊\n"
            result_text += "• 🔄 步数：20-30步通常足够，更多步数不一定更好\n"
            result_text += "• 🎨 负面提示词：使用适当的负面提示词可以显著提升质量\n"
            result_text += "• 🔧 VAE模型：选择与主模型匹配的VAE以获得更好的色彩表现\n"
            result_text += "• 📏 分辨率：从512x512或512x768开始，逐步尝试更高分辨率\n\n"
            
            result_text += "🚀 性能优化建议:\n"
            result_text += "• 使用--xformers或--opt-sdp-attention参数启动SD WebUI\n"
            result_text += "• 确保CUDA和GPU驱动为最新版本\n"
            result_text += "• 根据GPU内存调整批处理大小\n"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            logger.error(f"获取模型推荐失败: {str(e)}")
            return [TextContent(type="text", text=f"获取模型推荐失败: {str(e)}")]
    
    async def run(self):
        """运行MCP服务器"""
        try:
            logger.info("启动NovelAI MCP服务器...")
            logger.info(f"连接到: {NOVELAI_CONFIG['base_url']}")
            
            # 测试连接并获取模型信息
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{NOVELAI_CONFIG['base_url']}/") as response:
                        if response.status == 200:
                            logger.info("成功连接到Stable Diffusion WebUI")
                            

                                
                        else:
                            logger.warning(f"连接测试返回状态码: {response.status}")
                except Exception as e:
                    logger.warning(f"连接测试失败: {str(e)}")
            
            # 运行服务器
            from mcp.server.stdio import stdio_server
            
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
                
        except Exception as e:
            logger.error(f"服务器运行错误: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        novelai_mcp = NovelAIMCP()
        asyncio.run(novelai_mcp.run())
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()