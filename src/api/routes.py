"""API routes - OpenAI compatible endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
from typing import List, Optional, Any, Dict
import json
import re
import time
import aiosqlite
from pydantic import BaseModel
from ..core.auth import verify_api_key_header
from ..core.models import ChatCompletionRequest
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..services.sora_client import _generate_sentinel_token_lightweight
from ..core.logger import debug_logger

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None

def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler

def _extract_remix_id(text: str) -> str:
    """Extract remix ID from text

    Supports two formats:
    1. Full URL: https://sora.chatgpt.com/p/s_68e3a06dcd888191b150971da152c1f5
    2. Short ID: s_68e3a06dcd888191b150971da152c1f5

    Args:
        text: Text to search for remix ID

    Returns:
        Remix ID (s_[a-f0-9]{32}) or empty string if not found
    """
    if not text:
        return ""

    # Match Sora share link format: s_[a-f0-9]{32}
    match = re.search(r's_[a-f0-9]{32}', text)
    if match:
        return match.group(0)

    return ""

@router.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key_header)):
    """List available models"""
    models = []

    for model_id, config in MODEL_CONFIG.items():
        description = f"{config['type'].capitalize()} generation"
        if config['type'] == 'image':
            description += f" - {config['width']}x{config['height']}"
        elif config['type'] == 'video':
            description += f" - {config['orientation']}"
        elif config['type'] == 'prompt_enhance':
            description += f" - {config['expansion_level']} ({config['duration_s']}s)"

        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "sora2api",
            "description": description
        })

    return {
        "object": "list",
        "data": models
    }

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header),
    http_request: Request = None
):
    """Create chat completion (unified endpoint for image and video generation)"""
    start_time = time.time()

    try:
        # Log client request
        debug_logger.log_request(
            method="POST",
            url="/v1/chat/completions",
            headers=dict(http_request.headers) if http_request else {},
            body=request.dict(),
            source="Client"
        )

        # Extract prompt from messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        last_message = request.messages[-1]
        content = last_message.content

        # Handle both string and array format (OpenAI multimodal)
        prompt = ""
        image_data = request.image  # Default to request.image if provided
        video_data = request.video  # Video parameter
        remix_target_id = request.remix_target_id  # Remix target ID

        if isinstance(content, str):
            # Simple string format
            prompt = content
            # Extract remix_target_id from prompt if not already provided
            if not remix_target_id:
                remix_target_id = _extract_remix_id(prompt)
        elif isinstance(content, list):
            # Array format (OpenAI multimodal)
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        prompt = item.get("text", "")
                        # Extract remix_target_id from prompt if not already provided
                        if not remix_target_id:
                            remix_target_id = _extract_remix_id(prompt)
                    elif item.get("type") == "image_url":
                        # Extract base64 image from data URI
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:image"):
                            # Extract base64 data from data URI
                            if "base64," in url:
                                image_data = url.split("base64,", 1)[1]
                            else:
                                image_data = url
                        else:
                            # It's a URL, pass it as-is (will be downloaded in generation_handler)
                            image_data = url
                    elif item.get("type") == "video_url":
                        # Extract video from video_url
                        video_url = item.get("video_url", {})
                        url = video_url.get("url", "")
                        if url.startswith("data:video") or url.startswith("data:application"):
                            # Extract base64 data from data URI
                            if "base64," in url:
                                video_data = url.split("base64,", 1)[1]
                            else:
                                video_data = url
                        else:
                            # It's a URL, pass it as-is (will be downloaded in generation_handler)
                            video_data = url
        else:
            raise HTTPException(status_code=400, detail="Invalid content format")

        # Validate model
        if request.model not in MODEL_CONFIG:
            raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}")

        # Check if this is a video model
        model_config = MODEL_CONFIG[request.model]
        is_video_model = model_config["type"] == "video"

        # For video models with video parameter, we need streaming
        if is_video_model and (video_data or remix_target_id):
            if not request.stream:
                # Non-streaming mode: only check availability
                result = None
                async for chunk in generation_handler.handle_generation_with_retry(
                    model=request.model,
                    prompt=prompt,
                    image=image_data,
                    video=video_data,
                    remix_target_id=remix_target_id,
                    stream=False
                ):
                    result = chunk

                if result:
                    duration_ms = (time.time() - start_time) * 1000
                    response_data = json.loads(result)
                    debug_logger.log_response(
                        status_code=200,
                        headers={"Content-Type": "application/json"},
                        body=response_data,
                        duration_ms=duration_ms,
                        source="Client"
                    )
                    return JSONResponse(content=response_data)
                else:
                    duration_ms = (time.time() - start_time) * 1000
                    error_response = {
                        "error": {
                            "message": "Availability check failed",
                            "type": "server_error",
                            "param": None,
                            "code": None
                        }
                    }
                    debug_logger.log_response(
                        status_code=500,
                        headers={"Content-Type": "application/json"},
                        body=error_response,
                        duration_ms=duration_ms,
                        source="Client"
                    )
                    return JSONResponse(
                        status_code=500,
                        content=error_response
                    )

        # Handle streaming
        if request.stream:
            async def generate():
                try:
                    async for chunk in generation_handler.handle_generation_with_retry(
                        model=request.model,
                        prompt=prompt,
                        image=image_data,
                        video=video_data,
                        remix_target_id=remix_target_id,
                        stream=True
                    ):
                        yield chunk
                except Exception as e:
                    # Try to parse structured error (JSON format)
                    error_data = None
                    try:
                        error_data = json.loads(str(e))
                    except:
                        pass

                    # Return OpenAI-compatible error format
                    if error_data and isinstance(error_data, dict) and "error" in error_data:
                        # Structured error (e.g., unsupported_country_code)
                        error_response = error_data
                    else:
                        # Generic error
                        error_response = {
                            "error": {
                                "message": str(e),
                                "type": "server_error",
                                "param": None,
                                "code": None
                            }
                        }
                    error_chunk = f'data: {json.dumps(error_response)}\n\n'
                    yield error_chunk
                    yield 'data: [DONE]\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response (availability check only)
            result = None
            async for chunk in generation_handler.handle_generation_with_retry(
                model=request.model,
                prompt=prompt,
                image=image_data,
                video=video_data,
                remix_target_id=remix_target_id,
                stream=False
            ):
                result = chunk

            if result:
                duration_ms = (time.time() - start_time) * 1000
                response_data = json.loads(result)
                debug_logger.log_response(
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                    body=response_data,
                    duration_ms=duration_ms,
                    source="Client"
                )
                return JSONResponse(content=response_data)
            else:
                # Return OpenAI-compatible error format
                duration_ms = (time.time() - start_time) * 1000
                error_response = {
                    "error": {
                        "message": "Availability check failed",
                        "type": "server_error",
                        "param": None,
                        "code": None
                    }
                }
                debug_logger.log_response(
                    status_code=500,
                    headers={"Content-Type": "application/json"},
                    body=error_response,
                    duration_ms=duration_ms,
                    source="Client"
                )
                return JSONResponse(
                    status_code=500,
                    content=error_response
                )

    except Exception as e:
        # Return OpenAI-compatible error format
        duration_ms = (time.time() - start_time) * 1000
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": None
            }
        }
        debug_logger.log_error(
            error_message=str(e),
            status_code=500,
            response_text=str(e),
            source="Client"
        )
        debug_logger.log_response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            body=error_response,
            duration_ms=duration_ms,
            source="Client"
        )
        return JSONResponse(
            status_code=500,
            content=error_response
        )


class CreateTaskRequest(BaseModel):
    """Non-stream task creation request for polling-style clients."""
    type: str  # text2img/img2img/text2video/img2video/character_only
    model: Optional[str] = None
    prompt: Optional[str] = ""
    image: Optional[str] = None  # base64/data URI/http(s) URL
    video: Optional[str] = None  # base64/data URI/http(s) URL (for character_only)
    remix_target_id: Optional[str] = None  # reserved


class GetSentinelTokenRequest(BaseModel):
    """Get sentinel token request.

    Supports multiple naming styles for compatibility:
    - proxy_url / proxyUrl
    - use_browser / useBrowser / browser
    """
    proxy_url: Optional[str] = None
    proxyUrl: Optional[str] = None
    use_browser: Optional[bool] = None
    useBrowser: Optional[bool] = None
    browser: Optional[bool] = None


@router.post("/v1/sentinel_token")
async def get_sentinel_token(
    request: GetSentinelTokenRequest,
    api_key: str = Depends(verify_api_key_header),
):
    """Get sentinel token (browser preferred, fallback to manual PoW).

    Returns:
        { "sentinel_token": str, "user_agent": str }
    """
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    proxy_url = request.proxy_url or request.proxyUrl
    use_browser = (
        request.use_browser
        if request.use_browser is not None
        else request.useBrowser
        if request.useBrowser is not None
        else request.browser
        if request.browser is not None
        else False
    )

    # Browser UA used by lightweight Playwright path (align with sora_client.py)
    browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

    sentinel_token: Optional[str] = None
    user_agent: str = browser_user_agent

    # 1) If requested, try browser/cached token first
    if use_browser:
        try:
            sentinel_token = await _generate_sentinel_token_lightweight(proxy_url)
            if sentinel_token:
                return {"sentinel_token": sentinel_token, "user_agent": user_agent}
        except Exception as e:
            # Any browser failure falls back to manual PoW
            debug_logger.log_info(f"[Sentinel] Browser token failed, fallback to manual: {e}")
            sentinel_token = None

    # 2) Manual PoW (always as fallback)
    try:
        sentinel_token, user_agent = await generation_handler.sora_client._generate_sentinel_token(
            token=None,
            user_agent=None,
            proxy_url=proxy_url,
        )
        return {"sentinel_token": sentinel_token, "user_agent": user_agent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/tasks")
async def create_task(
    request: CreateTaskRequest,
    api_key: str = Depends(verify_api_key_header),
):
    """Create a generation task and return task_id immediately (poll via GET /v1/tasks/{task_id})."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    task_type = (request.type or "").strip()
    prompt = request.prompt or ""

    def _media_meta(v: Optional[str]) -> Optional[Dict[str, Any]]:
        """Avoid storing large base64 payloads in request_logs."""
        if not v:
            return None
        s = str(v)
        kind = "url" if (s.startswith("http://") or s.startswith("https://")) else "inline"
        # Keep only metadata; never store the raw payload here.
        return {"kind": kind, "length": len(s)}

    try:
        if task_type in ("text2img", "img2img"):
            if not request.model:
                raise HTTPException(status_code=400, detail="model is required for image tasks")
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required for image tasks")
            image = request.image if task_type == "img2img" else None
            task_id = await generation_handler.submit_image_task(model=request.model, prompt=prompt, image=image)
        elif task_type in ("text2video", "img2video"):
            if not request.model:
                raise HTTPException(status_code=400, detail="model is required for video tasks")
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required for video tasks")
            image = request.image if task_type == "img2video" else None
            task_id = await generation_handler.submit_video_task(model=request.model, prompt=prompt, image=image)
        elif task_type == "character_only":
            if not request.video:
                raise HTTPException(status_code=400, detail="video is required for character_only")
            task_id = await generation_handler.submit_character_only_task(video=request.video)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task type: {task_type}")

        # Best-effort: register request log for admin "请求日志" panel.
        # Use status_code=-1 to indicate in-progress; task progress is resolved via task_id.
        try:
            await generation_handler._log_request(
                token_id=None,
                operation=f"create_task:{task_type}",
                request_data={
                    "type": task_type,
                    "model": request.model,
                    "prompt": prompt,
                    "image": _media_meta(request.image) if task_type in ("img2img", "img2video") else None,
                    "video": _media_meta(request.video) if task_type == "character_only" else None,
                    "remix_target_id": request.remix_target_id,
                },
                response_data={},
                status_code=-1,
                duration=-1.0,
                task_id=task_id,
            )
        except Exception:
            # Never block task creation due to logging failure
            pass

        return {"task_id": task_id, "status": "processing"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key_header),
):
    """Get task status/progress/result for polling-style clients."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    task = await generation_handler.db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    result_urls: Any = None
    if task.result_urls:
        try:
            result_urls = json.loads(task.result_urls)
        except Exception:
            result_urls = task.result_urls

    def _dt(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "model": task.model,
        "prompt": task.prompt,
        "result_urls": result_urls,
        "post_id": getattr(task, "post_id", None),
        "watermark_free_url": getattr(task, "watermark_free_url", None),
        "source_result_url": getattr(task, "source_result_url", None),
        "error_message": task.error_message,
        "created_at": _dt(task.created_at),
        "completed_at": _dt(task.completed_at),
    }


@router.get("/v1/stats/video-dispatch")
async def get_video_dispatch_stats(api_key: str = Depends(verify_api_key_header)):
    """Get dispatch stats for video generation.

    Returns:
        {
          "effectiveTokenCount": int,     # 当前可用（可派发）token 数量（视频维度）
          "globalMax": int,               # 当前可用 token 的最高并发（按 token.video_concurrency 求和，<=0 按 1）
          "runningTotal": int,            # 当前运行中任务数（tasks.status='processing' 且 model LIKE 'sora2%'）
          "asOf": "2026-02-03T12:34:56"
        }

    Notes:
        - 该接口用于上游（nodeserve / shuzhi-java）做“全局并发”调度判断。
        - runningTotal 仅统计视频模型（model 前缀 sora2*），避免图片任务影响视频并发调度。
    """
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    now = datetime.now()

    # 1) 计算“当前可用 token 的最高并发”
    # 口径与 load_balancer(for_video_generation=True) 过滤保持一致（视频启用 + 支持Sora2 + 不在 cooldown）
    tokens = await generation_handler.token_manager.get_active_tokens()

    effective_tokens = []
    for t in tokens or []:
        if t is None or not getattr(t, "id", None):
            continue
        if not getattr(t, "video_enabled", False):
            continue
        if not getattr(t, "sora2_supported", False):
            continue

        # cooldown 到期则尝试刷新（失败则按当前值兜底，不让统计接口抛错）
        cooldown_until = getattr(t, "sora2_cooldown_until", None)
        if cooldown_until and cooldown_until <= now:
            try:
                await generation_handler.token_manager.refresh_sora2_remaining_if_cooldown_expired(t.id)
                t = await generation_handler.db.get_token(t.id) or t
            except Exception:
                pass

        cooldown_until = getattr(t, "sora2_cooldown_until", None)
        if cooldown_until and cooldown_until > now:
            continue

        effective_tokens.append(t)

    def _norm_concurrency(c: Optional[int]) -> int:
        # 约定：video_concurrency 非正数（含 -1 不限制）在“最高并发统计”中按 1 计，避免出现无限并发。
        if c is None or c <= 0:
            return 1
        return int(c)

    global_max = 0
    for t in effective_tokens:
        try:
            global_max += _norm_concurrency(getattr(t, "video_concurrency", None))
        except Exception:
            global_max += 1

    # 2) 统计 runningTotal（processing 的 sora2 视频任务）
    running_total = 0
    try:
        async with aiosqlite.connect(generation_handler.db.db_path) as db:
            cur = await db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ? AND model LIKE ?",
                ("processing", "sora2%"),
            )
            row = await cur.fetchone()
            running_total = int(row[0] or 0) if row else 0
    except Exception:
        # 统计失败不影响主流程：返回 0（上游会更保守地不派发）
        running_total = 0

    return {
        "effectiveTokenCount": len(effective_tokens),
        "globalMax": int(global_max),
        "runningTotal": int(running_total),
        "asOf": now.isoformat(),
    }
