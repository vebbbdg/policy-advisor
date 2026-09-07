"""
GenAI Chatbot - Full-Stack LLM Application with RAG
工业级LLM对话系统主入口
- 多会话管理
- RAG知识库检索
- SSE流式响应
- 全局日志
- CORS跨域支持
"""
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core.model import init_llm_model
from core.memory import keep_recent_messages, inject_rag_prompt
from core.session import session_manager
from core.rag import rag_engine, translate_query, UPLOAD_DIR
from core.uploads import validate_upload, safe_stored_name, MAX_UPLOAD_BYTES
from core.usage import usage_tracker
from core.auth import UserContext, get_current_user, issue_guest_token, issue_login_token
from core.logger import logger

# ====================== 应用初始化 ======================
app = FastAPI(
    title="GenAI Chatbot | Full-Stack LLM Application",
    description="Production-grade RAG-powered chatbot with streaming responses",
    version="2.0.0"
)

# CORS跨域配置：通过环境变量CORS_ORIGINS控制（逗号分隔）
# 注意：通配符"*"与credentials=True不兼容，仅在显式指定源时启用凭证
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 速率限制（阶段 3.3-B）======================
# 按客户端 IP 限流：3.2 认证落地后可升级为按用户限流。
# 唯一花钱的是 DeepSeek LLM 调用（chat-stream），故限额精准打在该端点。
# 部署到反向代理（阶段 4 Render）后需改为信任 X-Forwarded-For，否则拿到的都是代理 IP。
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 全局配置
MAX_HISTORY_PAIRS = 15
RAG_ENABLED = True
# 每个 IP 每小时可调用 chat-stream 的次数（防 API 费用烧穿），可用环境变量覆盖
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "20/hour")
# 访客（未登录）最多可发送的消息条数，用尽后引导登录（阶段 3.2 获客转化）
GUEST_MESSAGE_LIMIT = int(os.getenv("GUEST_MESSAGE_LIMIT", "5") or 5)
model = init_llm_model()

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# ====================== 数据模型 ======================
class ChatInput(BaseModel):
    message: str
    session_id: str | None = None
    use_rag: bool = True


class CreateSessionInput(BaseModel):
    title: str = "New Chat"


class LoginInput(BaseModel):
    email: str


# ====================== 页面路由 ======================
@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ====================== 认证 API（阶段 3.2，公开端点，无需 token）======================
@app.post("/api/auth/guest")
async def auth_guest():
    """领取访客 token：前端首次访问调用，可发 GUEST_MESSAGE_LIMIT 条消息，用尽引导登录"""
    token, user_id = issue_guest_token()
    return {
        "token": token, "user_id": user_id, "is_guest": True,
        "message_limit": GUEST_MESSAGE_LIMIT,
    }


@app.post("/api/auth/login")
async def auth_login(data: LoginInput):
    """
    【占位登录】接受邮箱即签发非访客 token。
    TODO(阶段 3.2 完整实现)：接入邮箱验证码 / Google OAuth 时，在此加凭据校验后再签发。
    """
    email = (data.email or "").strip()
    if not email or "@" not in email:
        raise HTTPException(400, "a valid email is required")
    token, user_id = issue_login_token(email)
    logger.info(f"Placeholder login: {user_id}")
    return {"token": token, "user_id": user_id, "is_guest": False}


# ====================== 会话管理 API ======================
@app.post("/api/sessions")
async def create_session(data: CreateSessionInput, user: UserContext = Depends(get_current_user)):
    """创建新会话（归属当前用户）"""
    session_id = session_manager.create_session(data.title, user_id=user.user_id)
    logger.info(f"New session created: {session_id} (user={user.user_id})")
    return {"session_id": session_id, "title": data.title}


@app.get("/api/sessions")
async def list_sessions(user: UserContext = Depends(get_current_user)):
    """获取当前用户的会话列表（左侧边栏，按 user_id 隔离）"""
    sessions = session_manager.list_sessions(user_id=user.user_id)
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: UserContext = Depends(get_current_user)):
    """删除指定会话（仅限本人；越权/不存在一律 404，不泄露他人会话是否存在）"""
    if not session_manager.owns_session(session_id, user.user_id):
        raise HTTPException(404, "session not found")
    session_manager.delete_session(session_id)
    logger.info(f"Session deleted: {session_id}")
    return {"status": "success"}


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str, user: UserContext = Depends(get_current_user)):
    """重置指定会话对话（仅限本人会话）"""
    if not session_manager.owns_session(session_id, user.user_id):
        raise HTTPException(404, "session not found")
    session_manager.reset_session(session_id)
    return {"status": "success"}


# ====================== 核心对话 API ======================
@app.post("/api/chat-stream")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_stream(request: Request, data: ChatInput, user: UserContext = Depends(get_current_user)):
    """SSE流式对话接口（支持RAG增强）；需认证，每 IP 每小时限 CHAT_RATE_LIMIT 次"""
    # 阶段 3.2：访客配额——发满 GUEST_MESSAGE_LIMIT 条即拒绝并引导登录（403，与限流 429 区分）
    if user.is_guest and session_manager.count_user_messages(user.user_id) >= GUEST_MESSAGE_LIMIT:
        raise HTTPException(
            status_code=403,
            detail=f"Guest limit reached ({GUEST_MESSAGE_LIMIT} messages). Please log in to continue.",
        )

    # 获取或创建会话（校验归属，防越权访问他人会话；无有效归属则新建一个属于当前用户的会话）
    session_id = data.session_id
    if not session_id or not session_manager.owns_session(session_id, user.user_id):
        session_id = session_manager.create_session(user_id=user.user_id)

    try:
        # 1. 保存用户消息
        session_manager.add_message(session_id, "user", data.message)

        # 2. 获取历史并裁剪
        messages = session_manager.get_messages(session_id)
        optimized_msgs = keep_recent_messages(messages, MAX_HISTORY_PAIRS)

        # 3. RAG检索增强（如果开启）；同步向量检索移入线程池，避免阻塞事件循环
        context = None
        citations = []
        if data.use_rag and RAG_ENABLED:
            try:
                # 阶段 2.1：中文提问先翻译成英文再检索（英文语料+英文嵌入模型）；纯英文提问原样返回
                query = await asyncio.to_thread(translate_query, data.message)
                # 阶段 3.3-A：按 RETRIEVER_MODE 环境变量分派检索模式（默认 hybrid，评估 Recall@3=0.906）
                docs = await asyncio.to_thread(rag_engine.retrieve_by_mode, query, 3)
                if docs:
                    context = rag_engine.format_docs(docs)
                    # 阶段 2.2：打包引用元数据（来源+日期+主题），随 SSE 返回前端渲染来源卡片
                    # 注意：top-k 片段可能来自同一文档，按 source_url 去重，每个来源只展示一张卡片
                    seen_urls = set()
                    for d in docs:
                        meta = d.metadata
                        url = meta.get("source_url", "")
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        citations.append({
                            "title": meta.get("title") or meta.get("source", "unknown"),
                            "source_url": url,
                            "crawl_date": meta.get("crawl_date", ""),
                            "policy_topic": meta.get("policy_topic", ""),
                        })
                    # 将检索到的文档注入系统提示词
                    rag_prompt = (
                        "Use the following retrieved context to answer the user's question. "
                        "If the answer is not in the context, say you don't know based on the documents. "
                        f"\n\nRetrieved context:\n{context}"
                    )
                    # 重建system消息为新dict，不污染会话存储中的原始prompt
                    optimized_msgs = inject_rag_prompt(optimized_msgs, rag_prompt)
                    logger.info(f"RAG: injected {len(context)} chars of context, {len(citations)} citations")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        def stream_generator():
            full_reply = ""
            usage = None
            try:
                for chunk in model.stream(optimized_msgs):
                    # 阶段 3.3-C：流式最后一个 chunk 携带 usage_metadata（需 model 开启 stream_usage）
                    if getattr(chunk, "usage_metadata", None):
                        usage = chunk.usage_metadata
                    if chunk.content:
                        full_reply += chunk.content
                        # SSE格式：JSON数据包含内容和session_id
                        payload = json.dumps({
                            "content": chunk.content,
                            "session_id": session_id,
                            "rag_used": context is not None
                        })
                        yield f"data: {payload}\n\n"

                # 完成标记：携带引用元数据（阶段 2.2）
                yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'citations': citations})}\n\n"
                # 保存AI回复
                session_manager.add_message(session_id, "assistant", full_reply)
                # 阶段 3.3-C：累计本次对话的 token 消耗（超日预算会告警，仅告警不拦截）
                usage_tracker.record(usage)
                logger.info(f"Session {session_id}: response completed ({len(full_reply)} chars)")

            except Exception as e:
                error_msg = json.dumps({"error": str(e)})
                yield f"data: {error_msg}\n\n"
                logger.error(f"Stream error in session {session_id}: {e}")

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== RAG 知识库 API ======================
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档到知识库（扩展名白名单 + 大小限制 + 安全重命名）"""
    # 扩展名白名单校验
    try:
        validate_upload(file.filename, size=0)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 分块读取并强制大小上限，避免超大文件占满内存
    buffer = bytearray()
    while chunk := await file.read(1024 * 1024):
        buffer.extend(chunk)
        if len(buffer) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES} bytes)")

    # uuid前缀重命名：防同名覆盖与路径穿越
    stored_name = safe_stored_name(file.filename)
    file_path = UPLOAD_DIR / stored_name
    with open(file_path, "wb") as f:
        f.write(buffer)

    try:
        # 文档解析+向量化是CPU/IO密集同步操作，移入线程池
        chunk_count = await asyncio.to_thread(
            rag_engine.add_document, str(file_path), file.filename
        )
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": chunk_count,
            "total_chunks": rag_engine.get_document_count()
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@app.get("/api/documents/stats")
async def document_stats():
    """获取知识库统计信息"""
    return {
        "total_chunks": rag_engine.get_document_count(),
        "enabled": RAG_ENABLED
    }


@app.delete("/api/documents")
async def clear_documents():
    """清空知识库"""
    rag_engine.clear_all()
    return {"status": "success"}


# ====================== 健康检查 ======================
@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "rag_enabled": RAG_ENABLED,
        "document_chunks": rag_engine.get_document_count(),
        # 阶段 3.3-C：当日 LLM token 用量（可观测；超日预算在日志告警，不拦截请求）
        "llm_usage_today": usage_tracker.snapshot()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
