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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from core.model import init_llm_model
from core.memory import keep_recent_messages, inject_rag_prompt
from core.session import session_manager
from core.rag import rag_engine, UPLOAD_DIR
from core.uploads import validate_upload, safe_stored_name, MAX_UPLOAD_BYTES
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

# 全局配置
MAX_HISTORY_PAIRS = 15
RAG_ENABLED = True
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


# ====================== 页面路由 ======================
@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ====================== 会话管理 API ======================
@app.post("/api/sessions")
async def create_session(data: CreateSessionInput):
    """创建新会话"""
    session_id = session_manager.create_session(data.title)
    logger.info(f"New session created: {session_id}")
    return {"session_id": session_id, "title": data.title}


@app.get("/api/sessions")
async def list_sessions():
    """获取所有会话列表（左侧边栏）"""
    sessions = session_manager.list_sessions()
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    session_manager.delete_session(session_id)
    logger.info(f"Session deleted: {session_id}")
    return {"status": "success"}


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """重置指定会话对话"""
    session_manager.reset_session(session_id)
    return {"status": "success"}


# ====================== 核心对话 API ======================
@app.post("/api/chat-stream")
async def chat_stream(data: ChatInput):
    """SSE流式对话接口（支持RAG增强）"""
    # 获取或创建会话
    session_id = data.session_id
    if not session_id or not session_manager.session_exists(session_id):
        session_id = session_manager.create_session()

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
                docs = await asyncio.to_thread(rag_engine.retrieve, data.message, 3)
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
            try:
                for chunk in model.stream(optimized_msgs):
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
        "document_chunks": rag_engine.get_document_count()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
