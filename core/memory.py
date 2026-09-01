def keep_recent_messages(msg_list, max_pairs: int = 10):
    """
    工业级LLM上下文窗口管理
    保留系统提示词 + 最近N轮对话，防止上下文溢出
    硅谷GenAI项目标准优化方案
    """
    if not msg_list:
        return []

    # 保留系统prompt
    system_msg = msg_list[0] if msg_list[0]["role"] == "system" else None
    chat_history = msg_list[1:] if system_msg else msg_list

    # 每一轮对话=user+assistant两条
    if len(chat_history) > max_pairs * 2:
        chat_history = chat_history[-max_pairs * 2:]

    if system_msg:
        return [system_msg] + chat_history
    return chat_history


def inject_rag_prompt(msg_list, rag_prompt: str):
    """
    将RAG检索上下文注入系统提示词
    返回新列表，system消息会重建为新dict，
    避免原地修改会话存储中的原始system消息（防止多轮对话后prompt无限增长）
    """
    result = []
    for msg in msg_list:
        if msg["role"] == "system":
            result.append({
                "role": "system",
                "content": msg["content"] + "\n\n" + rag_prompt,
            })
        else:
            result.append(msg)
    return result