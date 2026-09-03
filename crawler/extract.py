"""HTML → Markdown 清洗（阶段 1.2）

官方站点（USCIS/Drupal、DHS）页面充满导航、页脚、脚本样板，
这里只保留正文结构（标题/段落/列表），输出干净可检索的 Markdown。
刻意不引入 markdownify 等重依赖：官方页面结构简单，轻量转换足够。
"""
import re

from bs4 import BeautifulSoup, NavigableString, Tag

# 导航样板与非内容标签，直接丢弃
_DROP_TAGS = {
    "script", "style", "noscript", "svg", "iframe", "template",
    "nav", "footer", "header", "aside", "form", "button", "select",
}

_BLOCK_TAGS = {"p", "div", "section", "article", "table", "tr", "blockquote", "figure"}


def html_to_markdown(html: str) -> str:
    """把整页 HTML 转为干净 Markdown：剥样板 → 定位正文 → 结构化渲染"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    # 正文容器优先级：main > article > body
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = _render(root)
    # 折叠多余空行，行内空白归一
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def extract_title(html: str) -> str:
    """提取页面标题（h1 优先，回退 <title>）"""
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        return _inline(h1)
    if soup.title:
        return _inline(soup.title)
    return ""


def _render(node: Tag) -> str:
    """递归渲染标签为 Markdown"""
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text + " ")
        elif isinstance(child, Tag):
            name = child.name
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                parts.append("\n" + "#" * int(name[1]) + " " + _inline(child) + "\n")
            elif name == "li":
                parts.append("\n- " + _inline(child) + "\n")
            elif name == "br":
                parts.append("\n")
            elif name in _BLOCK_TAGS:
                parts.append("\n" + _render(child) + "\n")
            else:
                # 行内标签（a/strong/em/span...）只取文本，避免 Markdown 链接噪声
                parts.append(_render(child))
    return "".join(parts)


def _inline(node: Tag) -> str:
    """取标签的行内纯文本（折叠空白）"""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True))
