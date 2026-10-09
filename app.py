"""واجهة Streamlit العربية للنموذج الأولي للمساعد الشرعي الذكي."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from html import escape
from typing import Any
from urllib.parse import urlparse

import streamlit as st

from rag_agent import RAGAgent


logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_rag_agent() -> RAGAgent:
    """إنشاء الوكيل مرة واحدة فقط لتفادي إعادة تحميل الفهارس."""
    return RAGAgent()


def has_value(value: Any) -> bool:
    """يتحقق من أن القيمة قابلة للعرض وليست فارغة أو None."""
    return value is not None and str(value).strip() != ""


def display_value(value: Any) -> str:
    """يحوّل القيمة إلى نص مناسب للواجهة مع تنسيق درجة الصلة."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).strip()


def get_first_value(source: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """يعيد أول حقل متاح من أسماء الحقول البديلة."""
    for key in keys:
        value = source.get(key)
        if has_value(value):
            return value
    return None


def is_safe_link(value: Any) -> bool:
    """يسمح فقط بروابط HTTP(S) الكاملة قبل عرضها للمستخدم."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def render_source(source: Mapping[str, Any]) -> None:
    """يعرض بطاقة مصدر بالحقول المتاحة فقط."""
    fields = (
        ("نوع المصدر", ("source_type", "type")),
        ("اسم المصدر", ("source", "source_name", "title")),
        ("اسم الكتاب", ("book", "book_name")),
        ("اسم العالم", ("scholar", "author", "mufti")),
        ("السورة", ("surah", "sura")),
        ("رقم الآية", ("ayah", "ayah_range", "ayah_start", "verse")),
        ("درجة الصلة", ("score", "relevance_score")),
    )
    link = get_first_value(source, ("url", "link", "href"))
    details = [
        (label, display_value(value))
        for label, keys in fields
        if has_value(value := get_first_value(source, keys))
    ]

    with st.container(border=True):
        if details:
            for label, value in details:
                st.markdown(f"**{escape(label)}:** {escape(value)}")
        else:
            st.caption("لا تتوفر بيانات وصفية إضافية لهذا المصدر.")

        if is_safe_link(link):
            st.link_button("فتح المصدر", str(link).strip(), use_container_width=False)


def render_result(question: str, result: Mapping[str, Any]) -> None:
    """يعرض الإجابة وحالة التحقق والمصادر المسترجعة بأمان."""
    answer = result.get("answer")
    sources = result.get("sources")

    st.subheader("نتيجة البحث")
    with st.container(border=True):
        st.markdown("**السؤال المدخل**")
        st.write(question)

    st.markdown("### الإجابة")
    with st.container(border=True):
        st.write(display_value(answer) if has_value(answer) else "لم تتوفر إجابة نصية لهذه النتيجة.")

    if "verified" in result and result.get("verified") is not None:
        if result["verified"] is True:
            st.success("حالة التحقق: تم التحقق من دعم الإجابة بالمصادر المتاحة.")
        elif result["verified"] is False:
            st.warning("هذه النتيجة تحتاج إلى مراجعة إضافية قبل الاعتماد عليها.")
        else:
            st.info(f"حالة التحقق: {display_value(result['verified'])}")

    st.markdown("### الأدلة والمراجع")
    if not isinstance(sources, (list, tuple)) or not sources:
        st.info("لم نعثر على مصادر كافية لعرضها لهذه النتيجة.")
        return

    valid_sources = [source for source in sources if isinstance(source, Mapping)]
    if not valid_sources:
        st.info("لم نعثر على مصادر كافية لعرضها لهذه النتيجة.")
        return

    for source in valid_sources:
        render_source(source)


st.set_page_config(
    page_title="المساعد الشرعي الذكي",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تنسيق هادئ يدعم العربية واتجاه RTL في كامل الصفحة.
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stAppViewContainer"] {
            background: #f7faf9;
            color: #123047;
        }
        .block-container { max-width: 980px; padding-top: 2.5rem; padding-bottom: 3rem; }
        h1, h2, h3, p, label, .stMarkdown { direction: rtl; text-align: right; }
        .hero-title { color: #103b5c; margin-bottom: 0.15rem; }
        .hero-description { color: #406174; font-size: 1.08rem; margin-bottom: 0.65rem; }
        .prototype-badge {
            display: inline-block; background: #dff1e8; color: #216245; border-radius: 999px;
            padding: 0.28rem 0.75rem; font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem;
        }
        .fixed-notice {
            background: #e8f2f8; border-right: 4px solid #0f4c6e; border-radius: 10px;
            color: #163c55; padding: 0.85rem 1rem; margin: 1rem 0 1.5rem;
        }
        [data-testid="stTextArea"] textarea {
            direction: rtl; text-align: right; border-radius: 10px; border-color: #a9c3d1;
        }
        [data-testid="stButton"] button {
            background: #0f4c6e; color: #ffffff; border: 0; border-radius: 9px;
            font-weight: 700; min-height: 2.8rem;
        }
        [data-testid="stButton"] button:hover { background: #0b3c58; color: #ffffff; }
        [data-testid="stSidebar"] { background: #edf5f1; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff; border-color: #d6e4e9; border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("حول المنصة")
    st.write("مسار النظام: تحليل السؤال ← البحث في المصادر ← ترتيب النتائج ← توليد إجابة موثقة")
    st.markdown("**المصادر المتاحة**")
    st.markdown("- القرآن الكريم\n- التفسير\n- الأحاديث\n- الفتاوى")
    st.divider()
    st.caption("فريق 7 – صيف طيبة 2026")

st.markdown('<span class="prototype-badge">نموذج أولي</span>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">المساعد الشرعي الذكي</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-description">منصة تساعدك على الوصول إلى معلومات شرعية مدعومة بالأدلة والمراجع من مصادر موثوقة.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="fixed-notice">هذه المنصة أداة مساعدة للبحث والتحقق، ولا تُعد بديلًا عن العالم أو المختص الشرعي.</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "اكتب سؤالك الشرعي",
    placeholder="مثال: ما حكم القرض بفائدة؟",
    height=150,
)

if st.button("البحث في المصادر", type="primary", use_container_width=True):
    clean_question = question.strip()
    if not clean_question:
        st.warning("يرجى كتابة سؤال شرعي قبل بدء البحث.")
    else:
        try:
            with st.spinner("يجري البحث في المصادر وترتيب النتائج..."):
                agent = get_rag_agent()
                response = agent.run_query(clean_question)

            if not isinstance(response, Mapping):
                raise TypeError("RAGAgent.run_query returned an unexpected response type")
            render_result(clean_question, response)
        except Exception as exc:
            # يسجل التفصيل التقني في السجل، مع إبقاء رسالة الواجهة آمنة ومختصرة.
            logger.exception("RAG search failed with %s", type(exc).__name__)
            st.error("تعذر إتمام البحث الآن. يرجى المحاولة لاحقًا أو التحقق من جاهزية بيانات المشروع.")
