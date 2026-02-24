"""Render pydantic field titles ahead of field descriptions."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from docutils import nodes
from sphinx import addnodes


def _resolve_class(class_path: str) -> type[Any] | None:
    parts = class_path.split(".")
    for idx in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:idx])
        attr_parts = parts[idx:]
        try:
            obj: Any = import_module(module_name)
        except Exception:
            continue
        try:
            for attr in attr_parts:
                obj = getattr(obj, attr)
            if isinstance(obj, type):
                return obj
        except Exception:
            continue
    return None


def _get_field_title(model_cls: type[Any], field_name: str) -> str | None:
    model_fields = getattr(model_cls, "model_fields", None)
    if not isinstance(model_fields, dict):
        return None
    field_info = model_fields.get(field_name)
    if field_info is None:
        return None
    title = getattr(field_info, "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def add_field_title_before_description(
    app: Any, domain: str, objtype: str, contentnode: addnodes.desc_content
) -> None:
    """Prepend field title before the field description text."""
    if domain != "py" or objtype != "pydantic_field":
        return

    desc_node = contentnode.parent
    if not isinstance(desc_node, addnodes.desc):
        return

    object_id: str | None = None
    for child in desc_node:
        if isinstance(child, addnodes.desc_signature):
            ids = child.get("ids", [])
            if ids:
                object_id = ids[0]
                break

    if not object_id or "." not in object_id:
        return

    class_path, field_name = object_id.rsplit(".", 1)
    model_cls = _resolve_class(class_path)
    if model_cls is None:
        return

    title = _get_field_title(model_cls, field_name)
    if not title:
        return

    first_paragraph = next(
        (child for child in contentnode.children if isinstance(child, nodes.paragraph)),
        None,
    )
    if first_paragraph is None:
        paragraph = nodes.paragraph()
        paragraph += nodes.strong(text=f"{title}:")
        contentnode.insert(0, paragraph)
        return

    text = first_paragraph.astext()
    if text.startswith(f"{title}:"):
        return

    first_paragraph.insert(0, nodes.strong(text=f"{title}: "))


def setup(app: Any) -> dict[str, Any]:
    app.connect("object-description-transform", add_field_title_before_description)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
