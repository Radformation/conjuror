"""Filters for object entries shown in local page TOC."""

from __future__ import annotations

from typing import Any

from sphinx import addnodes


def hide_pydantic_fields_from_toc(
    app: Any, domain: str, objtype: str, contentnode: addnodes.desc_content
) -> None:
    """Keep class entries in TOC while hiding pydantic field entries."""
    if domain != "py" or objtype != "pydantic_field":
        return

    desc_node = contentnode.parent
    if isinstance(desc_node, addnodes.desc):
        desc_node["no-contents-entry"] = True


def setup(app: Any) -> dict[str, Any]:
    app.connect("object-description-transform", hide_pydantic_fields_from_toc)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
