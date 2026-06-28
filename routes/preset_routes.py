"""Preset routes — /api/presets GET, /api/presets/custom POST, user templates CRUD."""

import logging
import uuid
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from src.request_models import PresetUpdateRequest
from core.middleware import require_admin
from src.auth_helpers import effective_user

logger = logging.getLogger(__name__)


class UserTemplateRequest(BaseModel):
    id: str = ""
    name: str = Field(..., min_length=1, max_length=100)
    system_prompt: str = Field("", max_length=10000)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(0, ge=0, le=65536)
    # Character-card import fields. Optional — only set when a persona was
    # imported from a V2 character card (or hand-populated). They round-trip
    # through GET /api/presets/templates so the frontend can show the avatar
    # and seed a first-message greeting.
    avatar: str = Field("", max_length=2000)
    first_mes: str = Field("", max_length=10000)
    alternate_greetings: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    source: str = Field("", max_length=64)


def setup_preset_routes(preset_manager) -> APIRouter:
    router = APIRouter(tags=["presets"])

    @router.get("/api/presets")
    async def get_presets() -> Dict[str, Any]:
        return preset_manager.presets

    @router.post("/api/presets/custom")
    async def update_custom_preset(preset_update: PresetUpdateRequest, _admin: None = Depends(require_admin)) -> Dict[str, Any]:
        try:
            # Preserve an already-saved avatar/first_mes when the request
            # doesn't carry one (the Inject-tab save path doesn't send them,
            # and we don't want to wipe a persona's avatar just because the
            # user retuned temperature). The frontend explicitly sends an
            # empty avatar only on persona deactivation.
            existing = preset_manager.presets.get("custom") or {}
            avatar = preset_update.avatar or existing.get("avatar", "")
            first_mes = preset_update.first_mes or existing.get("first_mes", "")
            success = preset_manager.update_custom(
                preset_update.temperature,
                preset_update.max_tokens,
                preset_update.system_prompt,
                preset_update.name,
                preset_update.enabled,
                preset_update.inject_prefix,
                preset_update.inject_suffix,
                avatar=avatar,
                first_mes=first_mes,
            )
            if success:
                return {"success": True, "message": "Custom preset updated"}
            return {"success": False, "message": "Failed to save preset"}
        except Exception as e:
            logger.error(f"Preset update error: {e}")
            raise HTTPException(500, "Failed to update custom preset")

    @router.get("/api/presets/templates")
    async def get_user_templates() -> List[Dict]:
        return preset_manager.get_user_templates()

    @router.post("/api/presets/templates")
    async def save_user_template(req: UserTemplateRequest, _admin: None = Depends(require_admin)) -> Dict[str, Any]:
        template = req.model_dump()
        if not template["id"]:
            template["id"] = f"user-{uuid.uuid4().hex[:8]}"
        success = preset_manager.save_user_template(template)
        if success:
            return {"success": True, "template": template}
        return {"success": False, "message": "Failed to save template"}

    @router.delete("/api/presets/templates/{template_id}")
    async def delete_user_template(template_id: str, _admin: None = Depends(require_admin)) -> Dict[str, Any]:
        success = preset_manager.delete_user_template(template_id)
        if success:
            return {"success": True}
        return {"success": False, "message": "Failed to delete template"}

    @router.post("/api/presets/import-card")
    async def import_character_card(request: Request, _admin: None = Depends(require_admin)) -> Dict[str, Any]:
        """Import a Character Card (V2 spec) from an uploaded .png or .json.

        Accepts multipart/form-data (a single ``file`` field) or a raw JSON
        body (the card document itself). Parses via ``src.charcard`` and
        saves the derived persona as a user template.
        """
        from src.charcard import parse_card_bytes, build_persona_from_charcard
        import uuid as _uuid

        data_bytes: bytes = b""
        filename = ""
        content_type = (request.headers.get("content-type") or "").lower()

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(400, "No 'file' field in upload")
            # Starlette UploadFile — read async.
            data_bytes = await upload.read()
            filename = getattr(upload, "filename", "") or ""
        elif content_type.startswith("application/json"):
            try:
                raw = await request.body()
            except Exception:
                raw = b""
            # Could be a raw card JSON, or a wrapper with a base64 file.
            try:
                payload = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            if isinstance(payload, dict) and isinstance(payload.get("file"), str):
                # base64-encoded file payload (data-uri or raw b64).
                b64 = payload["file"]
                if "," in b64 and b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                try:
                    import base64 as _b64
                    data_bytes = _b64.b64decode(b64)
                except Exception:
                    raise HTTPException(400, "Invalid base64 file payload")
                filename = str(payload.get("filename") or "")
            else:
                # Treat the body itself as the card JSON document.
                data_bytes = raw
                filename = "card.json"
        else:
            raise HTTPException(415, f"Unsupported content type: {content_type}")

        if not data_bytes:
            raise HTTPException(400, "Empty upload")

        card = parse_card_bytes(data_bytes, filename)
        if not card:
            raise HTTPException(400, "Could not parse a Character Card V2 from this file")

        persona = build_persona_from_charcard(card)
        template = {
            "id": f"user-{_uuid.uuid4().hex[:8]}",
            "name": persona["name"],
            "system_prompt": persona["system_prompt"],
            "temperature": 0.8,
            "max_tokens": 0,
            "avatar": persona.get("avatar", ""),
            "first_mes": persona.get("first_mes", ""),
            "alternate_greetings": persona.get("alternate_greetings", []),
            "tags": persona.get("tags", []),
            "source": persona.get("source", "charcard_v2"),
        }
        success = preset_manager.save_user_template(template)
        if not success:
            raise HTTPException(500, "Failed to save imported persona")
        return {"success": True, "template": template}

    @router.post("/api/presets/expand")
    async def expand_character_prompt(request: Request) -> Dict[str, Any]:
        """Use AI to expand a rough character description into a full system prompt."""
        from src.ai_interaction import _resolve_model
        from src.llm_core import llm_call_async

        data = await request.json()
        draft = (data.get("prompt") or "").strip()
        name = (data.get("name") or "").strip()

        if not draft and not name:
            return {"success": False, "message": "Nothing to expand"}

        user_input = ""
        if name:
            user_input += f"Character name: {name}\n"
        if draft:
            user_input += f"Notes: {draft}\n"

        messages = [
            {"role": "system", "content": (
                "You are an expert at writing character system prompts for AI assistants. "
                "The user will give you a character name and/or rough notes. "
                "Write a concise, effective system prompt (3-6 sentences) that captures the character's personality, "
                "speaking style, knowledge areas, and behavioral guidelines. "
                "Output ONLY the system prompt text — no quotes, no preamble, no explanation."
            )},
            {"role": "user", "content": user_input},
        ]

        try:
            model_spec = data.get("model") or ""
            user = effective_user(request)
            url, model, headers = _resolve_model(model_spec, owner=user)
            result = await llm_call_async(url, model, messages, temperature=0.8, max_tokens=500, headers=headers)
            return {"success": True, "prompt": result.strip()}
        except Exception as e:
            logger.error(f"Expand prompt failed: {e}")
            return {"success": False, "message": str(e)}

    # ── Group presets ──
    @router.get("/api/presets/groups")
    async def get_group_presets():
        """Get saved group chat presets."""
        return {"groups": preset_manager.get_group_presets()}

    @router.post("/api/presets/groups")
    async def save_group_presets(request: Request, _admin: None = Depends(require_admin)):
        """Save group chat presets."""
        data = await request.json()
        preset_manager.save_group_presets(data.get("groups", []))
        return {"ok": True}

    return router
