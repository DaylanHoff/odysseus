# src/charcard.py
"""Character Card (V2 spec) import — SillyTavern / CharHub format.

Parses character cards from:
  * `.json` — the raw V2 (or V1) character-card document.
  * `.png`  — a PNG whose `tEXt`/`iTXt` chunk with keyword `chara` (or `ccv3`)
    holds the card JSON as base64-encoded UTF-8 (the standard V2 embedding).

No third-party image dependency: PNG chunks are walked with the stdlib
(`struct`/`zlib`). Only the text chunks we care about are decoded.
"""
import base64
import json
import logging
import struct
import zlib
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# PNG signature.
_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# Chunk keywords that carry a character card (base64 JSON).
_CHARA_KEYWORDS = (b"chara", b"ccv3")

# V2 spec identifier.
_V2_SPEC = "chara_card_v2"


def _decode_text_chunk(body: bytes) -> Optional[Tuple[str, bytes]]:
    """Decode a PNG tEXt/zTXt/iTXt chunk body into (keyword, value-bytes).

    Returns None if the chunk isn't one we can read. iTXt compression is
    handled (zlib). For tEXt/zTXt the value is Latin-1; for iTXt it's UTF-8.
    """
    if not body:
        return None
    nul = body.find(b"\x00")
    if nul < 0:
        return None
    keyword = body[:nul]
    rest = body[nul + 1:]
    # tEXt: keyword\0 text (Latin-1)
    # We don't strictly need to distinguish here — the chara payload is
    # base64 ASCII, so Latin-1 == ASCII bytes. Decode the keyword for return.
    try:
        kw_str = keyword.decode("latin-1")
    except Exception:
        return None
    return kw_str, rest


def _decode_itxt(body: bytes) -> Optional[Tuple[str, bytes]]:
    """Decode an iTXt chunk body into (keyword, text-bytes).

    iTXt layout: keyword\0 compression_flag(1) compression_method(1)
    language_tag\0 translated_keyword\0 text. Text is UTF-8; compressed
    text is zlib-decompressed.
    """
    if not body:
        return None
    nul = body.find(b"\x00")
    if nul < 0:
        return None
    keyword = body[:nul]
    rest = body[nul + 1:]
    if len(rest) < 2:
        return None
    comp_flag = rest[0]
    # comp_method = rest[1]  # always 0 (zlib) per PNG spec
    rest = rest[2:]
    # language tag
    nul2 = rest.find(b"\x00")
    if nul2 < 0:
        return None
    rest = rest[nul2 + 1:]
    # translated keyword
    nul3 = rest.find(b"\x00")
    if nul3 < 0:
        return None
    text = rest[nul3 + 1:]
    try:
        kw_str = keyword.decode("latin-1")
    except Exception:
        return None
    if comp_flag:
        try:
            text = zlib.decompress(text)
        except Exception:
            return None
    return kw_str, text


def _iter_png_text_chunks(data: bytes):
    """Yield (keyword_str, value_bytes) for every tEXt/iTXt/zTXt chunk.

    zTXt (compressed Latin-1) is also supported for completeness.
    """
    if not data.startswith(_PNG_SIG):
        raise ValueError("Not a PNG file (bad signature)")
    off = len(_PNG_SIG)
    n = len(data)
    while off + 8 <= n:
        length = struct.unpack(">I", data[off:off + 4])[0]
        ctype = data[off + 4:off + 8]
        body = data[off + 8:off + 8 + length]
        off += 8 + length + 4  # skip 4-byte CRC
        if ctype == b"tEXt":
            decoded = _decode_text_chunk(body)
        elif ctype == b"zTXt":
            decoded = _decode_text_chunk(body)
            if decoded:
                kw, val = decoded
                # zTXt: first byte after keyword-nul is compression method,
                # then zlib-compressed Latin-1 text.
                if len(val) >= 1:
                    try:
                        val = zlib.decompress(val[1:])
                    except Exception:
                        pass
                decoded = (kw, val)
        elif ctype == b"iTXt":
            decoded = _decode_itxt(body)
        else:
            decoded = None
        if decoded:
            yield decoded
        if ctype == b"IEND":
            break


def extract_charcard_from_png(data: bytes) -> Optional[Dict[str, Any]]:
    """Extract the character-card JSON from a PNG's `chara` text chunk.

    Returns the parsed dict, or None if no usable chunk was found.
    """
    for keyword, value in _iter_png_text_chunks(data):
        kw_bytes = keyword.encode("latin-1", errors="ignore")
        if kw_bytes not in _CHARA_KEYWORDS:
            continue
        if not value:
            continue
        # The value is base64 of UTF-8 JSON (standard V2 embedding).
        text = value.decode("latin-1", errors="ignore").strip()
        try:
            raw_json = base64.b64decode(text)
        except Exception:
            logger.debug("chara chunk %r: base64 decode failed", keyword)
            continue
        try:
            card = json.loads(raw_json.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("chara chunk %r: JSON parse failed: %s", keyword, e)
            continue
        if isinstance(card, dict):
            return card
    return None


def _str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _str_list(v: Any) -> list:
    if isinstance(v, list):
        return [_str(x) for x in v if _str(x)]
    if _str(v):
        return [_str(v)]
    return []


def parse_charcard_json(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate + normalize a raw character-card dict into Odysseus shape.

    Accepts both V2 (``spec == chara_card_v2`` with a ``data`` object) and
    V1 (flat fields). Returns None if the input isn't a recognizable card.
    """
    if not isinstance(raw, dict):
        return None

    spec = _str(raw.get("spec"))
    data_obj = raw.get("data")
    if spec == _V2_SPEC and isinstance(data_obj, dict):
        src = data_obj
    elif spec and spec != _V2_SPEC:
        # Unknown spec version — try the V2 path if a data object exists,
        # otherwise fall back to treating the top level as the source.
        src = data_obj if isinstance(data_obj, dict) else raw
    else:
        # V1 card: flat fields at top level.
        src = raw

    name = _str(src.get("name")) or _str(raw.get("name"))
    if not name:
        return None

    normalized = {
        "spec": spec or _V2_SPEC,
        "spec_version": _str(raw.get("spec_version")) or _str(src.get("spec_version")) or "2.0",
        "name": name,
        "description": _str(src.get("description")),
        "personality": _str(src.get("personality")),
        "first_mes": _str(src.get("first_mes")),
        "avatar": _str(src.get("avatar")),
        "mes_example": _str(src.get("mes_example")),
        "scenario": _str(src.get("scenario")),
        "creator_notes": _str(src.get("creator_notes")),
        "system_prompt": _str(src.get("system_prompt")),
        "post_history_instructions": _str(src.get("post_history_instructions")),
        "tags": _str_list(src.get("tags")),
        "creator": _str(src.get("creator")),
        "character_version": _str(src.get("character_version")),
        "alternate_greetings": _str_list(src.get("alternate_greetings")),
        # character_book is large and not currently surfaced; keep a flag only.
        "has_character_book": isinstance(src.get("character_book"), dict),
    }
    return normalized


def build_persona_from_charcard(card: Dict[str, Any]) -> Dict[str, Any]:
    """Compose Odysseus persona fields from a normalized charcard dict.

    Returns a dict with: name, system_prompt, avatar, first_mes,
    alternate_greetings, tags, source ('charcard_v2').
    """
    name = _str(card.get("name")) or "Imported Character"

    # Prefer the card's explicit system_prompt; otherwise synthesize one
    # from the descriptive fields. This keeps the persona effective even
    # when the card author left system_prompt blank (common).
    system_prompt = _str(card.get("system_prompt")).strip()
    if not system_prompt:
        parts = []
        desc = _str(card.get("description")).strip()
        if desc:
            parts.append(desc)
        personality = _str(card.get("personality")).strip()
        if personality:
            parts.append(f"Personality: {personality}")
        scenario = _str(card.get("scenario")).strip()
        if scenario:
            parts.append(f"Scenario: {scenario}")
        if parts:
            system_prompt = "\n\n".join(parts)
        else:
            system_prompt = f"You are {name}. Respond in character."

    first_mes = _str(card.get("first_mes")).strip()
    alternate_greetings = [g for g in card.get("alternate_greetings", []) if _str(g)]

    return {
        "name": name,
        "system_prompt": system_prompt,
        "avatar": _str(card.get("avatar")),
        "first_mes": first_mes,
        "alternate_greetings": alternate_greetings,
        "tags": card.get("tags", []),
        "creator": _str(card.get("creator")),
        "character_version": _str(card.get("character_version")),
        "source": "charcard_v2",
    }


def parse_card_bytes(data: bytes, filename: str = "") -> Optional[Dict[str, Any]]:
    """Parse a character card from raw file bytes (PNG or JSON).

    Dispatches on the PNG signature first, then tries JSON. Returns the
    normalized card dict (from ``parse_charcard_json``) or None.
    """
    raw_card: Optional[Dict[str, Any]] = None
    if data.startswith(_PNG_SIG):
        raw_card = extract_charcard_from_png(data)
        if raw_card is None:
            logger.warning("PNG %r had no readable chara text chunk", filename)
            return None
    else:
        try:
            raw_card = json.loads(data.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Could not parse %r as JSON: %s", filename, e)
            return None
    return parse_charcard_json(raw_card)
