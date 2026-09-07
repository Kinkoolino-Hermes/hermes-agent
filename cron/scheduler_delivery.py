"""Cron delivery targets, transports, and receipt-backed delivery outcomes.

This is the defining delivery module for scheduler receipt behavior.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import load_config
from cron.executions import (
    observe_transport_unknown,
    preregister_receipt_plan,
    receipt_summary,
    record_transport_receipt,
)

logger = logging.getLogger("cron.scheduler")

_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal", "matrix", "mattermost",
    "homeassistant", "dingtalk", "feishu", "wecom", "wecom_callback", "weixin", "sms",
    "email", "webhook", "bluebubbles", "qqbot", "yuanbao",
})
_HOME_TARGET_ENV_VARS = {
    "matrix": "MATRIX_HOME_ROOM", "telegram": "TELEGRAM_HOME_CHANNEL",
    "discord": "DISCORD_HOME_CHANNEL", "slack": "SLACK_HOME_CHANNEL",
    "signal": "SIGNAL_HOME_CHANNEL", "mattermost": "MATTERMOST_HOME_CHANNEL",
    "sms": "SMS_HOME_CHANNEL", "email": "EMAIL_HOME_ADDRESS",
    "dingtalk": "DINGTALK_HOME_CHANNEL", "feishu": "FEISHU_HOME_CHANNEL",
    "wecom": "WECOM_HOME_CHANNEL", "weixin": "WEIXIN_HOME_CHANNEL",
    "bluebubbles": "BLUEBUBBLES_HOME_CHANNEL", "qqbot": "QQBOT_HOME_CHANNEL",
    "whatsapp": "WHATSAPP_HOME_CHANNEL", "whatsapp_cloud": "WHATSAPP_CLOUD_HOME_CHANNEL",
}
_LEGACY_HOME_TARGET_ENV_VARS = {"QQBOT_HOME_CHANNEL": "QQ_HOME_CHANNEL"}

def _resolve_cron_surface_mode(pconfig, logical_platform_name: str) -> str:
    """Resolve the continuable-cron delivery surface for a platform config.

    Returns ``"in_channel"`` or ``"thread"`` (default). Two config shapes:

    - Native adapter: the flat key ``platforms.<p>.extra.cron_continuable_surface``
      (shipped shape, unchanged).
    - Relay-fronted: ``platforms.relay.extra.<logical>.cron_continuable_surface``
      — the same per-logical-platform sub-block the relay's documented Slack
      knobs use (``reply_in_thread``, ``dm_top_level_threads_as_sessions``;
      see RelayAdapter._relay_slack_extra). The sub-block wins over a flat
      key when both exist, matching _relay_slack_extra precedence, and is
      scoped to its logical platform so a ``slack:`` block cannot leak onto
      another fronted platform.

    Precedence nuance vs _relay_slack_extra: that helper is all-or-nothing
    (a sub-dict REPLACES the flat extra entirely), while this one falls back
    to the flat key when the sub-block exists but omits the knob. The
    difference is deliberate — the flat key is the legacy staging shape and
    must keep working — but note a flat ``cron_continuable_surface`` then
    applies to EVERY platform this relay fronts; only the per-platform D6
    capability gate contains it. Scope the knob under the sub-block on
    multi-platform relays.

    Field gap (2026-08-18): the scheduler read only the flat key, so on the
    relay lane — where pconfig is platforms.relay — operators had NO working
    location for the knob and briefs always threaded.
    """
    try:
        extra = getattr(pconfig, "extra", None) or {}
        sub = extra.get(str(logical_platform_name or "").lower())
        if isinstance(sub, dict) and sub.get("cron_continuable_surface") is not None:
            raw = sub.get("cron_continuable_surface")
        else:
            raw = extra.get("cron_continuable_surface")
        if raw is not None and str(raw).strip().lower() == "in_channel":
            return "in_channel"
    except Exception:
        pass
    return "thread"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata.

    Treats non-dict origins (free-form provenance strings, ints, lists from
    migration scripts or hand-edited jobs.json) as missing instead of
    crashing with ``AttributeError`` on ``origin.get(...)``. Without this
    guard, a job tagged with e.g. ``"combined-digest-replaces-x-and-y"``
    crashed every fire attempt with
    ``'str' object has no attribute 'get'`` — ``mark_job_run`` recorded the
    failure, but the next tick re-loaded the same poisoned origin and
    crashed identically until the field was patched manually (#18722).
    """
    origin = job.get("origin")
    if type(origin) is not dict:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform is None or chat_id is None:
        return None
    if type(platform) is not str or type(chat_id) not in {str, int}:
        raise ValueError("origin target identity is invalid")
    thread_id = origin.get("thread_id")
    if thread_id is not None and type(thread_id) not in {str, int}:
        raise ValueError("origin target identity is invalid")
    if platform and chat_id not in {"", 0}:
        return origin
    return None


def _cron_mirror_delivery_enabled(job: dict, cfg: Optional[dict] = None) -> bool:
    """Whether a cron delivery should also be mirrored into the target chat's
    gateway session transcript.

    Default OFF — preserves the historical isolation guarantee (cron deliveries
    live only in the cron job's own session, never the target chat's history)
    byte-for-byte for everyone who does not opt in.

    CARVE-OUT: the ``in_channel`` continuable surface seeds its target
    session independently of this knob (see ``_deliver_result`` /
    ``_seed_cron_channel_session``). in_channel is itself opt-in
    (``cron_continuable_surface: in_channel`` + the adapter capability bit),
    and the seed IS the feature — a continuable flat brief without its seed
    is a brief the next reply can't see. This knob keeps governing the
    SEPARATE default/thread-surface transcript mirror only.

    Precedence (first decisive value wins):
      1. Per-job ``attach_to_session`` (bool) — set via the ``cronjob`` tool,
         lets one briefing job opt in without flipping global behaviour.
      2. Global ``cron.mirror_delivery`` (bool) in config.yaml.
      3. False.

    When enabled, the cron's final output is appended to the target session as
    an assistant turn via the existing ``gateway.mirror.mirror_to_session`` —
    the same primitive ``send_message`` uses — so the next user reply in that
    chat sees the brief in context (no "what is Task #2?" amnesia). This is
    alternation- and cache-safe: the append lands at a turn boundary between
    user turns, never mid-loop, and never mutates the cached system prompt.
    """
    per_job = job.get("attach_to_session")
    if isinstance(per_job, bool):
        return per_job
    try:
        if cfg is None:
            cfg = load_config() or {}
        return bool((cfg.get("cron", {}) or {}).get("mirror_delivery", False))
    except Exception:
        return False


def _target_matches_origin(origin: dict, platform_name: str, chat_id: str,
                           thread_id: Optional[str]) -> bool:
    """True when a delivery target is the job's own origin conversation.

    Mirroring is scoped to the origin session by design (see
    ``_maybe_mirror_cron_delivery``). A job created from a live gateway chat
    stamps that chat as ``origin`` (``cronjob_tools._origin_from_env``), and
    that session is guaranteed to exist — it is the very conversation the user
    was in when they scheduled the job. Fan-out targets (``deliver=all``,
    explicit ``platform:chat_id`` to some *other* chat, or a home-channel
    fallback for an origin-less API/script job) are deliberately NOT mirrored:
    they are broadcasts, not a continuation of a conversation, and may point at
    a chat the user never opened an agent session in.

    This makes the historical "cold-start" worry a non-case: when the mirror
    semantically applies (target == origin) the session always exists; when no
    session exists, the target was never the origin conversation, so we simply
    do not mirror.
    """
    if not origin:
        return False
    if str(origin.get("platform", "")).lower() != str(platform_name).lower():
        return False
    if str(origin.get("chat_id", "")) != str(chat_id):
        return False
    # thread_id must match when the origin pins one (topic-scoped chats); a
    # target that lost the thread_id is not the same conversation lane.
    origin_thread = origin.get("thread_id")
    if origin_thread is not None and str(origin_thread) != str(thread_id or ""):
        return False
    return True


# Resolution-provenance ranking for the dedup OR-merge in
# _resolve_delivery_targets: higher rank = stronger mirror claim. Broadcast
# expansions rank 0 so "origin,all"/"all,origin" hitting the same chat keeps
# the origin(-fallback) tag regardless of token order.
_MIRROR_PROVENANCE_RANK = {
    "origin": 3,
    "origin_fallback": 2,
    "explicit": 1,
}


def _target_mirror_eligible(
    job: dict,
    target: dict,
    *,
    global_mirror: bool,
    origin_match: Optional[bool] = None,
) -> bool:
    """Whether a resolved delivery target may receive the transcript mirror.

    The June origin-scoping refactor gated mirroring on target == origin,
    which correctly excluded broadcasts but also silenced two legitimate
    conversation shapes — both hit by script-provisioned ("managed") crons,
    which never capture an origin (``_origin_from_env`` only fires for jobs
    created from a live gateway chat):

    - ``origin_fallback``: ``deliver=origin`` with no captured origin resolves
      to the home channel — the user's primary conversation standing in for
      the origin, not a broadcast. Eligible under the same flags as a true
      origin target. (Field report 2026-08-17: brief delivered to the Slack
      DM, mirror silently skipped, reply hit a context-less session.)
    - ``explicit``: a ``platform:chat_id`` target is eligible ONLY when the
      job itself opts in via ``attach_to_session: true`` — the job author
      declaring this target a conversation (managed per-user DM briefings).
      The global ``cron.mirror_delivery`` flag never activates explicit
      targets: it must not start writing transcript entries into arbitrary
      explicitly-addressed chats (shared channels, other users' DMs).

    Broadcast expansions (``all``, bare-platform home targets) carry no
    provenance tag and are never eligible — unchanged invariant.

    ``origin_match`` lets the caller pass a precomputed
    ``_target_matches_origin`` result (``_deliver_result`` already computes it
    for the same target); when ``None`` it is computed here so tests and
    future callers stay self-contained.
    """
    if origin_match is None:
        origin = _resolve_origin(job) or {}
        origin_match = _target_matches_origin(
            origin, target.get("platform", ""), target.get("chat_id", ""),
            target.get("thread_id"),
        )
    if origin_match:
        return True
    resolved_from = target.get("_resolved_from")
    if resolved_from == "origin_fallback":
        # Same activation rules as an origin target: per-job attach wins,
        # else the global flag. This deliberately restates the precedence
        # _cron_mirror_delivery_enabled encodes (keep the two in sync): the
        # sole production caller pre-merges it into `global_mirror`, but the
        # helper must stay correct standalone — a per-job False must beat a
        # raw global True for any caller that does not pre-merge.
        per_job = job.get("attach_to_session")
        if isinstance(per_job, bool):
            return per_job
        return bool(global_mirror)
    if resolved_from == "explicit":
        return job.get("attach_to_session") is True
    return False


def _inchannel_seed_allowed(*, is_dm: bool, user_id: Optional[str]) -> bool:
    """Whether the flat in_channel session seed may run for a target.

    Group-channel session keys are user-isolated
    (``…:group:<chat_id>:<user_id>`` — see _seed_cron_channel_session); a
    seed without a real user_id would create an orphan session that no
    inbound reply ever resolves to, which is worse than no seed (the plain
    mirror can still land if a session exists). DM keys don't embed
    user_id, so DM targets are always seedable. Origin-captured jobs carry
    the scheduler's user_id; origin-less managed jobs typically don't, and
    their group-channel targets must fall back to the plain mirror.
    """
    return bool(is_dm or user_id)


def _maybe_mirror_cron_delivery(
    job: dict,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
) -> None:
    """Best-effort mirror of a cron delivery into the origin chat's session.

    No-op unless ``enabled`` (resolved once by the caller, and already scoped to
    the origin target — see ``_target_matches_origin``). Reuses the shipped
    ``mirror_to_session`` so cron rides exactly the same path that interactive
    ``send_message`` mirroring already uses, including passing ``user_id`` so a
    per-user-isolated group chat resolves to the exact member who scheduled the
    job (parity with ``send_message``). All failures are swallowed — a delivery
    that succeeded must never be reported as failed because the transcript
    mirror hit a problem.

    Because the caller only enables this for the target that equals the job's
    origin conversation, the session is expected to exist (the job was born in
    that session). A missing session therefore indicates an origin-less /
    fan-out delivery that should not have been mirrored anyway, and is treated
    as a silent no-op — never a synthetic session is created.
    """
    if not enabled:
        return
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.mirror import mirror_to_session

        # Mirror as a USER turn with a labelled prefix, NOT an assistant turn.
        # The brief is not the agent speaking; an assistant-role mirror lands as
        # assistant→assistant after the agent's last turn and breaks strict
        # alternation (issue #2221, the exact failure #2313 removed). A
        # user-role turn collapses safely via repair_message_sequence's
        # consecutive-user merge on every provider, and the prefix preserves the
        # "this came from cron" context that the dropped SQLite mirror metadata
        # would otherwise lose on replay.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=thread_id,
            user_id=user_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': mirrored delivery into %s:%s session transcript",
                job.get("id", "?"), platform_name, chat_id,
            )
        else:
            logger.debug(
                "Job '%s': delivery mirror skipped for %s:%s "
                "(no matching gateway session — cold start)",
                job.get("id", "?"), platform_name, chat_id,
            )
    except Exception as e:
        logger.debug(
            "Job '%s': delivery mirror failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )


def _open_continuable_cron_thread(
    job: dict,
    adapter,
    chat_id: str,
    loop,
) -> Optional[str]:
    """Open a dedicated thread for a continuable cron job (thread-preferred).

    Returns the new ``thread_id`` on success, or ``None`` when the platform has
    no thread primitive (WhatsApp/Signal/SMS) or creation failed — the ``None``
    return is the caller's signal to fall back to the origin-DM mirror, the same
    open-thread-or-fallback shape as ``GatewayRunner._process_handoff``. Reuses
    the shipped ``adapter.create_handoff_thread``; no new adapter surface.
    """
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    task_name = job.get("name") or job.get("id", "cron")
    thread_name = f"Hermes — {task_name}"
    try:
        from agent.async_utils import safe_schedule_threadsafe

        coro = create_thread(str(chat_id), thread_name)
        future = safe_schedule_threadsafe(coro, loop)  # type: ignore[arg-type]
        if future is None:
            return None
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as e:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back to "
            "DM-session mirror: %s",
            job.get("id", "?"), getattr(adapter, "name", "?"), e,
        )
        return None


def _seed_cron_thread_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    thread_id: str,
    mirror_text: str,
    chat_name: Optional[str] = None,
    is_dm: bool = False,
    scope_id: Optional[str] = None,
) -> None:
    """Seed the freshly-opened cron thread's session with the brief.

    Without this the brief is *visible* in the new thread but absent from any
    transcript, so the user's first reply in-thread would hit a session with no
    record of it ("what is Task #2?"). We create the thread-keyed session (the
    same key the user's reply will resolve to — ``build_session_key`` keys
    threads as participant-shared, so no ``user_id`` is needed) and append the
    brief as an assistant turn via the shipped ``mirror_to_session``.

    ``scope_id`` is the workspace/server scope (Slack team id).
    ``build_session_key`` embeds it in every Slack key, so a scoped reply's
    key carries it — the seed must reproduce it or the seeded row is
    unreachable (the scope-less flat-seed sibling of the is_dm keying bug).
    Best-effort None for platforms without scope.

    ``is_dm`` selects the seeded ``chat_type``: a thread under a DM must seed
    ``chat_type="dm"`` because the user's in-thread DM reply arrives with
    chat_type="dm" and ``build_session_key`` routes DM threads through the DM
    arm (``...:dm:<chat>:<thread>``) — a "thread"-typed seed lands in
    ``...:thread:<chat>:<thread>``, a row no DM reply ever resolves to
    (continuation amnesia, Alice live 2026-08-20, job 8e21a957b77b). Channel
    threads keep ``chat_type="thread"`` (their replies really do arrive as
    threads). Same sibling-lane class as the flat seed's ``is_dm``
    (dcca9d8cfe).

    Mirrors ``GatewayRunner._process_handoff``'s seed step, but standalone:
    cron reaches the live ``SessionStore`` through the adapter's
    ``_session_store`` handle rather than the gateway object. Best-effort — a
    delivery that already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        seeded_session_id: Optional[str] = None
        session_store = getattr(adapter, "_session_store", None)
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                # Discord thread destinations must key on the thread's OWN id
                # to match how the Discord adapter keys organic in-thread
                # messages (chat_id == thread_id). Other platforms (Slack,
                # Telegram) use chat_id == parent_channel for thread messages,
                # so the parent chat_id is correct for them. See the matching
                # guard in GatewayRunner._process_handoff.
                if platform_enum == Platform.DISCORD:
                    seed_chat_id = str(thread_id)
                else:
                    seed_chat_id = str(chat_id)
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=seed_chat_id,
                    chat_name=chat_name,
                    # DM threads key through the DM arm (see docstring); the
                    # reply's chat_type is what the seed must reproduce.
                    chat_type="dm" if is_dm else "thread",
                    user_id="system:cron",
                    user_name="Cron",
                    thread_id=str(thread_id),
                    scope_id=str(scope_id) if scope_id else None,
                )
                # Ensure the thread-keyed session row exists so the mirror has
                # a target and the user's later reply joins the same session.
                # Capture the exact id — the mirror writes into THIS row, not
                # an origin-heuristic rediscovery (which bails on populated
                # chats; same class as the flat-seed live failure 2026-08-19).
                _entry = session_store.get_or_create_session(dest_source)
                seeded_session_id = getattr(_entry, "session_id", None)

        from gateway.mirror import mirror_to_session

        # User-role + labelled prefix (see _maybe_mirror_cron_delivery): the
        # seeded brief must not read as an assistant turn, or the user's first
        # in-thread reply produces assistant→user→... off a phantom assistant
        # message. Pass the seed user_id so the mirror resolves the exact
        # thread-keyed session row we just created.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=str(thread_id),
            user_id="system:cron",
            role="user",
            session_id=seeded_session_id,
        )
        if ok:
            logger.info(
                "Job '%s': opened continuable thread %s on %s:%s and seeded the brief",
                job.get("id", "?"), thread_id, platform_name, chat_id,
            )
        else:
            logger.warning(
                "Job '%s': thread seed did NOT land on %s:%s thread=%s — an "
                "in-thread reply will not see this brief",
                job.get("id", "?"), platform_name, chat_id, thread_id,
            )
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-
        # amnesia bug (Alice 2026-08-19) — it must be visible in production.
        logger.warning(
            "Job '%s': seeding cron thread session failed for %s:%s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, thread_id, e,
        )


def _seed_cron_channel_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    *,
    is_dm: bool,
    user_id: Optional[str],
    chat_name: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> bool:
    """Seed the FLAT (thread_id=None) session for an ``in_channel`` cron delivery.

    The ``in_channel`` surface (D1/D2) delivers the brief flat into the channel
    with no thread, so the continuation surface is the whole-channel /
    whole-DM session keyed ``thread_id=None`` — the same bucket
    ``reply_in_thread: false`` routes an inbound plain reply to.

    Unlike the thread path, the shipped delivery-mirror alone is NOT sufficient
    here: ``mirror_to_session`` only APPENDS to a session that already EXISTS
    (``_find_session_id`` → no-op when none matches), and a flat channel
    ``(…, None)`` row is only created when a human posts a top-level message the
    bot processes — a ``chat_postMessage`` cron delivery never goes through the
    inbound handler, so the row is usually absent and the mirror silently drops
    the brief (verified live: the brief never landed, the reply had no context).
    So we CREATE the flat session row first, exactly like
    ``_seed_cron_thread_session`` does for threads, then mirror into it.

    The session KEY must match what the user's later inbound reply resolves to
    (``build_session_key``):
    - **Channel** (``chat_type="group"``): key is
      ``…:group:<chat_id>:<user_id>`` — user-isolated — so the seed MUST carry
      the **origin's real ``user_id``** (the member who scheduled the job), NOT
      a synthetic ``system:cron`` id, or the reply keys to a different session.
    - **1:1 DM** (``chat_type="dm"``): the key is ``…:dm:<chat_id>`` and does
      NOT embed ``user_id``, so any ``user_id`` resolves to the same session.
    ``chat_type`` mirrors the inbound handler's own choice
    (``"dm" if is_dm else "group"``, ``adapter.py``), so the seeded key is
    byte-identical to the reply's key.

    Returns True if a seed row was created and the brief mirrored, else False
    (caller falls back to the plain mirror). Best-effort — a delivery that
    already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return False
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        chat_type = "dm" if is_dm else "group"
        session_store = getattr(adapter, "_session_store", None)
        seeded_session_id: Optional[str] = None
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=str(chat_id),
                    chat_name=chat_name,
                    chat_type=chat_type,
                    user_id=str(user_id) if user_id else None,
                    thread_id=None,  # flat — the whole-channel/DM session
                    # Workspace scope: build_session_key embeds it in every
                    # Slack key, so a scoped reply only resolves to this row
                    # when the seed carries it too (see thread-seed docstring).
                    scope_id=str(scope_id) if scope_id else None,
                )
                # Create the flat session row so the mirror has a target and the
                # user's later plain reply joins the SAME session. Capture the
                # exact session id: the mirror must write into THIS row, not
                # re-discover it via origin heuristics (which bail out on
                # populated chats where the flat session coexists with
                # per-message thread sessions — live failure, Alice 2026-08-19).
                _entry = session_store.get_or_create_session(dest_source)
                seeded_session_id = getattr(_entry, "session_id", None)

        from gateway.mirror import mirror_to_session

        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=None,
            user_id=str(user_id) if user_id else None,
            session_id=seeded_session_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': seeded flat in_channel session on %s:%s (chat_type=%s)",
                job.get("id", "?"), platform_name, chat_id, chat_type,
            )
        return bool(ok)
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the "agent has no idea
        # about its own brief" bug (Alice 2026-08-19) — it must be visible in
        # production logs.
        logger.warning(
            "Job '%s': seeding in_channel session failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )
        return False


def _cron_job_origin_log_suffix(job: dict) -> str:
    """Return safe provenance details for security warnings about a cron job.

    The scheduler normally has no live HTTP request object when it detects a
    bad stored ``context_from`` reference. Including the job's saved origin
    makes future probe logs actionable without exposing secrets: platform/chat
    metadata for gateway-created jobs, and optional source-IP fields for API
    surfaces that persist them in origin metadata.
    """
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return ""

    fields = []
    for key in ("platform", "chat_id", "thread_id", "source_ip", "remote", "forwarded_for"):
        value = origin.get(key)
        if value is None:
            continue
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            fields.append(f"origin_{key}={text[:200]!r}")
    return " " + " ".join(fields) if fields else ""


def _plugin_cron_env_var(platform_name: str) -> str:
    """Return the cron home-channel env var registered by a plugin platform.

    Falls through the platform registry so plugins that set
    ``cron_deliver_env_var`` on their ``PlatformEntry`` get cron delivery
    support without editing this module.
    """
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name.lower())
        if entry and entry.cron_deliver_env_var:
            return entry.cron_deliver_env_var
    except Exception:
        pass
    return ""


def _is_known_delivery_platform(platform_name: str) -> bool:
    """Whether ``platform_name`` is a valid cron delivery target.

    Hardcoded built-ins in ``_KNOWN_DELIVERY_PLATFORMS`` are checked first;
    plugin platforms registered via ``PlatformEntry`` are accepted if they
    provide a ``cron_deliver_env_var``.
    """
    name = platform_name.lower()
    if name in _KNOWN_DELIVERY_PLATFORMS:
        return True
    return bool(_plugin_cron_env_var(name))


def _resolve_home_env_var(platform_name: str) -> str:
    """Return the env var name for a platform's cron home channel.

    Built-in platforms are in ``_HOME_TARGET_ENV_VARS``; plugin platforms are
    resolved from the platform registry.
    """
    name = platform_name.lower()
    env_var = _HOME_TARGET_ENV_VARS.get(name)
    if env_var:
        return env_var
    return _plugin_cron_env_var(name)


def _get_config_home_channel(platform_name: str):
    """Return the persisted ``HomeChannel`` for a platform from gateway config.

    ``/sethome`` declares ``config.yaml`` canonical (it is the only store that
    survives for relay-fronted logical platforms, whose adapters are not
    natively enabled) and mirrors the value into the legacy
    ``<PLATFORM>_HOME_CHANNEL`` env var only as a best-effort compatibility
    shim.  Cron historically read ONLY the env mirror, so a home channel that
    existed solely in config.yaml — e.g. Discord fronted by the relay
    connector, where no ``DISCORD_HOME_CHANNEL`` was ever exported — was
    invisible and jobs silently fell back to local-only.  Reading the
    canonical store here fixes that for every relay-fronted platform at once.
    """
    try:
        from gateway.config import load_gateway_config, Platform

        config = load_gateway_config()
        platform = Platform(platform_name.lower())
        return config.get_home_channel(platform)
    except Exception:
        logger.debug(
            "config home_channel lookup failed for platform %r",
            platform_name, exc_info=True,
        )
        return None


def _env_home_target_chat_id(platform_name: str) -> str:
    """Return the home chat id from the legacy env mirror only (no config).

    Reads through ``get_secret`` (not raw ``os.getenv``) so a profile-scoped
    secret scope wins in a multiplex gateway. ``DISCORD_HOME_CHANNEL`` lives in
    each profile's ``.env``; in a multiplex process the winning cron tick runs
    with the job-owning profile's scope installed (run_one_job sets it), so
    reading via ``get_secret`` resolves the OWNING profile's chat id rather
    than the host process's ``os.environ`` (#83182, chat-id leg — the token
    leg was fixed earlier; chat id / thread id resolve through the same leak).
    """
    env_var = _resolve_home_env_var(platform_name)
    if not env_var:
        return ""
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None  # type: ignore
    if get_secret is not None:
        value = get_secret(env_var, "")
        if not value:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = get_secret(legacy, "")
        return value or ""
    value = os.getenv(env_var, "")
    if not value:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(legacy, "")
    return value


def _get_home_target_chat_id(platform_name: str) -> str:
    """Return the configured home target chat/room ID for a delivery platform.

    Resolution order: platform env var (legacy mirror, kept first so an
    operator override keeps winning) → legacy env var name → the canonical
    ``home_channel`` block persisted in config.yaml by ``/sethome``.
    """
    value = _env_home_target_chat_id(platform_name)
    if value:
        return value
    home = _get_config_home_channel(platform_name)
    if home is not None and home.chat_id:
        return str(home.chat_id)
    return ""


def _get_home_target_thread_id(platform_name: str) -> Optional[str]:
    """Return the optional thread/topic ID for a platform home target.

    Telegram-only override: ``TELEGRAM_CRON_THREAD_ID`` takes precedence over
    ``TELEGRAM_HOME_CHANNEL_THREAD_ID`` for cron delivery. When topic mode is
    enabled, deliveries that land in the root DM (thread_id unset) end up in
    the system-only lobby where the user cannot reply — the gateway returns
    the lobby reminder and drops ``reply_to_message_id`` (#24409). Pointing
    cron at a dedicated topic via this env var lets replies work as expected
    without changing the lobby invariant.
    """
    env_var = _resolve_home_env_var(platform_name)
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None  # type: ignore

    def _scope_get(name: str) -> str:
        if get_secret is None:
            return ""
        v = get_secret(name, "")
        return v if v is not None else ""

    if platform_name.lower() == "telegram":
        cron_thread = _scope_get("TELEGRAM_CRON_THREAD_ID").strip()
        if cron_thread:
            return cron_thread
    if get_secret is not None:
        value = _scope_get(f"{env_var}_THREAD_ID").strip() if env_var else ""
        if not value and env_var:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = _scope_get(f"{legacy}_THREAD_ID").strip()
    else:
        value = os.getenv(f"{env_var}_THREAD_ID", "").strip() if env_var else ""
        if not value and env_var:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = os.getenv(f"{legacy}_THREAD_ID", "").strip()
    if value:
        return value
    # Canonical config.yaml fallback — same rationale as
    # _get_home_target_chat_id, and thread affinity only applies when the
    # chat itself resolved from the same config block (an env-provided chat
    # id keeps its env-provided thread semantics).
    if not _env_home_target_chat_id(platform_name):
        home = _get_config_home_channel(platform_name)
        if home is not None and home.thread_id:
            return str(home.thread_id)
    return None


def _iter_home_target_platforms():
    """Iterate built-in + plugin platform names that expose a home channel.

    Used by the ``deliver=origin`` fallback when the job has no origin.
    """
    for name in _HOME_TARGET_ENV_VARS:
        yield name
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.cron_deliver_env_var and entry.name not in _HOME_TARGET_ENV_VARS:
                yield entry.name
    except Exception:
        pass


def _relay_fronted_delivery_platforms(connected: set) -> set:
    """Logical platforms deliverable through a connected relay connector.

    ``get_connected_platforms()`` only sees NATIVELY configured platforms.
    On a relay-fronted deployment (relay in ``config.platforms``, the real
    platform credential living in the connector) the fronted platforms are
    absent from that set although fire-time routing delivers to them via
    ``resolve_delivery_transport`` + ``RelayAdapter.fronts_platform``. This
    keeps validation symmetric with routing by consulting the same
    env-derived deploy stamp (``GATEWAY_RELAY_PLATFORMS``) the live
    adapter's identity set is seeded from. No relay connected -> empty set,
    so native topologies keep the strict credential check unchanged.
    """
    if "relay" not in connected:
        return set()
    try:
        from gateway.relay import relay_fronted_platforms

        return relay_fronted_platforms()
    except Exception:
        logger.debug("relay fronted-platform lookup failed", exc_info=True)
        return set()


def cron_delivery_targets() -> list[dict]:
    """Return the platforms a cron job can auto-deliver to.

    Single source of truth for any UI (dashboard dropdown, etc.) that lets a
    user pick a cron delivery target. A platform is included when it is a valid
    cron delivery platform AND its gateway is configured (enabled + credentials
    present). Each entry reports whether the platform's home target (the
    room/channel cron posts to) is set — a platform can be configured for
    interactive use but still lack the home target an unattended cron job needs.

    Returns a list of dicts: ``{"id", "name", "home_target_set", "home_env_var"}``
    ordered by the gateway's canonical platform order. Callers should always
    prepend the implicit ``local`` option themselves — it needs no config.
    """
    targets: list[dict] = []
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        connected = {p.value for p in gateway_config.get_connected_platforms()}
        connected |= _relay_fronted_delivery_platforms(connected)
    except Exception:
        logger.debug("cron_delivery_targets: gateway config unavailable", exc_info=True)
        connected = set()

    for name in _iter_home_target_platforms():
        if name not in connected:
            continue
        if not _is_known_delivery_platform(name):
            continue
        env_var = _resolve_home_env_var(name)
        targets.append(
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "home_target_set": bool(_get_home_target_chat_id(name)),
                "home_env_var": env_var or None,
            }
        )

    # Bot Chat targets: one per local profile. Machine-local by design (the
    # scheduler delivers via a local chat subprocess), so the names listed
    # here are exactly the names that resolve at fire time — no gateway
    # config, no home channel needed.
    try:
        from hermes_cli.profiles import list_profile_names

        for profile_name in list_profile_names():
            targets.append(
                {
                    "id": f"{BOT_CHAT_PLATFORM}:{profile_name}",
                    "name": f"Bot Chat ({profile_name})",
                    "home_target_set": True,
                    "home_env_var": None,
                }
            )
    except Exception:
        logger.debug("cron_delivery_targets: profile listing unavailable", exc_info=True)
    return targets


def _origin_thread_is_stale(origin: dict) -> bool:
    """True when a Slack origin's thread is a stale creation-turn artifact.

    Relay-fronted Slack in thread-per-message mode stamps each top-level
    message's own id as the session thread (a session KEY, not a durable
    location). Jobs persisted before origin capture learned to drop that
    stamp carry it as ``origin.thread_id`` forever. Heuristic that repairs
    them at fire time without touching genuine threads: when the origin
    chat IS the configured Slack home chat (the ``/sethome`` conversation),
    a pinned origin thread is the creation-message artifact — the user's
    delivery expectation for their home conversation is top-level (or the
    home target's own configured thread). Non-home chats keep their
    threads: a job deliberately created inside a working thread stays there.
    """
    if str(origin.get("platform") or "").lower() != "slack":
        return False
    if not origin.get("thread_id"):
        return False
    home_chat = _get_home_target_chat_id("slack")
    return bool(home_chat) and str(origin.get("chat_id")) == str(home_chat)


def _origin_delivery_thread(origin: dict):
    """The thread a deliver=origin job should use, stale stamps dropped."""
    if _origin_thread_is_stale(origin):
        home_thread = _get_home_target_thread_id("slack")
        return home_thread if home_thread else None
    return origin.get("thread_id")


def _resolve_single_delivery_target(job: dict, deliver_value: str) -> Optional[dict]:
    """Resolve one concrete auto-delivery target for a cron job."""

    origin = _resolve_origin(job)

    if deliver_value == "local":
        return None

    # bot-chat[:<profile>] — checked before the generic platform:chat_id
    # split below so the profile-name argument is never misparsed as a
    # chat_id on an unknown platform.
    bot_chat_profile = parse_bot_chat_deliver_token(deliver_value)
    if bot_chat_profile is not None:
        return _resolve_bot_chat_target(job, bot_chat_profile)

    if deliver_value == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": _origin_delivery_thread(origin),
                # Resolution provenance for mirror eligibility (see
                # _target_mirror_eligible): this IS the origin conversation.
                "_resolved_from": "origin",
            }
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in _iter_home_target_platforms():
            chat_id = _get_home_target_chat_id(platform_name)
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": _get_home_target_thread_id(platform_name),
                    # The fallback stands in for the user's primary
                    # conversation (NOT a broadcast) — mirror-eligible so
                    # continuable crons work for script-provisioned jobs
                    # that never captured an origin.
                    "_resolved_from": "origin_fallback",
                }
        return None

    if ":" in deliver_value:
        platform_name, rest = deliver_value.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import (
            prepare_send_message_platforms,
            resolve_send_target,
        )

        prepare_send_message_platforms()
        # pass_unresolved_references: stored jobs have no model in the loop to react
        # to a resolution error, and a target the directory doesn't know
        # (fresh install, platform-native id) used to be handed to the
        # adapter as written. Dropping it here silently loses the job's
        # output.
        chat_id, thread_id, resolution_error = resolve_send_target(
            platform_key, rest, pass_unresolved_references=True
        )
        if resolution_error:
            logger.warning(
                "Invalid cron delivery target '%s': %s",
                deliver_value,
                resolution_error,
            )
            return None

        if (
            thread_id is None
            and platform_key == "slack"
            and origin
            and str(origin.get("platform") or "").lower() == platform_key
            and str(origin.get("chat_id")) == str(chat_id)
            and origin.get("thread_id")
            and not _origin_thread_is_stale(origin)
        ):
            thread_id = origin.get("thread_id")

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
            # Explicit platform:chat target — mirror-eligible only under the
            # job's own attach_to_session opt-in (see _target_mirror_eligible).
            "_resolved_from": "explicit",
        }

    platform_name = deliver_value
    if origin and origin.get("platform") == platform_name:
        chat_id = _get_home_target_chat_id(platform_name)
        if chat_id:
            return {
                "platform": platform_name,
                "chat_id": chat_id,
                "thread_id": _get_home_target_thread_id(platform_name),
            }
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if not _is_known_delivery_platform(platform_name):
        return None
    chat_id = _get_home_target_chat_id(platform_name)
    if not chat_id:
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": _get_home_target_thread_id(platform_name),
    }


def _get_bot_chat_delivery_timeout() -> int:
    """Timeout for one bot-chat delivery turn (the target bot runs a full
    agent turn on the injected output, so this is minutes, not seconds).

    ``cron.bot_chat_delivery_timeout_seconds`` in config.yaml; default 600.
    """
    try:
        cfg = load_config()
        value = int(cfg.get("cron", {}).get("bot_chat_delivery_timeout_seconds", 600))
        return value if value > 0 else 600
    except Exception:
        return 600


def _bot_chat_query_message(job: dict, content: str) -> str:
    """Compose the exact child query bytes for planning and dispatch."""
    job_name = job.get("name", job.get("id", "?"))
    return (
        f'[Cronjob "{job_name}" output — scheduled job, not the user. '
        f"Review it, act on anything that needs action, and summarize "
        f"for the chat.]\n\n{content}"
    )


def _deliver_to_bot_chat(job: dict, content: str, profile: str) -> Optional[str]:
    """Deliver job output into a profile's canonical Bot Chat as an inbound turn.

    Runs ``hermes [-p <profile>] chat --in ~ -c "Bot Chat" --create-if-missing
    -Q --query-file <tmp>`` — the exact lane Bot Mode agent-to-agent messages
    use, so the adopt-before-mint canonical-session rules apply and the target
    bot receives the output as a real user-role message it can act on.
    Alternation-safe by construction: this is an inbound turn on the chat
    command lane, not a transcript splice.

    ``profile`` is ``""`` for the job's own profile (subprocess inherits this
    scheduler's HERMES_HOME) or a validated local profile name.  Returns None
    on success or an error string for ``last_delivery_error``.
    """
    import shutil as _shutil
    import tempfile

    job_id = job.get("id", "?")

    hermes_bin = _shutil.which("hermes")
    if hermes_bin:
        argv = [hermes_bin]
    else:
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("hermes_cli") is not None:
                argv = [sys.executable, "-m", "hermes_cli.main"]
            else:
                return "bot-chat delivery failed"
        except Exception:
            return "bot-chat delivery failed"

    env = os.environ.copy()
    if profile:
        argv += ["-p", profile]
        # -p owns profile resolution in the child; a leftover HERMES_HOME
        # from THIS scheduler's profile must not shadow it.
        env.pop("HERMES_HOME", None)

    # The prefix tells the receiving bot this is scheduled output, not the
    # human typing. Planning calls the same helper before any side effect.
    message = _bot_chat_query_message(job, content)

    query_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", prefix="hermes-cron-botchat-",
            delete=False,
        ) as fh:
            fh.write(message)
            query_file = fh.name

        argv += [
            "chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing",
            "-Q", "--query-file", query_file,
        ]

        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_get_bot_chat_delivery_timeout(),
            env=env,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            logger.warning("Job '%s': bot-chat delivery confirmation unavailable", job_id)
            return "bot-chat delivery confirmation unavailable"
        logger.info(
            "Job '%s': bot-chat child completed without provider receipt", job_id
        )
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Job '%s': bot-chat delivery confirmation unavailable", job_id)
        return "bot-chat delivery confirmation unavailable"
    except Exception:
        logger.warning("Job '%s': bot-chat delivery confirmation unavailable", job_id)
        return "bot-chat delivery confirmation unavailable"
    finally:
        if query_file:
            try:
                os.unlink(query_file)
            except OSError:
                pass


def _normalize_deliver_value(deliver) -> str:
    """Normalize a stored/submitted ``deliver`` value to its canonical string form.

    The contract is that ``deliver`` is a string (``"local"``, ``"origin"``,
    ``"telegram"``, ``"telegram:-1001:17"``, or comma-separated combinations).
    Historically some callers — MCP clients passing an array, direct edits of
    ``jobs.json``, or stale code paths — have stored a list/tuple like
    ``["telegram"]``.  ``str(["telegram"])`` would serialize to the literal
    string ``"['telegram']"``, which is not a known platform and fails
    resolution silently.  Flatten lists/tuples into a comma-separated string
    so both forms work.  Returns ``"local"`` for anything falsy.
    """
    if deliver is None:
        return "local"
    if type(deliver) is str:
        return deliver or "local"
    if type(deliver) in {list, tuple}:
        parts = [p.strip() for p in deliver if type(p) is str and p.strip()]
        return ",".join(parts) if parts else "local"
    return "local"


def _normalize_delivery_target_identity(target: Any) -> dict:
    """Return a content-free target using only inert built-in scalar values."""
    if type(target) is not dict:
        raise ValueError("delivery target must be an object")
    platform = target.get("platform")
    chat_id = target.get("chat_id")
    thread_id = target.get("thread_id")
    if type(platform) is not str or not platform:
        raise ValueError("delivery target platform is invalid")
    if type(chat_id) not in {str, int} or chat_id in {"", 0}:
        raise ValueError("delivery target chat_id is invalid")
    if thread_id is not None and (
        type(thread_id) not in {str, int} or thread_id in {"", 0}
    ):
        raise ValueError("delivery target thread_id is invalid")
    normalized = {
        "platform": platform,
        "chat_id": str(chat_id),
        "thread_id": str(thread_id) if thread_id is not None else None,
    }
    resolved_from = target.get("_resolved_from")
    if resolved_from is not None:
        if type(resolved_from) is not str or resolved_from not in {
            "origin", "origin_fallback", "explicit",
        }:
            raise ValueError("delivery target provenance is invalid")
        normalized["_resolved_from"] = resolved_from
    return normalized


# Routing intent tokens — resolved at fire time, not create time, so a
# job created before Telegram was wired up will pick up Telegram once it
# comes online.  ``all`` expands into the set of connected platforms
# (those with a configured home chat_id) in _expand_routing_tokens.
_ROUTING_TOKENS = frozenset({"all"})

# Pseudo-platform for delivering job output INTO a profile's canonical
# "Bot Chat" session as a real inbound turn (the bot sees it, runs a turn,
# and can respond — Bot Mode's agent-to-agent lane, not a transcript
# mirror).  ``bot-chat`` targets the job's own profile; ``bot-chat:<name>``
# targets a named profile on THIS machine.  Deliberately excluded from the
# ``all`` routing token: ``all`` fans out to messaging home channels, and a
# bot-chat delivery costs a full agent turn.
BOT_CHAT_PLATFORM = "bot-chat"
BOT_CHAT_SELF_TARGET = "_self"


def parse_bot_chat_deliver_token(part: str) -> Optional[str]:
    """Return the target profile for a ``bot-chat[:<name>]`` deliver token.

    Returns ``""`` for the bare token (the job's own profile), the profile
    name for the explicit form, or ``None`` when ``part`` is not a bot-chat
    token at all.  Case-insensitive on the token; the profile name is
    normalized by the profile layer at resolve time.
    """
    raw = (part or "").strip()
    lowered = raw.lower()
    if lowered == BOT_CHAT_PLATFORM:
        return ""
    prefix = BOT_CHAT_PLATFORM + ":"
    if lowered.startswith(prefix):
        return raw[len(prefix):].strip()
    return None


def _resolve_bot_chat_target(job: dict, profile_arg: str) -> Optional[dict]:
    """Resolve a bot-chat deliver token to a concrete delivery target.

    ``profile_arg`` is ``""`` for the job's own profile (the HERMES_HOME
    this scheduler runs under — machine-local and self-referential, so no
    ``-p`` flag is needed at send time) or an explicit profile name that
    must exist in THIS machine's profile root.  Cross-machine delivery is
    intentionally unsupported: names resolve only against the local
    ``~/.hermes/profiles/`` tree, so same-named profiles on other gateways
    can never be targeted by accident.
    """
    if not profile_arg:
        # Own profile: the child inherits HERMES_HOME; the ledger still needs a
        # concrete non-empty requested-target identity.
        return {
            "platform": BOT_CHAT_PLATFORM,
            "chat_id": BOT_CHAT_SELF_TARGET,
            "thread_id": None,
        }
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists

        canon = normalize_profile_name(profile_arg)
        if not profile_exists(canon):
            logger.warning(
                "Job '%s': bot-chat delivery profile '%s' not found on this "
                "machine — skipping target",
                job.get("id", "?"), profile_arg,
            )
            return None
        return {"platform": BOT_CHAT_PLATFORM, "chat_id": canon, "thread_id": None}
    except Exception:
        logger.warning(
            "Job '%s': failed to resolve bot-chat profile '%s'",
            job.get("id", "?"), profile_arg, exc_info=True,
        )
        return None


def _expand_routing_tokens(part: str) -> List[str]:
    """Expand a routing-intent token to concrete platform names.

    ``all`` expands to every platform in ``_iter_home_target_platforms()``
    that has a configured home chat_id right now.  Unknown / non-token
    values pass through unchanged as a single-element list, so the caller
    can treat every token uniformly.
    """
    token = part.lower()
    if token not in _ROUTING_TOKENS:
        return [part]
    expanded: List[str] = []
    for platform_name in _iter_home_target_platforms():
        if _get_home_target_chat_id(platform_name):
            expanded.append(platform_name)
    return expanded


def _delivery_lane_value(job: dict, *, for_failure: bool = False):
    """Raw deliver-lane value for a run outcome: the failure lane when
    ``for_failure`` and the job overrides it, else ``deliver``. Keeps
    delivery bookkeeping (outcome classification, unresolved-origin,
    incident 'alerted' marking) reading the SAME lane the notice was
    actually routed through (NS-788 review finding B1)."""
    if for_failure:
        failure_deliver = job.get("failure_deliver")
        if failure_deliver is not None and str(failure_deliver).strip():
            return failure_deliver
    return job.get("deliver", "local")


def _resolve_delivery_targets(job: dict, *, for_failure: bool = False) -> List[dict]:
    """Resolve all concrete auto-delivery targets for a cron job.

    Accepts the legacy comma-separated ``deliver`` string plus the
    ``all`` routing-intent token, which expands to every platform with
    a configured home channel.  Tokens may be combined with explicit
    targets: ``origin,all`` and ``all,telegram:-100:17`` both work.
    Duplicate (platform, chat_id, thread_id) tuples are collapsed by the
    existing dedup pass.

    ``for_failure=True`` resolves failure-category engine notices
    (failure summaries, interrupted-run notices, drift/preflight
    alerts): when the job carries a ``failure_deliver`` value, targets
    resolve from it INSTEAD of ``deliver`` — ``failure_deliver: local``
    is the structural opt-out for shared channels (NS-788, Coatue).
    Absent ``failure_deliver``, failure delivery follows ``deliver``
    exactly as before.
    """
    deliver_raw = _delivery_lane_value(job, for_failure=for_failure)
    deliver = _normalize_deliver_value(deliver_raw)
    if deliver == "local":
        return []

    raw_parts = [p.strip() for p in deliver.split(",") if p.strip()]

    # Expand routing intents.
    parts: List[str] = []
    for raw in raw_parts:
        parts.extend(_expand_routing_tokens(raw))

    seen = {}
    targets = []
    for part in parts:
        target = _resolve_single_delivery_target(job, part)
        if target is not None:
            target = _normalize_delivery_target_identity(target)
            key = (target["platform"].lower(), target["chat_id"], target["thread_id"])
            if key not in seen:
                seen[key] = target
                targets.append(target)
            else:
                # OR-merge resolution provenance on dedup: "origin,all" (either
                # order) resolving to the same chat must keep the
                # origin/origin_fallback tag — a mirror-eligible token must not
                # lose eligibility to token order (see _target_mirror_eligible).
                kept = seen[key]
                if _MIRROR_PROVENANCE_RANK.get(str(target.get("_resolved_from") or ""), 0) > \
                        _MIRROR_PROVENANCE_RANK.get(str(kept.get("_resolved_from") or ""), 0):
                    kept["_resolved_from"] = target.get("_resolved_from")
    return targets


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    targets = _resolve_delivery_targets(job)
    return targets[0] if targets else None


# Media extension sets — audio routing is centralized in gateway.platforms.base
# via should_send_media_as_audio() so Telegram-specific rules stay in one place.
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter,
    chat_id: str,
    media_files: list,
    metadata: dict | None,
    loop,
    job: dict,
    platform=None,
    receipts_out: Optional[list] = None,
) -> list:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.

    Returns a list of per-file error strings (empty when every attachment
    delivered). Callers surface these into the job's delivery errors so a
    dropped attachment is visible in ``last_error``/run status instead of
    only in the gateway log (the silent-drop half of the manual-run
    attachment bug: text delivered, file vanished, job marked ok).
    """
    from pathlib import Path

    from gateway.platforms.base import (
        BasePlatformAdapter,
        SendResult,
        should_send_media_as_audio,
    )

    errors: list = []
    requested = [(str(p), v) for p, v in (media_files or [])]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Report paths the safety filter dropped: the model referenced them in
    # MEDIA: tags but they will never be sent (missing file, denied prefix,
    # or strict-mode policy miss).
    kept = {p for p, _ in media_files}
    for raw_path, _v in requested:
        try:
            from gateway.platforms.base import validate_media_delivery_path

            if validate_media_delivery_path(raw_path) not in kept:
                errors.append(
                    f"attachment dropped by media path policy: {raw_path}"
                )
        except Exception:
            errors.append(f"attachment dropped by media path policy: {raw_path}")

    for media_ordinal, (media_path, _is_voice) in enumerate(media_files):
        try:
            send_metadata = dict(metadata or {})
            send_metadata["_transport_receipt_component"] = "media"
            send_metadata["_transport_receipt_ordinal"] = media_ordinal
            ext = Path(media_path).suffix.lower()
            route_platform = platform if platform is not None else getattr(adapter, "platform", None)
            if should_send_media_as_audio(route_platform, ext, is_voice=_is_voice):
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=send_metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=send_metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=send_metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=send_metadata)

            from agent.async_utils import safe_schedule_threadsafe
            future = safe_schedule_threadsafe(coro, loop)
            if future is None:
                msg = f"cannot send media {media_path}: gateway loop unavailable"
                logger.warning("Job '%s': %s", job.get("id", "?"), msg)
                errors.append(msg)
                return errors
            try:
                # Large attachments (long TTS audio, concatenated recordings,
                # big exports) can legitimately exceed a fixed 30s upload
                # window. Configurable, matching the other cron timeouts
                # (cron.media_send_timeout_seconds in config.yaml, or the
                # HERMES_CRON_MEDIA_SEND_TIMEOUT env override).
                result = future.result(timeout=_get_media_send_timeout())
            except TimeoutError:
                future.cancel()
                raise
            receipt_bound = (
                type(metadata) is dict
                and "_transport_receipt_requested_target" in metadata
            )
            if type(result) is SendResult:
                result_success = result.success is True
                result_error = result.error
                if receipts_out is not None:
                    receipts_out.extend(result.receipts)
            else:
                legacy = _inert_legacy_send_result_fields(result)
                if receipt_bound or legacy is None:
                    errors.append("media adapter returned an invalid result")
                    return errors
                result_success = legacy["success"] is True
                result_error = legacy["error"]
            if not result_success:
                msg = (
                    f"media send failed for {media_path}: "
                    f"{result_error or 'unknown'}"
                )
                logger.warning("Job '%s': %s", job.get("id", "?"), msg)
                errors.append(msg)
        except Exception as e:
            # Argument-less exceptions (notably TimeoutError, the most likely
            # failure on this path) have an empty str(), which would render
            # the reason as nothing at all. Fall back to the class name.
            msg = (
                f"failed to send media {media_path}: {str(e) or type(e).__name__}"
            )
            logger.warning("Job '%s': %s", job.get("id", "?"), msg)
            errors.append(msg)
    return errors


def _confirm_adapter_delivery(send_result, job_id: str = "?", unverified: Optional[list] = None) -> bool:
    """Return True only if ``send_result`` unambiguously confirms delivery.

    A live adapter that returns ``None`` (e.g. a swallowed exception, a busy
    platform, or a code path that returns early without producing a
    ``SendResult``) must NOT be treated as success — doing so causes the
    scheduler to log ``"delivered to <chat> via live adapter"`` while the
    gateway never actually sees the message (#47056).

    Likewise, a result carrying no ``success`` at all (a partial mock, or a
    ``dict`` from a code path that never reached the adapter) is a contract
    violation: it does not actually tell us whether the send succeeded.
    Require an explicit, truthy ``success`` to count as confirmed.

    Both shapes are inspected the same way, because ``_deliver_to_platform``
    returns either a ``SendResult`` object or a plain ``dict``:

    * ``delivered is False`` is a REJECTION even when ``success`` is truthy.
      The silence-narration filter returns
      ``{"success": True, "delivered": False}`` — a successfully *dropped*
      message, not a delivered one.  Reading only ``success`` there is how a
      cron brief was logged as delivered while the user got nothing (#77763).
    * No ``message_id`` and no ``raw_response`` means we have no positive
      evidence of a send.  That is not proof of failure either (some adapters
      legitimately return a bare success), so it is still accepted — but
      logged at WARNING so an UNVERIFIED delivery is visible in the log
      instead of masquerading as a confirmed one.  Telegram ``SendResult``
      objects carry ``message_id``; the dict-filter shape does not.
    """
    from gateway.platforms.base import SendResult

    if type(send_result) is dict:
        if type(send_result.get("success")) is not bool:
            return False
        success = send_result["success"]
        delivered = send_result.get("delivered")
        message_id = send_result.get("message_id")
        raw_response = (
            send_result.get("raw_response")
            if type(send_result.get("raw_response")) is dict else None
        )
    elif type(send_result) is SendResult:
        if type(send_result.success) is not bool:
            return False
        success = send_result.success
        delivered = getattr(send_result, "delivered", None)
        message_id = send_result.message_id
        raw_response = send_result.raw_response
    else:
        legacy = _inert_legacy_send_result_fields(send_result)
        if legacy is None:
            return False
        success = legacy["success"]
        delivered = None
        message_id = legacy["message_id"]
        raw_response = legacy["raw_response"]
    if success is not True or delivered is False:
        return False
    if message_id is None and not raw_response:
        logger.warning(
            "Job '%s': live adapter reported success with no delivery evidence "
            "(no message_id, no raw_response) — treating as delivered but "
            "UNVERIFIED",
            job_id,
        )
        if unverified is not None:
            unverified.append(True)
    return True


def _inert_legacy_send_result_fields(send_result: Any) -> Optional[dict]:
    """Read the one supported inert legacy container without object magic."""
    if type(send_result) is not SimpleNamespace:
        return None
    fields = object.__getattribute__(send_result, "__dict__")
    if type(fields) is not dict or type(fields.get("success")) is not bool:
        return None
    error = fields.get("error")
    message_id = fields.get("message_id")
    if error is not None and type(error) is not str:
        error = None
    if message_id is not None and type(message_id) not in {str, int}:
        message_id = None
    return {
        "success": fields["success"],
        "error": error,
        "message_id": message_id,
        "raw_response": fields.get("raw_response") if type(fields.get("raw_response")) is dict else None,
    }


def _is_channel_dm_topic(
    runtime_adapter: Any,
    chat_id: Any,
    loop: Any,
    job_id: str,
) -> bool:
    """Decide whether an (already-ambiguous) Telegram topic target is a genuine
    Bot API *channel* Direct-Messages topic (route via
    ``direct_messages_topic_id``) rather than a forum-style topic in a private
    chat (route via ``message_thread_id``).

    Callers gate this on the ambiguous shape first
    (``telegram:<positive_chat_id>:<numeric_thread_id>``) — that shape is
    identical for both cases, so shape alone cannot decide (this was the #52060
    regression).  The real signal is the chat *type*: a genuine channel DM topic
    lives on a ``channel`` chat.  Probe the live adapter's ``get_chat_info`` once
    and only return True when the chat is a channel.

    Fails SAFE to ``message_thread_id`` (returns False) for adapters without a
    probe, or any probe error/timeout — that is the pre-#22773 behaviour and the
    correct default for the common forum-topic case.
    """
    # Resolve on the CLASS, not the instance (general pitfall #11): a MagicMock
    # instance auto-creates a truthy ``get_chat_info`` attribute, so an
    # instance-level probe would misclassify test doubles. Real adapters expose
    # the coroutine on the class regardless.
    get_chat_info = getattr(type(runtime_adapter), "get_chat_info", None)
    if not callable(get_chat_info):
        return False
    try:
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            get_chat_info(runtime_adapter, str(chat_id)), loop,  # type: ignore[arg-type]
        )
        if future is None:
            return False
        # Lighter than a send (metadata-only Bot API call), so a shorter bound
        # than the 30s/60s send waits elsewhere in this file is intentional.
        info = future.result(timeout=10)
    except Exception:
        logger.debug(
            "Job '%s': get_chat_info probe failed for chat=%s — "
            "defaulting to message_thread_id routing",
            job_id, chat_id, exc_info=True,
        )
        return False
    is_channel = isinstance(info, dict) and str(info.get("type") or "").lower() == "channel"
    if is_channel:
        logger.info(
            "Job '%s': chat=%s is a channel — routing via direct_messages_topic_id",
            job_id, chat_id,
        )
    return is_channel


def _receipt_text_chunks_for_target(
    adapters: Any, platform_name: str, content: str, media_files=None,
) -> list[str]:
    """Return exact adapter-planned chunks when the adapter can prove them.

    Opaque and standalone transports deliberately get one logical component:
    a later multi-ack cannot be upgraded to delivery unless it binds that plan.
    Matrix and Telegram expose this preflight because their send paths own the
    deterministic formatting/splitting algorithm.
    """
    if not adapters:
        if platform_name.lower() == "telegram":
            from tools.send_message_senders import _plan_standalone_telegram_text

            return _plan_standalone_telegram_text(
                content, media_files=media_files,
            )[1]
        return [content]
    candidate = None
    try:
        from gateway.config import Platform
        candidate = adapters.get(Platform(platform_name.lower()))
    except Exception:
        candidate = None
    if candidate is None:
        try:
            candidate = adapters.get(platform_name) or adapters.get(platform_name.lower())
        except Exception:
            candidate = None
    if candidate is None:
        if platform_name.lower() == "telegram":
            from tools.send_message_senders import _plan_standalone_telegram_text

            return _plan_standalone_telegram_text(
                content, media_files=media_files,
            )[1]
        return [content]
    planner = getattr(candidate, "plan_transport_text", None)
    if not callable(planner):
        return [content]
    try:
        chunks = planner(content)
    except Exception as exc:
        raise ValueError("transport planner failed before dispatch") from exc
    if (
        type(chunks) not in {list, tuple}
        or len(chunks) == 0
        or not all(type(chunk) is str and chunk for chunk in chunks)
    ):
        raise ValueError("transport planner returned invalid chunks")
    return list(chunks)


def _persist_target_text_receipts(
    receipts: Any,
    attempts: dict,
    requested_target: dict[str, str],
    components: Optional[set[str]] = None,
    expected_actual_target: Optional[dict[str, str]] = None,
) -> bool:
    """Persist exact acknowledgements and prove the selected planned set.

    Matching partial acknowledgements are retained even when the final result
    is false. ``components=None`` requires every preregistered component;
    callers passing a set require only those component kinds.
    """
    from gateway.platforms.base import TransportReceipt, TransportTarget

    if type(receipts) is not tuple:
        return False
    if type(attempts) is not dict or type(requested_target) is not dict:
        return False
    if components is not None and type(components) is not set:
        return False
    if expected_actual_target is not None and type(expected_actual_target) is not dict:
        return False
    if not all(type(receipt) is TransportReceipt for receipt in receipts):
        return False
    if not attempts:
        return bool(receipts)
    expected = {
        key for key in attempts
        if key[:3] == (
            requested_target["platform"], requested_target["chat_id"],
            requested_target["thread_id"],
        ) and (components is None or key[3] in components)
    }
    observed = set()
    persisted_all = True
    expected_actual = expected_actual_target or requested_target
    try:
        planned_target = (
            expected_actual["platform"],
            expected_actual["chat_id"],
            expected_actual["thread_id"],
        )
    except (KeyError, TypeError):
        return False
    if not all(type(value) is str for value in planned_target):
        return False
    for receipt in receipts:
        try:
            requested = receipt.requested_target
            key = (
                requested.platform, requested.chat_id,
                requested.thread_id or "", receipt.component, receipt.ordinal,
            )
            attempt_id = attempts.get(key)
            persisted = bool(attempt_id) and record_transport_receipt(attempt_id, receipt)
        except Exception:
            persisted = False
            key = None
        actual = receipt.actual_target
        actual_target = (
            (actual.platform, actual.chat_id, actual.thread_id or "")
            if type(actual) is TransportTarget
            else None
        )
        if (
            persisted
            and key is not None
            and receipt.outcome == "delivered"
            and actual_target == planned_target
        ):
            observed.add(key)
        else:
            persisted_all = False
    return bool(expected) and persisted_all and observed == expected


def _receipt_delivery_outcome(execution_id: str) -> Optional[str]:
    """Project a transport outcome only when this execution has a receipt plan."""
    try:
        counts = receipt_summary(execution_id)
    except Exception:
        return None
    if counts.get("unknown", 0) > 0:
        return "unknown"
    if counts.get("failed", 0) > 0:
        return "failed"
    if counts.get("delivered", 0) > 0 and counts.get("targets_delivered", 0) > 0:
        return "delivered"
    return None


def _cron_delivery_notify_enabled(cfg: Optional[dict]) -> bool:
    """Resolve ``cron.delivery.notify`` (config.yaml). Default True.

    Only an explicit boolean ``False`` (or a YAML ``false``/``off`` that parses
    to it) disables the push notification; a missing/malformed section keeps
    the default so a typo can never silently make cron briefs silent.
    """
    try:
        cron_cfg = (cfg or {}).get("cron")
        if not isinstance(cron_cfg, dict):
            return True
        delivery_cfg = cron_cfg.get("delivery")
        if not isinstance(delivery_cfg, dict):
            return True
        return delivery_cfg.get("notify", True) is not False
    except Exception:
        return True


def _record_delivery_verification(job: dict, unverified_targets: list) -> None:
    """Persist the UNVERIFIED-delivery marker on the job record.

    ``last_delivery_unverified`` is a list of ``platform:chat_id`` targets
    whose live adapter acked the send with no message_id/raw_response, or
    ``None`` once a run delivered with positive evidence (or to no live
    target). Skips the write when nothing changed so the common verified
    path costs no jobs.json save. Never raises — status bookkeeping must not
    fail a delivery.
    """
    new_value = list(unverified_targets) or None
    if (job.get("last_delivery_unverified") or None) == new_value:
        return
    try:
        from cron.jobs import update_job

        update_job(job["id"], {"last_delivery_unverified": new_value})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "Job '%s': could not record delivery verification: %s", job.get("id"), exc,
        )


def _deliver_result(
    job: dict,
    content: str,
    adapters=None,
    loop=None,
    *,
    execution_id: Optional[str] = None,
    fire_identity: Optional[str] = None,
    for_failure: bool = False,

) -> Optional[str]:
    """
    Deliver job output to the configured target(s) (origin chat, specific platform, etc.).

    When ``adapters`` and ``loop`` are provided (gateway is running), tries to
    use the live adapter first — this supports E2EE rooms (e.g. Matrix) where
    the standalone HTTP path cannot encrypt.  Falls back to standalone send if
    the adapter path fails or is unavailable.

    ``for_failure=True`` routes failure-category engine notices through the
    job's ``failure_deliver`` override when present (NS-788).

    Returns None on success, or an error string on failure.
    """
    if type(job) is not dict or type(content) is not str:
        return "delivery input is invalid; no delivery was sent"
    try:
        targets = _resolve_delivery_targets(job, for_failure=for_failure)
    except (TypeError, ValueError):
        return "delivery target is invalid; no delivery was sent"
    if not targets:
        deliver_value = _normalize_deliver_value(
            _delivery_lane_value(job, for_failure=for_failure)
        )
        if deliver_value == "local":
            return None  # local-only jobs don't deliver — not a failure
        # deliver=origin with no resolvable origin and no configured home
        # channels: treat as local rather than reporting an error.  CLI-created
        # jobs never capture a {platform, chat_id} origin, so failing here would
        # make every CLI `deliver=origin` (or auto-detect) job emit a spurious
        # "no delivery target resolved" error on every run (#43014).  The output
        # is still persisted in last_output for `cron list`/resume.
        if deliver_value == "origin":
            logger.info(
                "Job '%s': deliver=origin but no origin or home channels — "
                "skipping delivery (output saved in last_output)",
                job.get("name", job.get("id", "?")),
            )
            return None
        msg = f"no delivery target resolved for deliver={deliver_value}"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    # Restart-safe workers intentionally have no live gateway adapter objects.
    # Hand the send back through a durable queue so the current or replacement
    # gateway performs it with relay/E2EE parity.  The execution id is the
    # idempotency key; the queue never retries an uncertain claimed send.
    # Match on this job's own attempt: a worker's script may itself dispatch
    # another job in-process (``hermes cron run``), and that nested delivery
    # must not be keyed under the outer execution id.
    external_execution = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER", "")
    if (
        external_execution
        and adapters is None
        and external_execution == str(job.get("execution_id") or "")
    ):
        from cron.delivery_queue import enqueue_and_wait

        return enqueue_and_wait(
            external_execution,
            job,
            content,
            for_failure=for_failure,
        )

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    user_cfg = None
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    # cron.delivery.notify (default True): mark live-adapter cron sends as
    # FINAL notifications so the platform pushes them (Telegram's "important"
    # mode otherwise sends with disable_notification=True). Configurable so
    # operators who prefer silent briefs can opt back out.
    notify_delivery = _cron_delivery_notify_enabled(user_cfg)
    # Set when a live adapter acked a send with NO delivery evidence (no
    # message_id / raw_response — the Slack/Matrix/Mattermost bare
    # SendResult(success=True) shape). Persisted on the job as
    # ``last_delivery_unverified`` so `hermes cron list` shows the state
    # instead of it living only in a WARNING log line.
    unverified_targets: list = []

    if wrap_response:
        task_name = job.get("name", job["id"])
        job_id = job.get("id", "")
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"(job_id: {job_id})\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"To stop or manage this job, send me a new message (e.g. \"stop reminder {task_name}\")."
        )
    else:
        delivery_content = content

    # Extract MEDIA: tags so attachments are forwarded as files, not raw text
    from gateway.platforms.base import (
        BasePlatformAdapter,
        SendResult,
        TransportReceipt,
        TransportTarget,
    )

    # Bridge gateway media-policy config (strict / allow_dirs / trust_recent)
    # into the env vars the path validator reads. Gateway startup does this
    # at boot; a standalone process (manual `hermes cron run` from the CLI,
    # a cron tick without the gateway) historically did NOT — so manual runs
    # filtered attachment paths under a DIFFERENT policy than scheduled runs
    # and silently dropped files the gateway would deliver. Idempotent,
    # env-wins, never raises.
    from gateway.media_policy import apply_media_policy_env

    apply_media_policy_env(user_cfg)

    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)
    requested_media = [(str(p), v) for p, v in media_files]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Attachments the policy filter dropped will never be sent on ANY lane —
    # record them up front so the run status says so (previously one
    # stderr WARNING was the only trace: text delivered, file vanished).
    _policy_dropped = len(requested_media) - len(media_files)
    policy_drop_errors = (
        [
            f"{_policy_dropped} media attachment(s) dropped by media path "
            "policy (missing file, denied prefix, or strict-mode miss); "
            "see gateway.strict / media_delivery_allow_dirs in config.yaml"
        ]
        if _policy_dropped > 0
        else []
    )

    # Resolve the delivery-mirror gate ONCE (default off). When on, each
    # successful delivery is also appended to the target chat's gateway session
    # transcript so a user reply in that chat sees the cron output in context.
    # Mirror the CLEAN, unwrapped output (not the cron header/footer).
    try:
        mirror_enabled = _cron_mirror_delivery_enabled(job, user_cfg)
    except Exception:
        mirror_enabled = False
    # Keep the cleaned delivery text available independently of the optional
    # transcript-mirror knob. Continuable surfaces (notably in_channel) must
    # seed their target session even when attach_to_session=false and
    # cron.mirror_delivery=false; gating this value on mirror_enabled makes
    # the seed receive an empty string and return False, which is exactly the
    # live failure reproduced three times on Alice (job ef7bd2869d15).
    _, mirror_text = BasePlatformAdapter.extract_media(content)
    mirror_text = (mirror_text or "").strip()

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    delivery_errors = []
    # Direct isolated callers may opt out of ledger persistence. Scheduler,
    # provider, and manual routes pass the exact durable identity explicitly;
    # never rely on mutating their job snapshot.
    receipt_attempts = {}
    if execution_id is None:
        execution_id = job.get("execution_id")
    if fire_identity is None:
        fire_identity = job.get("fire_identity")
    if fire_identity is None:
        fire_identity = execution_id
    receipt_planning_adapters = (
        adapters
        if adapters is not None
        and loop is not None
        and getattr(loop, "is_running", lambda: False)()
        else None
    )
    if execution_id is not None:
        if (
            type(execution_id) is not str
            or not execution_id
            or type(fire_identity) is not str
            or not fire_identity
        ):
            return "delivery receipt identity is invalid; no delivery was sent"
        receipt_plan = []
        for target in targets:
            target_identity = dict(target)
            target_identity["thread_id"] = target_identity["thread_id"] or ""
            if target_identity["platform"] == BOT_CHAT_PLATFORM:
                receipt_plan.append({
                    "target": target_identity,
                    "component": "text",
                    "ordinal": 0,
                    "content": _bot_chat_query_message(job, content),
                })
                continue
            if cleaned_delivery_content.strip():
                try:
                    planned_chunks = _receipt_text_chunks_for_target(
                        receipt_planning_adapters, target["platform"],
                        cleaned_delivery_content.strip(),
                        media_files=media_files,
                    )
                except (TypeError, ValueError):
                    return "delivery receipt planner is invalid; no delivery was sent"
                for ordinal, chunk in enumerate(planned_chunks):
                    receipt_plan.append({
                        "target": target_identity, "component": "text", "ordinal": ordinal,
                        "content": chunk,
                    })
            for ordinal, (media_path, _is_voice) in enumerate(media_files):
                if type(media_path) is not str:
                    return "delivery media identity is invalid; no delivery was sent"
                receipt_plan.append({
                    "target": target_identity, "component": "media", "ordinal": ordinal,
                    "content": media_path,
                })
        if receipt_plan:
            try:
                attempts = preregister_receipt_plan(
                    execution_id,
                    fire_identity=fire_identity,
                    components=receipt_plan,
                )
            except Exception:
                # The DB exception can include filesystem/provider details; it
                # is not a safe delivery/operator payload.
                logger.warning("Job '%s': receipt-plan preregistration failed", job["id"])
                return "delivery receipt plan could not be persisted; no delivery was sent"
            for attempt in attempts:
                receipt_attempts[(
                    attempt["platform"], attempt["chat_id"], attempt["thread_id"],
                    attempt["component"], attempt["ordinal"],
                )] = attempt["id"]

    for target in targets:
        platform_name = target["platform"]
        chat_id = target["chat_id"]
        thread_id = target.get("thread_id")
        receipt_requested_target = {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id or "",
        }

        # bot-chat targets don't ride a gateway adapter: the output becomes a
        # real inbound turn in the target profile's canonical Bot Chat via the
        # chat CLI lane (the same one Bot Mode agent-to-agent sends use). The
        # bot runs a turn and can respond — handled before the Platform enum
        # below, which knows nothing about this pseudo-platform.
        if platform_name == BOT_CHAT_PLATFORM:
            bot_chat_profile = "" if chat_id == BOT_CHAT_SELF_TARGET else chat_id
            bot_chat_error = _deliver_to_bot_chat(job, content, bot_chat_profile)
            if receipt_attempts:
                requested = TransportTarget(
                    BOT_CHAT_PLATFORM,
                    chat_id,
                    thread_id,
                )
                if bot_chat_error == "bot-chat delivery failed":
                    receipt = TransportReceipt(
                        outcome="failed",
                        requested_target=requested,
                        failure_kind="pre_dispatch",
                        component="text",
                        ordinal=0,
                    )
                else:
                    # Child completion has no provider/session acknowledgement;
                    # any post-spawn result remains ambiguous.
                    receipt = TransportReceipt(
                        outcome="unknown",
                        requested_target=requested,
                        component="text",
                        ordinal=0,
                    )
                if receipt.outcome == "unknown":
                    attempt_id = receipt_attempts.get((
                        BOT_CHAT_PLATFORM, chat_id,
                        thread_id or "", "text", 0,
                    ))
                    persisted = bool(attempt_id) and observe_transport_unknown(
                        attempt_id, receipt,
                    )
                else:
                    persisted = _persist_target_text_receipts(
                        (receipt,), receipt_attempts, receipt_requested_target,
                        components={"text"},
                    )
                if not persisted:
                    delivery_errors.append(
                        "bot-chat delivery receipt could not be persisted; "
                        "delivery is unknown"
                    )
                elif receipt.outcome == "unknown":
                    delivery_errors.append(
                        "bot-chat delivery confirmation unavailable"
                    )
                else:
                    delivery_errors.append("bot-chat delivery failed")
            elif bot_chat_error:
                delivery_errors.append(bot_chat_error)
            continue

        # Diagnostic: log thread_id for topic-aware delivery debugging
        origin = _resolve_origin(job) or {}
        origin_thread = origin.get("thread_id")
        if origin_thread and not thread_id:
            logger.warning(
                "Job '%s': origin has thread_id=%s but delivery target lost it "
                "(deliver=%s, target=%s)",
                job["id"], origin_thread, job.get("deliver", "local"), target,
            )
        elif thread_id:
            logger.debug(
                "Job '%s': delivering to %s:%s thread_id=%s",
                job["id"], platform_name, chat_id, thread_id,
            )

        # Mirror scope: the origin conversation, the home-channel FALLBACK for
        # an origin-less deliver=origin job (a script-provisioned managed cron
        # standing in for the user's primary conversation — not a broadcast),
        # or an explicit target the job opted into via attach_to_session.
        # Broadcast/fan-out targets are never mirrored (_target_mirror_eligible).
        origin_target = _target_matches_origin(origin, platform_name, chat_id, thread_id)
        mirror_this_target = mirror_enabled and _target_mirror_eligible(
            job, target, global_mirror=mirror_enabled, origin_match=origin_target,
        )
        # Pass the origin's user_id so a per-user-isolated group chat resolves to
        # the exact member who scheduled the job — parity with send_message.
        # Resolved for ANY origin-matching target (not just mirror-enabled):
        # the in_channel seed below needs it too, and it must not depend on
        # the attach_to_session/mirror opt-in.
        origin_user_id = origin.get("user_id") if origin_target else None

        # DM shape of this target, needed by BOTH the in_channel flatten gate
        # below and the seed/_seed_cron_channel_session chat_type further down:
        # a 1:1 DM keys as ``dm`` (Slack DM channel ids start with "D"; or the
        # origin says so), everything else as ``group``.
        origin_chat_type = str(origin.get("chat_type") or "").lower()
        is_dm_target = origin_chat_type == "dm" or (
            not origin_chat_type and str(chat_id).startswith("D")
        )

        # Shared continuable-target gate for the in_channel surface. The
        # thread-flatten and the flat-session seed MUST use the SAME gate —
        # if they drift, the brief and its continuation session land in
        # different places (the split-surface bug the flatten exists to
        # prevent). Origin targets qualify unconditionally (independent of the
        # attach_to_session / mirror opt-in — see 3c52d3589f); non-origin
        # mirror-eligible targets (origin_fallback / opted-in explicit)
        # qualify only when the seed can actually create a resolvable session
        # (_inchannel_seed_allowed: DM-shaped, or a known user_id for
        # user-isolated group keys).
        inchannel_continuable = origin_target or (
            mirror_this_target
            and _inchannel_seed_allowed(is_dm=is_dm_target, user_id=origin_user_id)
        )

        # Built-in names resolve to their enum member; plugin platform names
        # create dynamic members via Platform._missing_().
        try:
            platform = Platform(platform_name.lower())
        except (ValueError, KeyError):
            msg = f"unknown platform '{platform_name}'"
            logger.warning("Job '%s': %s", job["id"], msg)
            delivery_errors.append(msg)
            continue

        from gateway.delivery import resolve_delivery_transport

        target_adapters = adapters
        if isinstance(adapters, SharedRouteAdapters):
            # Credentialless satellite: the primary adapter is a valid
            # transport for THIS target only when an exact primary route maps
            # it to this profile (#101113). Miss → fail closed below.
            shared = adapters.get(platform, target)
            target_adapters = {platform: shared} if shared is not None else {}
        transport = resolve_delivery_transport(platform, config, target_adapters)
        if transport is not None:
            pconfig = transport.config
            runtime_adapter = transport.adapter
        else:
            # No live transport. A relay-fronted platform's ONLY sender is the
            # gateway's live relay adapter — there is no standalone fallback
            # (the connector owns the credential). A manual in-process run
            # (`hermes cron run`) has no live relay adapter, so surface the
            # accurate remediation instead of the native configured/enabled
            # gate, which misdiagnoses relay-fronted deployments.
            from gateway.relay import relay_fronted_platforms

            if platform_name in relay_fronted_platforms():
                msg = (
                    f"platform '{platform_name}' is relay-fronted and has no "
                    "live gateway transport; start the gateway (its ticker "
                    "owns relay-fronted delivery and will fire the job on "
                    "schedule)"
                )
                logger.warning("Job '%s': %s", job["id"], msg)
                delivery_errors.append(msg)
                continue
            # Preserve the existing standalone delivery path, which uses the
            # logical platform's configured credential.
            pconfig = config.platforms.get(platform)
            runtime_adapter = None

        if transport is not None and transport.is_relay:
            # A relay transport carries the RELAY adapter's config, and
            # resolve_delivery_transport already applied relay's enablement
            # rule (config block absent OR enabled). The logical platform is
            # deliberately NOT natively enabled in a relay-fronted deployment
            # (its credential lives in the connector), so the native
            # configured/enabled gate below must not apply — it used to
            # reject exactly the targets the relay was resolved to serve.
            if pconfig is None:
                from gateway.config import PlatformConfig
                pconfig = PlatformConfig(enabled=True)
        elif not pconfig or not pconfig.enabled:
            msg = f"platform '{platform_name}' not configured/enabled"
            logger.warning("Job '%s': %s", job["id"], msg)
            delivery_errors.append(msg)
            continue

        # Prefer the resolved live transport when the gateway is running. This
        # supports E2EE native adapters and relay-fronted logical platforms.
        # The live-send path (which SEEDS the flat in_channel continuation
        # session via _seed_cron_channel_session) needs not just a live adapter
        # but a running event loop to schedule the async send onto. Compute that
        # gate ONCE so the in_channel thread_id clear below stays in lockstep
        # with the live-send/seed block further down (they used to drift): an
        # adapter can be present while the loop is absent/not-running, in which
        # case the live-send block is skipped and delivery falls through to the
        # standalone path — which cannot seed the flat session (r3609147550).
        live_adapter_ready = (
            runtime_adapter is not None
            and loop is not None
            and getattr(loop, "is_running", lambda: False)()
        )
        delivered = False
        target_errors = []
        ambiguous_live_timeout = False

        # Continuable cron surface (D1/D2/D6): resolve the delivery surface for
        # this platform generically from its config ``extra``. Default "thread"
        # (today's behaviour, byte-identical). "in_channel" delivers the brief
        # FLAT into the channel (no dedicated thread) so a plain channel reply
        # continues the job in-context via the shared-channel session
        # ``(platform, chat_id, None)`` — the same bucket ``reply_in_thread:
        # false`` routes inbound channel messages to. The key is read
        # generically here (any platform); the ``in_channel`` branch is gated on
        # the adapter capability flag ``supports_inchannel_continuable`` so an
        # unsupported platform fails SAFE to "thread" (Slack is the first
        # consumer; "first consumer ≠ definition").
        surface_mode = _resolve_cron_surface_mode(pconfig, platform_name)
        in_channel_surface = surface_mode == "in_channel"
        if in_channel_surface and runtime_adapter is not None:
            # Per-platform capability first: one RelayAdapter fronts N
            # platforms and the connector advertises the bit per platform at
            # handshake — the scalar attr only carries the PRIMARY identity's
            # bit. Native adapters (no per-platform query) keep the class
            # attribute path unchanged.
            per_platform_check = getattr(
                runtime_adapter, "supports_inchannel_continuable_for_platform",
                None,
            )
            if callable(per_platform_check):
                try:
                    surface_supported = bool(per_platform_check(platform_name))
                except Exception:
                    surface_supported = False
            else:
                surface_supported = bool(getattr(
                    runtime_adapter, "supports_inchannel_continuable", False
                ))
            if not surface_supported:
                # Fail safe (D6): platform has no in_channel continuation
                # primitive.
                logger.debug(
                    "Job '%s': cron_continuable_surface=in_channel not supported on "
                    "%s, using thread",
                    job.get("id", "?"), platform_name,
                )
                in_channel_surface = False

        if in_channel_surface and inchannel_continuable and live_adapter_ready:
            # Force flat delivery (D2): the continuable-channel target must
            # ignore any inherited origin/target thread_id, or the flat
            # continuable session seeded below (thread_id=None, via
            # _seed_cron_channel_session) never matches where the brief is
            # actually delivered — route_thread_id further down in this loop
            # reads `thread_id` and would otherwise route into the origin
            # thread instead of flat into the channel.
            #
            # Gated on `inchannel_continuable` (the SAME gate as the seed
            # below), NOT `mirror_this_target` alone: for origin targets the
            # seed fires on origin-match alone (in_channel is the
            # continuation surface, independent of the attach_to_session /
            # mirror opt-in), so the flatten must use the SAME gate — with
            # the default knobs off, a mirror-gated flatten kept delivering
            # into the origin thread while the flat session got seeded,
            # leaving the brief and its continuation surface in different
            # places.
            # Gated on `live_adapter_ready` (adapter present AND a running loop)
            # so the clear fires ONLY on the live-send path that actually seeds
            # the flat session — the SAME condition as the live-send block
            # below. `runtime_adapter is not None` alone is broader than that
            # path: an adapter can be present while the event loop is absent or
            # not running, in which case the live-send/seed block is skipped and
            # delivery falls through to the standalone path. Clearing thread_id
            # there would flatten a brief into a channel with NO seeded
            # continuable session behind it (and bypass the D6 capability
            # check), so the standalone fallback must keep the origin thread
            # (review r3609147550).
            #
            # Fan-out / broadcast / explicit-thread targets keep their thread_id
            # (they are not continuable and are never seeded). Placed AFTER
            # mirror_this_target / origin_user_id are computed above — those
            # need the ORIGINAL thread_id to match the origin conversation.
            thread_id = None

        # For an in_channel delivery the flat continuation session is created
        # explicitly below (the shipped mirror only APPENDS to an existing
        # session, and the flat channel row is otherwise absent for a
        # chat_postMessage delivery). ``is_dm_target`` (computed above with
        # origin_user_id) selects the session chat_type so the seeded key
        # matches the inbound reply's key. ``inchannel_seeded`` suppresses the
        # generic mirror below so the brief is not double-written.
        inchannel_seeded = False

        # Continuable cron (thread-preferred): when mirroring is enabled for the
        # origin target and the gateway is live, try to open a DEDICATED thread
        # for this job and deliver the brief into it. On thread-capable
        # platforms (Telegram/Discord/Slack) the brief + the user's replies live
        # in their own scrollback; the thread-keyed session is seeded so a reply
        # continues with full context. On DM-only platforms (WhatsApp/Signal)
        # create_handoff_thread returns None and we fall back to mirroring into
        # the origin DM session (handled after delivery). Cf. _process_handoff.
        #
        # in_channel surface (D2): SKIP thread creation entirely — leave
        # thread_id=None so the delivery posts flat, then
        # ``_seed_cron_channel_session`` (below) CREATES the shared-channel
        # session and mirrors the brief into it. The shipped mirror alone is
        # NOT enough here: ``mirror_to_session`` only APPENDS to an existing
        # session and a flat ``(platform, chat_id, None)`` row is otherwise
        # absent for a ``chat_postMessage`` delivery, so the seed must create
        # the row first (F5).
        thread_seeded = False
        opened_thread_id: Optional[str] = None
        if (
            mirror_this_target
            and not in_channel_surface
            and runtime_adapter is not None
            and loop is not None
            and not thread_id  # never override an explicit origin thread/topic
        ):
            new_thread_id = _open_continuable_cron_thread(
                job, runtime_adapter, chat_id, loop,
            )
            if new_thread_id:
                # Route THIS delivery into the new thread now (the send needs the
                # thread_id), but defer seeding the thread session until the
                # delivery actually succeeds — otherwise an open-succeeds /
                # deliver-fails case leaves a seeded brief the user never saw,
                # and (worse) suppresses the DM-fallback mirror via thread_seeded.
                thread_id = new_thread_id
                opened_thread_id = new_thread_id

        if live_adapter_ready:
            # Telegram topic routing (#22773, regression fixed #52060): a
            # ``telegram:<positive_chat_id>:<numeric_thread_id>`` cron target is
            # ambiguous — a forum-style topic in a private chat and a genuine
            # Bot API channel Direct-Messages topic share the same shape and
            # need OPPOSITE routing. Disambiguate at delivery time via
            # ``_is_channel_dm_topic`` (see its docstring for the full
            # rationale); ``thread_id`` goes in ``route_metadata`` so the
            # anchorless cron send bypasses the DeliveryRouter's private-chat
            # reply-anchor requirement. Compute the routed metadata ONCE so both
            # the text send (via DeliveryRouter) and the media send agree.
            from gateway.delivery import (
                DeliveryRouter,
                DeliveryTarget,
                _looks_like_int,
                looks_like_telegram_private_chat_id,
            )

            is_ambiguous_telegram_topic = (
                platform == Platform.TELEGRAM
                and thread_id is not None
                and looks_like_telegram_private_chat_id(str(chat_id))
                and _looks_like_int(str(thread_id))
            )
            route_via_dm_topic = is_ambiguous_telegram_topic and _is_channel_dm_topic(
                runtime_adapter, chat_id, loop, job["id"],
            )
            if route_via_dm_topic:
                # Genuine Bot API channel Direct-Messages topic (#22773 mode 2):
                # routed via direct_messages_topic_id, no bare thread_id.
                route_thread_id = None
                route_metadata = {
                    "direct_messages_topic_id": str(thread_id),
                    "job_id": job["id"],
                    "notify": notify_delivery,
                }
                # Media metadata mirrors the text routing so attachments land in
                # the same DM topic instead of the General lane (#22773).
                media_metadata = {
                    "direct_messages_topic_id": str(thread_id),
                    "notify": notify_delivery,
                }
            else:
                # Forum-style topic (private chat / supergroup) or non-topic
                # target: route via message_thread_id (#52060).  Put thread_id in
                # *route_metadata* (not just the DeliveryTarget) deliberately —
                # the DeliveryRouter's private-chat topic detection
                # (gateway/delivery.py) demands a reply anchor when thread_id is
                # absent from metadata; cron deliveries have no inbound reply
                # anchor, so the metadata key bypasses that check and lets the
                # adapter route via a plain message_thread_id.
                route_thread_id = str(thread_id) if thread_id is not None else None
                route_metadata = {"job_id": job["id"], "notify": notify_delivery}
                if route_thread_id:
                    route_metadata["thread_id"] = route_thread_id
                media_metadata = {"notify": notify_delivery}
                if thread_id:
                    media_metadata["thread_id"] = thread_id

            # Relay egress needs a tenant discriminator on the frame: the
            # connector's fail-closed guard resolves the workspace/guild from
            # metadata.scope_id, and after a gateway restart the RelayAdapter's
            # per-chat scope cache is COLD (learned only from inbound), while
            # DeliveryRouter stamps scope only for the configured HOME channel
            # (gateway/delivery.py). A scoped origin that is not the home chat
            # therefore egressed with no scope_id at all and could be rejected
            # before delivery — the delivery-leg sibling of the seed-key scope
            # fix. Origin-matching targets only: a fan-out/broadcast target's
            # tenant is NOT the origin's, and stamping the wrong scope is worse
            # than none (the router/home path handles fan-out home targets).
            if origin_target and origin.get("scope_id"):
                route_metadata.setdefault("scope_id", str(origin["scope_id"]))
                media_metadata = dict(media_metadata or {})
                media_metadata.setdefault("scope_id", str(origin["scope_id"]))

            # Provider routing may create, flatten, or fall back from a thread
            # after the global receipt plan was durably registered. Preserve
            # the logical requested identity separately; adapters record the
            # routed destination as actual_target.
            route_metadata["_transport_receipt_requested_target"] = receipt_requested_target

            try:
                # Send cleaned text (MEDIA tags stripped) — not the raw content.
                # Route through the gateway's DeliveryRouter so the live send
                # gets the same platform-specific routing as live messages —
                # in particular Telegram's three-mode topic routing.  The
                # standalone cron path lacked this, so DM-topic cron deliveries
                # landed in the General topic or were rejected by Bot API 10.0
                # (#22773).
                text_to_send = cleaned_delivery_content.strip()
                adapter_ok = True
                timed_out = False
                delivered_message_id = None
                send_result = None
                send_receipts = ()
                if not text_to_send and not media_files:
                    # Nothing to hand the adapter at all.  This used to fall
                    # straight through to the `if adapter_ok:` branch below and
                    # log "delivered to <chat> via live adapter" for a send that
                    # never happened (#77763).  Fail closed so the run reports
                    # the empty payload instead.
                    msg = (
                        f"live adapter send skipped (empty text and no media) "
                        f"for {platform_name}:{chat_id}"
                    )
                    logger.warning("Job '%s': %s", job["id"], msg)
                    target_errors.append(msg)
                    adapter_ok = False
                elif text_to_send:
                    from agent.async_utils import safe_schedule_threadsafe

                    router = DeliveryRouter(config, target_adapters)
                    route_target = DeliveryTarget(
                        platform=platform,
                        chat_id=str(chat_id),
                        thread_id=route_thread_id,
                        is_explicit=True,
                    )
                    # Pass thread routing via the target (not a bare metadata
                    # "thread_id"): the router only applies its Telegram DM-topic
                    # detection when "thread_id"/"message_thread_id" are absent
                    # from metadata, deriving the routing from target.thread_id
                    # or the explicit direct_messages_topic_id above.
                    future = safe_schedule_threadsafe(
                        router._deliver_to_platform(
                            route_target,
                            text_to_send,
                            route_metadata,
                        ),
                        loop,
                    )
                    if future is None:
                        adapter_ok = False
                        target_errors.append("live adapter event loop scheduling failed")
                    else:
                        send_result = None
                        timeout_handled = False
                        try:
                            send_result = future.result(timeout=60)
                        except TimeoutError:
                            # Cancellation only describes the local Future's
                            # state. It is neither an acknowledgement from the
                            # provider nor proof that no request crossed the
                            # wire. Conservatively classify either result as
                            # unknown and prohibit same-identity fallback.
                            future.cancel()
                            timed_out = True
                            timeout_handled = True
                            ambiguous_live_timeout = True
                            adapter_ok = False
                            msg = (
                                f"live adapter confirmation timed out for "
                                f"{platform_name}:{chat_id}; delivery is unknown"
                            )
                            target_errors.append(msg)
                            logger.warning("Job '%s': %s", job["id"], msg)
                        except Exception as ex:
                            target_errors.append(f"live adapter send failed: {ex}")
                            # Exceptions do not prove a provider request was not
                            # dispatched. Keep this target unknown and prohibit
                            # same-identity fallback.
                            ambiguous_live_timeout = True
                            partial_result = getattr(ex, "send_result", None)
                            if partial_result is None:
                                raise
                            # DeliveryRouter preserves a failed SendResult when
                            # it contains provider acknowledgements for earlier
                            # chunks. Skip success normalization but retain those
                            # receipts below.
                            send_result = partial_result
                            if type(partial_result) is SendResult:
                                send_receipts = partial_result.receipts
                            adapter_ok = False
                            timeout_handled = True

                        if timeout_handled:
                            # The timeout branch above already decided the
                            # outcome (assume-delivered if in flight, or
                            # adapter_ok=False to fall through if never
                            # dispatched).  send_result is None, so skip the
                            # confirmation/thread-fallback inspection below.
                            pass
                        else:
                            # _deliver_to_platform returns either a SendResult
                            # (.success attr) or, when the silence-narration
                            # filter drops the message, a plain dict
                            # {"success": True, "delivered": False, ...}.
                            # Normalize both shapes so a getattr default doesn't
                            # misread a dict, and so a None / success-less object
                            # Normalize only inert/known result containers. A
                            # filtered dict with delivered=False is not a delivery;
                            # a successful send without provider evidence remains
                            # explicitly UNVERIFIED.
                            send_receipts = ()
                            legacy_fields = None
                            send_raw_response = None
                            delivered_message_id = None
                            if type(send_result) is dict:
                                raw_response_value = send_result.get("raw_response")
                                send_raw_response = (
                                    raw_response_value
                                    if type(raw_response_value) is dict else None
                                )
                                message_id_value = send_result.get("message_id")
                                delivered_message_id = (
                                    message_id_value
                                    if type(message_id_value) in {str, int} else None
                                )
                            elif type(send_result) is SendResult:
                                send_raw_response = send_result.raw_response
                                delivered_message_id = send_result.message_id
                                send_receipts = send_result.receipts
                            else:
                                legacy_fields = _inert_legacy_send_result_fields(send_result)
                                if legacy_fields is not None:
                                    send_raw_response = legacy_fields["raw_response"]
                                    delivered_message_id = legacy_fields["message_id"]

                            _evidence_gap: list = []
                            send_success = _confirm_adapter_delivery(
                                send_result, job["id"], _evidence_gap,
                            )
                            if send_success and _evidence_gap:
                                unverified_targets.append(f"{platform_name}:{chat_id}")

                            if not send_success:
                                if type(send_result) is dict:
                                    error_value = send_result.get("error")
                                    filtered_value = send_result.get("filtered")
                                    err = (
                                        error_value
                                        if type(error_value) is str
                                        else filtered_value
                                        if type(filtered_value) is str
                                        else "unknown"

                                    )
                                    shape = "dict"
                                elif type(send_result) is SendResult:
                                    err = send_result.error
                                    shape = "SendResult"
                                elif legacy_fields is not None:
                                    err = legacy_fields["error"] or "unknown"
                                    shape = "legacy"
                                elif send_result is not None:
                                    err = "invalid adapter result"
                                    shape = "invalid"
                                else:
                                    err = "no response from adapter"
                                    shape = "None"
                                msg = (
                                    f"live adapter send to {platform_name}:{chat_id} "
                                    f"returned unconfirmed result ({shape}, error={err})"
                                )
                                if transport is not None and transport.is_relay:
                                    logger.warning("Job '%s': %s", job["id"], msg)
                                else:
                                    logger.warning(
                                        "Job '%s': %s, falling back to standalone",
                                        job["id"], msg,
                                    )
                                target_errors.append(msg)
                                # A negative legacy result does not prove the
                                # request never crossed the provider boundary.
                                # Preserve any earlier typed chunk receipts and
                                # never blind-resend this execution identity.
                                ambiguous_live_timeout = True
                                adapter_ok = False
                            elif not send_receipts and (
                                receipt_attempts or type(send_result) is SendResult
                            ):
                                # ``success``/``message_id`` are legacy operation
                                # fields, not provider acknowledgement evidence.
                                # A same-identity retry could duplicate a write
                                # which completed before an old adapter returned.
                                ambiguous_live_timeout = True
                                adapter_ok = False
                                msg = (
                                    f"live adapter send to {platform_name}:{chat_id} "
                                    "returned legacy success without typed receipt; "
                                    "delivery is unknown"
                                )
                                target_errors.append(msg)
                                logger.warning("Job '%s': %s", job["id"], msg)
                            elif (
                                send_raw_response
                                and thread_id
                                and send_raw_response.get("thread_fallback")
                            ):
                                requested_thread_id = send_raw_response.get("requested_thread_id") or thread_id
                                msg = (
                                    f"configured thread_id {requested_thread_id} for "
                                    f"{platform_name}:{chat_id} was not found; delivered without thread_id"
                                )
                                logger.warning("Job '%s': %s", job["id"], msg)
                                delivery_errors.append(msg)

                # A typed acknowledgement must be committed before any follow-up
                # send, fallback, mirror, or seed. A database error leaves its
                # preregistered attempt unknown and makes retry unsafe.
                if text_to_send and receipt_attempts and send_result is not None:
                    persisted_all = _persist_target_text_receipts(
                        send_receipts,
                        receipt_attempts,
                        receipt_requested_target,
                        components={"text"},
                        expected_actual_target={
                            "platform": platform_name,
                            "chat_id": chat_id,
                            "thread_id": (
                                str(route_metadata["direct_messages_topic_id"])
                                if route_metadata.get("direct_messages_topic_id") is not None
                                else route_thread_id or ""
                            ),
                        },
                    )
                    if not persisted_all:
                        ambiguous_live_timeout = True
                        adapter_ok = False
                        target_errors.append(
                            f"live adapter acknowledgement for {platform_name}:{chat_id} could not be persisted; delivery is unknown"
                        )

                # Send extracted media files as native attachments via the live
                # adapter, using the same DM-topic-aware routing as the text send
                # (#22773 — media previously used a bare thread_id and landed in
                # the General lane for private DM topics).  Skip on an in-flight
                # confirmation timeout: the gateway loop is contended, so each
                # media send would also block its 30s budget, and the text
                # payload is already assumed delivered (#38922).  Record the
                # skipped attachments so the drop is visible rather than silently
                # lost.
                _media_receipts = []
                _media_errors = []
                if adapter_ok and not timed_out and media_files:
                    routed_media_metadata = dict(media_metadata or {})
                    if transport is not None and transport.is_relay:
                        routed_media_metadata["_relay_logical_platform"] = platform.value
                        logical_home = config.get_home_channel(platform)
                        if logical_home is not None and logical_home.chat_id == chat_id:
                            if logical_home.user_id:
                                routed_media_metadata["user_id"] = logical_home.user_id
                            if logical_home.scope_id:
                                routed_media_metadata["scope_id"] = logical_home.scope_id
                    routed_media_metadata["_transport_receipt_requested_target"] = (
                        receipt_requested_target
                    )
                    _media_errors = _send_media_via_adapter(
                        runtime_adapter,
                        chat_id,
                        media_files,
                        routed_media_metadata or None,
                        loop,
                        job,
                        platform=platform,
                        receipts_out=_media_receipts,
                    )
                    # Surface per-file failures into the run status (parity
                    # with the standalone lane): text delivered but an
                    # attachment didn't is a visible partial failure, not ok.
                    for _me in _media_errors:
                        _msg = f"{_me} (target {platform_name}:{chat_id})"
                        delivery_errors.append(_msg)
                elif timed_out and media_files:
                    msg = (
                        f"{len(media_files)} media attachment(s) not delivered to "
                        f"{platform_name}:{chat_id} (live adapter confirmation timed out)"
                    )
                    logger.warning("Job '%s': %s", job["id"], msg)
                    delivery_errors.append(msg)

                media_receipts_persisted = bool(media_files) and _persist_target_text_receipts(
                    tuple(_media_receipts) if not timed_out else (),
                    receipt_attempts,
                    receipt_requested_target,
                    components={"media"},
                    expected_actual_target={
                        "platform": platform_name,
                        "chat_id": chat_id,
                        "thread_id": (
                            str(route_metadata["direct_messages_topic_id"])
                            if route_metadata.get("direct_messages_topic_id") is not None
                            else route_thread_id or ""
                        ),
                    },
                )
                if _media_errors:
                    # A failed SendResult does not prove that the provider did
                    # not accept the media. Preserve any receipts above, but
                    # never retry the whole target through the standalone lane:
                    # that could duplicate text or attachments after an
                    # ambiguous live-adapter write.
                    ambiguous_live_timeout = True
                    adapter_ok = False
                if media_files and not media_receipts_persisted:
                    # Preserve any partial typed acknowledgements, but keep the
                    # target unknown unless every planned media component was
                    # confirmed and persisted.
                    ambiguous_live_timeout = True
                    adapter_ok = False
                    target_errors.append(
                        f"media acknowledgement for {platform_name}:{chat_id} is unavailable; delivery is partial"
                    )

                if adapter_ok:
                    # Log WHERE it went, not just that it went: a ghost delivery
                    # that landed in the wrong lane (General topic instead of the
                    # routed thread) is indistinguishable from a real one without
                    # the routing identity (#77763).
                    logger.info(
                        "Job '%s': delivered to %s:%s via live adapter thread=%s message_id=%s",
                        job["id"], platform_name, chat_id,
                        route_thread_id if route_thread_id is not None else "-",
                        delivered_message_id if delivered_message_id is not None else "-",
                    )
                    delivered = True
                    # Seed the thread session only now that delivery into it
                    # succeeded (deferred from thread-open above).
                    if opened_thread_id and not thread_seeded:
                        _seed_cron_thread_session(
                            job, runtime_adapter, platform_name, chat_id,
                            opened_thread_id, mirror_text,
                            chat_name=origin.get("chat_name"),
                            is_dm=is_dm_target,
                            scope_id=origin.get("scope_id"),
                        )
                        thread_seeded = True
                    # in_channel surface: CREATE + seed the flat channel/DM
                    # session (the shipped mirror only appends to an existing
                    # session — the flat row is otherwise absent for a
                    # chat_postMessage delivery, so the brief would be lost).
                    # Gated on `inchannel_continuable` — the SHARED gate with
                    # the thread-flatten above (they must not drift, or the
                    # brief and its continuation session land in different
                    # places). Origin targets seed without requiring the
                    # mirror opt-in: in_channel IS the continuation surface —
                    # a continuable flat cron without its seed is a brief the
                    # next reply can't see (the bug Victor hit live
                    # 2026-08-19: agent had "no idea about the delivery
                    # message"). Mirror-eligible NON-origin targets
                    # (origin_fallback / opted-in explicit — see
                    # _target_mirror_eligible) also seed, guarded by
                    # _inchannel_seed_allowed inside the gate: group-channel
                    # keys are user-isolated, so a seed without a user_id
                    # (origin-less managed cron into a shared channel) would
                    # create an orphan session no reply resolves to — those
                    # fall back to the plain mirror instead.
                    if in_channel_surface and inchannel_continuable and not thread_seeded:
                        inchannel_seeded = _seed_cron_channel_session(
                            job, runtime_adapter, platform_name, chat_id,
                            mirror_text, is_dm=is_dm_target,
                            user_id=origin_user_id,
                            chat_name=origin.get("chat_name"),
                            scope_id=origin.get("scope_id"),
                        )
                        if not inchannel_seeded:
                            logger.warning(
                                "Job '%s': in_channel seed did NOT land on %s:%s "
                                "— a plain reply will not see this brief",
                                job["id"], platform_name, chat_id,
                            )
                        # Companion THREAD-surface seed (live gap, Alice
                        # 2026-08-19): a flat brief is still a Slack message
                        # the user can reply to IN ITS THREAD — the natural
                        # mobile/desktop affordance — and that reply keys to
                        # (chat, thread=<brief ts>), a session the flat seed
                        # never touches. Seed it too so BOTH reply surfaces
                        # continue the job. Uses the delivered message id as
                        # the thread anchor; best-effort like every seed.
                        if delivered_message_id:
                            _seed_cron_thread_session(
                                job, runtime_adapter, platform_name, chat_id,
                                str(delivered_message_id), mirror_text,
                                chat_name=origin.get("chat_name"),
                                is_dm=is_dm_target,
                                scope_id=origin.get("scope_id"),
                            )
                    elif in_channel_surface and not inchannel_continuable:
                        logger.warning(
                            "Job '%s': in_channel delivery to %s:%s is not a "
                            "continuable target (origin=%s:%s thread=%s; not the "
                            "origin conversation, and not a mirror-eligible "
                            "fallback/opted-in target the seed can key) — seed "
                            "skipped; the plain mirror below may still apply",
                            job["id"], platform_name, chat_id,
                            origin.get("platform"), origin.get("chat_id"),
                            origin.get("thread_id"),
                        )
                    _maybe_mirror_cron_delivery(
                        job, platform_name, chat_id, mirror_text,
                        thread_id=thread_id, user_id=origin_user_id,
                        enabled=mirror_this_target and not thread_seeded and not inchannel_seeded,
                    )
            except Exception as e:
                err_msg = f"live adapter delivery to {platform_name}:{chat_id} failed: {e}"
                if not any(err_msg in err for err in target_errors):
                    target_errors.append(err_msg)
                if transport is not None and transport.is_relay:
                    logger.warning("Job '%s': %s", job["id"], err_msg)
                else:
                    logger.warning(
                        "Job '%s': %s, falling back to standalone",
                        job["id"], err_msg,
                    )

        if ambiguous_live_timeout:
            # No standalone retry, mirror, or seed after an ambiguous live
            # send: any of them could duplicate an unconfirmed provider write.
            delivery_errors.extend(target_errors)
            continue

        if not delivered:
            if transport is not None and transport.is_relay:
                # Relay owns the logical destination and its connector owns the
                # platform credential. A native retry could duplicate delivery
                # and cannot be authenticated correctly, so fail closed.
                if not target_errors:
                    target_errors.append(
                        f"relay delivery to {platform_name}:{chat_id} failed"
                    )
                delivery_errors.extend(target_errors)
                continue
            # If the interpreter is finalizing (gateway SIGTERM / restart /
            # OOM), scheduling any new delivery is futile — asyncio.run and a
            # fresh ThreadPoolExecutor both raise "cannot schedule new futures
            # after interpreter shutdown". Skip gracefully with a warning
            # rather than emitting an ERROR traceback on every restart-race
            # (#58720, #55924).
            if _interpreter_shutting_down():
                msg = f"delivery to {platform_name}:{chat_id} skipped — interpreter is shutting down"
                logger.warning("Job '%s': %s", job["id"], msg)
                target_errors.append(msg)
                delivery_errors.extend(target_errors)
                continue
            # The live lane already failed closed on an empty payload; the
            # standalone senders do not. The Telegram adapter returns
            # SendResult(success=True) for empty content WITHOUT an API call,
            # so falling through here turns a phantom live delivery into a
            # phantom standalone one and logs it as delivered (#77763). Both
            # _send_to_platform call sites below are reached through this
            # point, so one guard closes the lane.
            if not cleaned_delivery_content.strip() and not media_files:
                msg = (
                    f"standalone send skipped (empty text and no media) "
                    f"for {platform_name}:{chat_id}"
                )
                logger.warning("Job '%s': %s", job["id"], msg)
                target_errors.append(msg)
                delivery_errors.extend(target_errors)
                continue
            # Standalone path: run the async send in a fresh event loop (safe from any thread)
            coro = _send_to_platform(
                platform, pconfig, chat_id, cleaned_delivery_content,
                thread_id=thread_id, media_files=media_files,
                receipt_bound=bool(receipt_attempts),
            )
            try:
                result = asyncio.run(coro)
            except RuntimeError as run_err:
                # asyncio.run() checks for a running loop before awaiting the coroutine;
                # when it raises, the original coro was never started — close it to
                # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
                # fresh thread that has no running loop.
                coro.close()
                # If the RuntimeError is the interpreter-finalization signal,
                # the fresh-thread fallback would fail identically — skip
                # gracefully instead of logging a shutdown-race traceback.
                if _interpreter_shutting_down(run_err):
                    msg = f"delivery to {platform_name}:{chat_id} skipped — interpreter is shutting down"
                    logger.warning("Job '%s': %s", job["id"], msg)
                    target_errors.append(msg)
                    delivery_errors.extend(target_errors)
                    continue
                # The thread-pool fallback can itself raise (SMTP ConnectionError,
                # future.result timeout, etc.). An exception raised inside this
                # `except RuntimeError` block is NOT caught by the sibling
                # `except Exception` below — it would escape _deliver_result()
                # and crash the whole delivery loop, silently skipping every
                # remaining target (#47163). Wrap the fallback in its own
                # try/except so a per-target failure is logged and the loop
                # continues to the next target.
                try:
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        def _run_standalone_send():
                            return asyncio.run(_send_to_platform(
                                platform, pconfig, chat_id, cleaned_delivery_content,
                                thread_id=thread_id, media_files=media_files,
                                receipt_bound=bool(receipt_attempts),
                            ))

                        # The fallback worker is a fresh thread: it does NOT
                        # inherit the multiplexed profile ContextVars (home
                        # override + secret scope). Run inside a copy of the
                        # active context so the standalone sender reads THIS
                        # profile's bot token, not the process default's
                        # (#100489) — same pattern as the session-db and
                        # heartbeat workers in this module.
                        _fallback_context = contextvars.copy_context()
                        future = pool.submit(
                            _fallback_context.run,
                            _run_standalone_send,
                        )
                        result = future.result(timeout=30)
                    finally:
                        pool.shutdown(wait=False)
                except Exception as e:
                    # A shutdown-race here is expected during teardown; downgrade
                    # to a warning so it doesn't read as a genuine failure.
                    if _interpreter_shutting_down(e):
                        msg = f"delivery to {platform_name}:{chat_id} skipped — interpreter is shutting down"
                        logger.warning("Job '%s': %s", job["id"], msg)
                        target_errors.append(msg)
                        delivery_errors.extend(target_errors)
                        continue
                    msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
                    logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
                    target_errors.extend([msg])
                    delivery_errors.extend(target_errors)
                    continue
            except Exception as e:
                msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
                logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
                target_errors.extend([msg])
                delivery_errors.extend(target_errors)
                continue

            if receipt_attempts:
                standalone_receipts = (
                    result.get("receipts", ()) if isinstance(result, dict) else ()
                )
                if not _persist_target_text_receipts(
                    standalone_receipts, receipt_attempts, receipt_requested_target,
                ):
                    if media_files:
                        msg = (
                            f"media acknowledgement for {platform_name}:{chat_id} "
                            "is unavailable; delivery is partial"
                        )
                    else:
                        msg = (
                            f"standalone send to {platform_name}:{chat_id} returned "
                            "without a complete typed receipt; delivery is unknown"
                        )
                    target_errors.append(msg)
                    delivery_errors.extend(target_errors)
                    continue

            if result and result.get("error"):
                # Include target context (platform/chat) so a bare error string
                # like "Discord send failed: TimeoutError: " is attributable.
                # Not inside an except block — the error comes from the send
                # result dict, so there is no traceback to attach.
                msg = f"delivery error: {result['error']} (target {platform_name}:{chat_id})"
                logger.error("Job '%s': %s", job["id"], msg)
                target_errors.extend([msg])
                delivery_errors.extend(target_errors)
                continue

            # Standalone senders report per-file attachment failures in
            # ``warnings`` while still returning success (the text leg
            # delivered). Surface them: a cron whose PDF/image silently
            # vanished used to mark the run ok with no trace — the exact
            # "manual run delivers text but no attachment" field report.
            _sender_warnings = (
                result.get("warnings") if isinstance(result, dict) else None
            ) or []
            for _w in _sender_warnings:
                msg = f"delivery warning: {_w} (target {platform_name}:{chat_id})"
                logger.error("Job '%s': %s", job["id"], msg)
                delivery_errors.append(msg)

            logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
            _maybe_mirror_cron_delivery(
                job, platform_name, chat_id, mirror_text,
                thread_id=thread_id, user_id=origin_user_id,
                enabled=mirror_this_target and not thread_seeded,
            )

    if policy_drop_errors:
        # Filter-time drops apply to every target; report them once.
        delivery_errors.extend(policy_drop_errors)
    _record_delivery_verification(job, unverified_targets)
    if delivery_errors:
        return "; ".join(delivery_errors)
    return None



_DEFAULT_MEDIA_SEND_TIMEOUT = 300


def _get_media_send_timeout() -> int:
    """Resolve the per-attachment media-send timeout from env/config.

    Mirrors the ``script_timeout_seconds`` resolution pattern: the
    HERMES_CRON_MEDIA_SEND_TIMEOUT env var wins, then
    ``cron.media_send_timeout_seconds`` in config.yaml, then the default
    (300s — large attachments like long TTS audio can legitimately exceed
    the old fixed 30s upload window).
    """
    env_value = os.getenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning(
                "Invalid HERMES_CRON_MEDIA_SEND_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("media_send_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron media-send timeout from config: %s", exc)

    return _DEFAULT_MEDIA_SEND_TIMEOUT

# Late-bound scheduler siblings avoid import cycles and preserve direct-module test seams.
from cron import scheduler as _sched  # noqa: E402
from cron import scheduler_preflight as _preflight  # noqa: E402
from cron import scheduler_script as _script  # noqa: E402
SharedRouteAdapters = _preflight.SharedRouteAdapters

def _interpreter_shutting_down(exc=None):
    return _sched._interpreter_shutting_down(exc)
