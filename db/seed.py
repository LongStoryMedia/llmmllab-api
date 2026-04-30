"""Seed the local dev database with a test user and API key.

Runs when TEST_USER_ID is set. Idempotent: skips if the test user
and API key already exist.
Writes credentials to .env.local so they survive restarts.
"""

import hashlib
import secrets
from pathlib import Path

from sqlalchemy import text

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="db_seed")

_TEST_API_KEY_NAME = "local-dev-key"
_ENV_LOCAL_PATH = Path(__file__).resolve().parent.parent / ".env.local"


async def seed_test_user_and_api_key(session_factory, user_id: str):
    """Create a test user and API key if they don't already exist.

    Returns the plaintext API key (either existing or newly created).
    """
    async with session_factory() as session:
        # Ensure the test user exists
        result = await session.execute(
            text("SELECT id FROM users WHERE id = :uid"), {"uid": user_id}
        )
        if result.fetchone() is None:
            await session.execute(
                text("INSERT INTO users (id) VALUES (:uid)"), {"uid": user_id}
            )
            logger.info(f"Created test user: {user_id}")

        # Check if a test API key already exists for this user
        result = await session.execute(
            text(
                "SELECT key_hash FROM api_keys "
                "WHERE user_id = :uid AND name = :name AND NOT is_revoked"
            ),
            {"uid": user_id, "name": _TEST_API_KEY_NAME},
        )
        existing_key = result.fetchone()

        if existing_key is not None:
            logger.info(f"Test API key already exists for user {user_id}")
            api_key = _read_api_key_from_env_local()
            if api_key:
                logger.info(f"Read existing API key from {_ENV_LOCAL_PATH}")
                return api_key
            logger.warning(
                "Test API key exists in DB but not in .env.local. "
                "Create a new one via POST /api-keys/create or delete the existing key."
            )
            return None

        # Generate and insert a new API key
        plaintext_key = secrets.token_hex(32)
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

        result = await session.execute(
            text(
                "INSERT INTO api_keys (user_id, key_hash, name, scopes)"
                " VALUES (:uid, :key_hash, :name, ARRAY[]::text[])"
                " RETURNING id"
            ),
            {
                "uid": user_id,
                "key_hash": key_hash,
                "name": _TEST_API_KEY_NAME,
            },
        )
        new_key_id = str(result.mappings().first()["id"])
        await session.commit()

        logger.info(f"Created test API key {new_key_id} for user {user_id}")
        _write_api_key_to_env_local(plaintext_key)

        return plaintext_key


def _read_api_key_from_env_local() -> str | None:
    """Read TEST_API_KEY from .env.local if it exists."""
    if not _ENV_LOCAL_PATH.exists():
        return None
    for line in _ENV_LOCAL_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "TEST_API_KEY":
            return v.strip()
    return None


def _write_api_key_to_env_local(api_key: str):
    """Write or update TEST_API_KEY in .env.local."""
    content = ""
    if _ENV_LOCAL_PATH.exists():
        lines = _ENV_LOCAL_PATH.read_text().splitlines()
        lines = [l for l in lines if not l.strip().startswith("TEST_API_KEY")]
        content = "\n".join(lines) + "\n" if lines else ""

    content += f'\n# Local dev API key (auto-generated)\nTEST_API_KEY={api_key}\n'
    _ENV_LOCAL_PATH.write_text(content)
    logger.info(f"Wrote test API key to {_ENV_LOCAL_PATH}")
