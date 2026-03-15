#!/usr/bin/env python3
"""
Constantinople API Backend — User accounts, API keys, credits, and billing.

This service sits in front of the inference gateway and manages:
- User registration and authentication
- API key creation and management
- Credit balance tracking (per-token billing)
- Usage history and analytics
- Payment processing (crypto on Base — USDC and ETH, TAO on Bittensor)

It proxies inference requests to the gateway while enforcing credit limits.

Database: PostgreSQL (Supabase) via asyncpg, with SQLite fallback.

Usage:
    # Postgres (recommended):
    DATABASE_URL=postgresql://... python api_backend.py --port 8090 --gateway http://localhost:8081

    # SQLite fallback:
    python api_backend.py --port 8090 --gateway http://localhost:8081 --db ./api.db
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import urllib.request

import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

try:
    from bittensor import Keypair
    from bittensor import Subtensor
    HAS_BITTENSOR = True
except ImportError:
    HAS_BITTENSOR = False

log = logging.getLogger("api-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Pricing ─────────────────────────────────────────────────────────────────

# Credits are denominated in micro-units (1 credit = 1,000,000 micro-credits)
# This avoids floating point issues in the ledger
MICRO = 1_000_000

# Default pricing (can be overridden via CLI or env)
DEFAULT_INPUT_PRICE = 0.50   # credits per 1M input tokens
DEFAULT_OUTPUT_PRICE = 1.50  # credits per 1M output tokens
FREE_TIER_CREDITS = 1.0      # free credits on signup

# ─── Postgres Schema ─────────────────────────────────────────────────────────

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    balance_micro BIGINT NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    stripe_customer_id TEXT,
    tier TEXT NOT NULL DEFAULT 'free'
);

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    key_hash TEXT UNIQUE NOT NULL,
    key_prefix TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    rate_limit_rpm INTEGER NOT NULL DEFAULT 60
);

CREATE TABLE IF NOT EXISTS usage_log (
    id SERIAL PRIMARY KEY,
    api_key_id INTEGER NOT NULL REFERENCES api_keys(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    request_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_micro BIGINT NOT NULL DEFAULT 0,
    endpoint TEXT,
    status_code INTEGER,
    latency_ms DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    amount_micro BIGINT NOT NULL,
    type TEXT NOT NULL,
    description TEXT,
    reference_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS crypto_invoices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    invoice_id TEXT UNIQUE NOT NULL,
    amount_usd DOUBLE PRECISION NOT NULL,
    credits DOUBLE PRECISION NOT NULL,
    chain TEXT NOT NULL DEFAULT 'base',
    currency TEXT NOT NULL DEFAULT 'USDC',
    deposit_address TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    tx_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    paid_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS tao_deposit_addresses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),
    ss58_address TEXT UNIQUE NOT NULL,
    seed_hex TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_known_balance_rao BIGINT NOT NULL DEFAULT 0,
    total_credited_rao BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bittensor_accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    ss58_address TEXT UNIQUE NOT NULL,
    address_type TEXT NOT NULL DEFAULT 'hotkey',
    subnets_registered TEXT,
    tao_balance DOUBLE PRECISION NOT NULL DEFAULT 0,
    credits_granted DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS bittensor_nonces (
    ss58_address TEXT PRIMARY KEY,
    nonce TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_log_key ON usage_log(api_key_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_crypto_invoices_user ON crypto_invoices(user_id, status);
CREATE INDEX IF NOT EXISTS idx_crypto_invoices_id ON crypto_invoices(invoice_id);
CREATE INDEX IF NOT EXISTS idx_tao_deposit_address ON tao_deposit_addresses(ss58_address);
CREATE INDEX IF NOT EXISTS idx_bittensor_accounts_address ON bittensor_accounts(ss58_address);
CREATE INDEX IF NOT EXISTS idx_bittensor_accounts_user ON bittensor_accounts(user_id);
"""

# ─── Database abstraction ─────────────────────────────────────────────────────

class PostgresDB:
    """Async Postgres database layer using asyncpg connection pool."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        # Run schema creation
        async with self.pool.acquire() as conn:
            for statement in PG_SCHEMA.strip().split(';'):
                statement = statement.strip()
                if statement:
                    await conn.execute(statement)
        log.info("[DB] PostgreSQL connected and schema initialized")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def fetchone(self, query: str, *args) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchall(self, query: str, *args) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

    async def execute(self, query: str, *args):
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetchval(self, query: str, *args):
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute_returning_id(self, query: str, *args) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)


class SQLiteDB:
    """Sync SQLite database layer (fallback when DATABASE_URL not set)."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                balance_micro INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                stripe_customer_id TEXT,
                tier TEXT NOT NULL DEFAULT 'free'
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_used_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                rate_limit_rpm INTEGER NOT NULL DEFAULT 60
            );
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_id INTEGER NOT NULL REFERENCES api_keys(id),
                user_id INTEGER NOT NULL REFERENCES users(id),
                request_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                model TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost_micro INTEGER NOT NULL DEFAULT 0,
                endpoint TEXT,
                status_code INTEGER,
                latency_ms REAL
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                amount_micro INTEGER NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                reference_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS crypto_invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                invoice_id TEXT UNIQUE NOT NULL,
                amount_usd REAL NOT NULL,
                credits REAL NOT NULL,
                chain TEXT NOT NULL DEFAULT 'base',
                currency TEXT NOT NULL DEFAULT 'USDC',
                deposit_address TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                tx_hash TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT NOT NULL,
                paid_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tao_deposit_addresses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),
                ss58_address TEXT UNIQUE NOT NULL,
                seed_hex TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_known_balance_rao INTEGER NOT NULL DEFAULT 0,
                total_credited_rao INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS bittensor_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                ss58_address TEXT UNIQUE NOT NULL,
                address_type TEXT NOT NULL DEFAULT 'hotkey',
                subnets_registered TEXT,
                tao_balance REAL NOT NULL DEFAULT 0,
                credits_granted REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login_at TEXT
            );
            CREATE TABLE IF NOT EXISTS bittensor_nonces (
                ss58_address TEXT PRIMARY KEY,
                nonce TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_usage_log_key ON usage_log(api_key_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_crypto_invoices_user ON crypto_invoices(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_crypto_invoices_id ON crypto_invoices(invoice_id);
            CREATE INDEX IF NOT EXISTS idx_tao_deposit_address ON tao_deposit_addresses(ss58_address);
            CREATE INDEX IF NOT EXISTS idx_bittensor_accounts_address ON bittensor_accounts(ss58_address);
            CREATE INDEX IF NOT EXISTS idx_bittensor_accounts_user ON bittensor_accounts(user_id);
        """)
        self.conn.commit()

    async def connect(self):
        pass  # already connected in __init__

    async def close(self):
        self.conn.close()

    async def fetchone(self, query: str, *args) -> Optional[dict]:
        q, params = self._adapt(query, args)
        row = self.conn.execute(q, params).fetchone()
        return dict(row) if row else None

    async def fetchall(self, query: str, *args) -> list[dict]:
        q, params = self._adapt(query, args)
        rows = self.conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    async def execute(self, query: str, *args):
        q, params = self._adapt(query, args)
        self.conn.execute(q, params)
        self.conn.commit()

    async def fetchval(self, query: str, *args):
        q, params = self._adapt(query, args)
        row = self.conn.execute(q, params).fetchone()
        return row[0] if row else None

    async def execute_returning_id(self, query: str, *args) -> int:
        # Strip RETURNING clause for SQLite, use lastrowid
        q, params = self._adapt(query, args)
        # Remove RETURNING ... clause
        import re
        q_stripped = re.sub(r'\s+RETURNING\s+\w+\s*$', '', q, flags=re.IGNORECASE)
        cur = self.conn.execute(q_stripped, params)
        self.conn.commit()
        return cur.lastrowid

    @staticmethod
    def _adapt(query: str, args: tuple) -> tuple:
        """Convert $1, $2, ... style params to ? style for SQLite."""
        import re

        # First, handle INTERVAL expressions that reference params ($N)
        # e.g. NOW() - INTERVAL '1 day' * $2  ->  datetime('now', '-' || ? || ' days')
        def _replace_interval_param(m):
            return "datetime('now', '-' || ? || ' days')"

        converted = re.sub(
            r"datetime\('now'\)\s*-\s*INTERVAL\s+'1\s+day'\s*\*\s*\$(\d+)",
            _replace_interval_param,
            query,
            flags=re.IGNORECASE,
        )
        # Also handle static intervals: NOW() - INTERVAL '30 days'
        converted = re.sub(
            r"datetime\('now'\)\s*-\s*INTERVAL\s+'(\d+)\s+days?'",
            r"datetime('now', '-\1 days')",
            converted,
            flags=re.IGNORECASE,
        )

        # Replace NOW() with datetime('now')
        converted = converted.replace('NOW()', "datetime('now')")

        # Now handle intervals that were created after NOW() replacement
        converted = re.sub(
            r"datetime\('now'\)\s*-\s*INTERVAL\s+'1\s+day'\s*\*\s*\?",
            "datetime('now', '-' || ? || ' days')",
            converted,
            flags=re.IGNORECASE,
        )
        converted = re.sub(
            r"datetime\('now'\)\s*-\s*INTERVAL\s+'(\d+)\s+days?'",
            r"datetime('now', '-\1 days')",
            converted,
            flags=re.IGNORECASE,
        )

        # Replace BOOLEAN TRUE/FALSE
        converted = converted.replace(' TRUE', ' 1').replace(' FALSE', ' 0')

        # Replace CAST(x AS DATE) with date(x) for SQLite
        converted = re.sub(r'CAST\((\w+)\s+AS\s+DATE\)', r'date(\1)', converted, flags=re.IGNORECASE)

        # Replace $N with ? (in order)
        converted = re.sub(r'\$\d+', '?', converted)

        return converted, args


# ─── Auth helpers ────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password with salt using SHA-256. Simple but adequate for API keys."""
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}:{h}"

def verify_password(password: str, stored_hash: str) -> bool:
    salt, h = stored_hash.split(":", 1)
    return hmac.compare_digest(
        hashlib.sha256(f"{salt}:{password}".encode()).hexdigest(), h
    )

def generate_api_key() -> str:
    """Generate a prefixed API key: cst-<random hex>."""
    return f"cst-{secrets.token_hex(24)}"

def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

# ─── Request/Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class BittensorAuthRequest(BaseModel):
    ss58_address: str = Field(..., min_length=46, max_length=48, description="SS58-encoded hotkey or coldkey address")
    signature: str = Field(..., description="Hex-encoded signature of the message")
    message: str = Field(..., description="The signed message: '<ss58_address>:<timestamp>'")

class BittensorNonceRequest(BaseModel):
    ss58_address: str = Field(..., min_length=46, max_length=48)

class CreateKeyRequest(BaseModel):
    name: str = "default"

class TopUpRequest(BaseModel):
    amount: float = Field(..., gt=0, le=10000, description="Credits to add")
    payment_method: str = Field(default="crypto", description="crypto or stripe")
    currency: str = Field(default="USDC", description="USDC or ETH")

# ─── API Backend class ───────────────────────────────────────────────────────

class APIBackend:
    def __init__(self, db, gateway_url: str,
                 input_price: float = DEFAULT_INPUT_PRICE,
                 output_price: float = DEFAULT_OUTPUT_PRICE,
                 free_credits: float = FREE_TIER_CREDITS,
                 stripe_secret: Optional[str] = None,
                 stripe_webhook_secret: Optional[str] = None,
                 crypto_deposit_address: Optional[str] = None,
                 admin_key: Optional[str] = None):
        self.db = db
        self.gateway_url = gateway_url.rstrip("/")
        self.input_price_micro = int(input_price * MICRO / 1_000_000)   # micro-credits per token
        self.output_price_micro = int(output_price * MICRO / 1_000_000)
        self.free_credits_micro = int(free_credits * MICRO)
        self.stripe_secret = stripe_secret
        self.stripe_webhook_secret = stripe_webhook_secret
        self.crypto_deposit_address = crypto_deposit_address or ""
        self.admin_key = admin_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        await self.db.close()

    # ── User management ──

    async def create_user(self, email: str, password: str, name: Optional[str] = None) -> dict:
        pw_hash = hash_password(password)
        try:
            user_id = await self.db.execute_returning_id(
                "INSERT INTO users (email, password_hash, name, balance_micro) VALUES ($1, $2, $3, $4) RETURNING id",
                email, pw_hash, name, self.free_credits_micro
            )

            # Record signup bonus
            await self.db.execute(
                "INSERT INTO transactions (user_id, amount_micro, type, description) VALUES ($1, $2, 'signup_bonus', 'Free tier signup bonus')",
                user_id, self.free_credits_micro
            )

            return {"id": user_id, "email": email, "name": name, "balance": self.free_credits_micro / MICRO}
        except Exception as e:
            err_str = str(e).lower()
            if "unique" in err_str or "duplicate" in err_str or "integrity" in err_str:
                raise HTTPException(status_code=409, detail="Email already registered")
            raise

    async def authenticate_user(self, email: str, password: str) -> dict:
        row = await self.db.fetchone(
            "SELECT * FROM users WHERE email = $1 AND is_active = TRUE", email
        )
        if not row or not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return row

    async def get_user_by_api_key(self, api_key: str) -> Optional[dict]:
        """Look up user from API key. Returns user dict or None."""
        key_hash = hash_api_key(api_key)
        row = await self.db.fetchone("""
            SELECT u.*, ak.id as api_key_id, ak.rate_limit_rpm
            FROM api_keys ak
            JOIN users u ON u.id = ak.user_id
            WHERE ak.key_hash = $1 AND ak.is_active = TRUE AND u.is_active = TRUE
        """, key_hash)
        if row:
            # Update last used (fire and forget)
            await self.db.execute(
                "UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = $1", key_hash
            )
            return row
        return None

    # ── API key management ──

    async def create_api_key(self, user_id: int, name: str = "default") -> dict:
        count = await self.db.fetchval(
            "SELECT COUNT(*) FROM api_keys WHERE user_id = $1 AND is_active = TRUE AND name NOT LIKE 'session-%'",
            user_id
        )
        if count >= 5:
            raise HTTPException(status_code=400, detail="Maximum 5 active API keys per account")

        key = generate_api_key()
        key_hash = hash_api_key(key)
        key_prefix = key[:12] + "..."

        await self.db.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES ($1, $2, $3, $4)",
            user_id, key_hash, key_prefix, name
        )

        return {"key": key, "prefix": key_prefix, "name": name}

    async def list_api_keys(self, user_id: int) -> list:
        rows = await self.db.fetchall(
            "SELECT id, key_prefix, name, created_at, last_used_at, is_active FROM api_keys WHERE user_id = $1",
            user_id
        )
        # Serialize timestamps
        for r in rows:
            for k in ('created_at', 'last_used_at'):
                if isinstance(r.get(k), datetime):
                    r[k] = r[k].isoformat()
        return rows

    async def revoke_api_key(self, user_id: int, key_id: int) -> bool:
        result = await self.db.execute(
            "UPDATE api_keys SET is_active = FALSE WHERE id = $1 AND user_id = $2",
            key_id, user_id
        )
        # asyncpg returns "UPDATE N" string
        if isinstance(result, str):
            return not result.endswith("0")
        return True

    # ── Credits ──

    async def get_balance(self, user_id: int) -> float:
        val = await self.db.fetchval(
            "SELECT balance_micro FROM users WHERE id = $1", user_id
        )
        return (val / MICRO) if val else 0.0

    async def deduct_credits(self, user_id: int, api_key_id: int, request_id: str,
                              input_tokens: int, output_tokens: int,
                              model: str, endpoint: str, latency_ms: float, status_code: int) -> float:
        """Deduct credits for a request. Returns cost in credits."""
        cost_micro = (input_tokens * self.input_price_micro) + (output_tokens * self.output_price_micro)

        await self.db.execute(
            "UPDATE users SET balance_micro = balance_micro - $1 WHERE id = $2",
            cost_micro, user_id
        )
        await self.db.execute(
            """INSERT INTO usage_log (api_key_id, user_id, request_id, model,
               input_tokens, output_tokens, cost_micro, endpoint, status_code, latency_ms)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            api_key_id, user_id, request_id, model,
            input_tokens, output_tokens, cost_micro, endpoint, status_code, latency_ms
        )
        return cost_micro / MICRO

    async def add_credits(self, user_id: int, amount: float, tx_type: str,
                           description: str = "", reference_id: str = "") -> float:
        """Add credits to user balance. Returns new balance."""
        amount_micro = int(amount * MICRO)
        await self.db.execute(
            "UPDATE users SET balance_micro = balance_micro + $1 WHERE id = $2",
            amount_micro, user_id
        )
        await self.db.execute(
            "INSERT INTO transactions (user_id, amount_micro, type, description, reference_id) VALUES ($1, $2, $3, $4, $5)",
            user_id, amount_micro, tx_type, description, reference_id
        )
        val = await self.db.fetchval(
            "SELECT balance_micro FROM users WHERE id = $1", user_id
        )
        return val / MICRO

    # ── Usage ──

    async def get_usage(self, user_id: int, days: int = 30) -> dict:
        rows = await self.db.fetchall("""
            SELECT
                CAST(timestamp AS DATE) as date,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(cost_micro) as cost_micro,
                COUNT(*) as requests
            FROM usage_log
            WHERE user_id = $1 AND timestamp >= NOW() - INTERVAL '1 day' * $2
            GROUP BY CAST(timestamp AS DATE)
            ORDER BY date DESC
        """, user_id, days)

        daily = []
        for r in rows:
            d = dict(r)
            # Serialize date
            if hasattr(d.get('date'), 'isoformat'):
                d['date'] = d['date'].isoformat()
            d["cost"] = d["cost_micro"] / MICRO
            del d["cost_micro"]
            daily.append(d)

        total_input = sum(d["input_tokens"] for d in daily)
        total_output = sum(d["output_tokens"] for d in daily)
        total_cost = sum(d["cost"] for d in daily)
        total_requests = sum(d["requests"] for d in daily)

        return {
            "period_days": days,
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": round(total_cost, 6),
            "daily": daily,
        }

    async def get_transactions(self, user_id: int, limit: int = 50) -> list:
        rows = await self.db.fetchall(
            "SELECT * FROM transactions WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
            user_id, limit
        )
        result = []
        for r in rows:
            d = dict(r)
            d["amount"] = d["amount_micro"] / MICRO
            del d["amount_micro"]
            if isinstance(d.get('created_at'), datetime):
                d['created_at'] = d['created_at'].isoformat()
            result.append(d)
        return result

    # ── Crypto invoices ──

    async def create_crypto_invoice(self, user_id: int, amount_usd: float,
                                     chain: str = "base", currency: str = "USDC") -> dict:
        if not self.crypto_deposit_address:
            raise HTTPException(status_code=503, detail="Crypto payments not configured. Contact support.")

        invoice_id = f"inv-{secrets.token_hex(8)}"
        import random
        unique_amount = round(amount_usd + random.randint(1, 99) / 100, 2)
        credits = amount_usd

        expires_dt = datetime.now(timezone.utc) + timedelta(hours=1)

        await self.db.execute(
            """INSERT INTO crypto_invoices
               (user_id, invoice_id, amount_usd, credits, chain, currency, deposit_address, expires_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            user_id, invoice_id, unique_amount, credits, chain, currency,
            self.crypto_deposit_address, expires_dt
        )

        expires_str = expires_dt.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "invoice_id": invoice_id,
            "amount_usd": unique_amount,
            "credits": credits,
            "chain": chain,
            "currency": currency,
            "deposit_address": self.crypto_deposit_address,
            "expires_at": expires_str,
            "status": "pending",
            "instructions": f"Send exactly ${unique_amount:.2f} {currency} on {chain.title()} to the deposit address. "
                          f"Your account will be credited {credits:.2f} credits automatically after ~3 block confirmations.",
        }

    async def get_pending_invoices(self, user_id: int) -> list:
        rows = await self.db.fetchall(
            "SELECT * FROM crypto_invoices WHERE user_id = $1 ORDER BY created_at DESC LIMIT 20",
            user_id
        )
        for r in rows:
            for k in ('created_at', 'expires_at', 'paid_at'):
                if isinstance(r.get(k), datetime):
                    r[k] = r[k].isoformat()
        return rows

    async def confirm_crypto_invoice(self, invoice_id: str, tx_hash: str) -> dict:
        row = await self.db.fetchone(
            "SELECT * FROM crypto_invoices WHERE invoice_id = $1 AND status = 'pending'",
            invoice_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Invoice not found or not pending")

        user_id = row["user_id"]
        credits = row["credits"]

        await self.db.execute(
            "UPDATE crypto_invoices SET status = 'paid', tx_hash = $1, paid_at = NOW() WHERE invoice_id = $2",
            tx_hash, invoice_id
        )

        new_balance = await self.add_credits(
            user_id, credits, "crypto",
            f"Crypto payment: {row['currency']} on {row['chain']}",
            tx_hash
        )

        return {"invoice_id": invoice_id, "credits_added": credits, "new_balance": new_balance, "tx_hash": tx_hash}

    # ── TAO payments ──

    async def get_or_create_tao_address(self, user_id: int) -> dict:
        if not HAS_BITTENSOR:
            raise HTTPException(status_code=503, detail="Bittensor SDK not installed")

        row = await self.db.fetchone(
            "SELECT ss58_address, created_at FROM tao_deposit_addresses WHERE user_id = $1",
            user_id
        )
        if row:
            ca = row["created_at"]
            if isinstance(ca, datetime):
                ca = ca.isoformat()
            return {"ss58_address": row["ss58_address"], "created_at": ca}

        mnemonic = Keypair.generate_mnemonic()
        kp = Keypair.create_from_mnemonic(mnemonic)
        seed_hex = kp.seed_hex if hasattr(kp, 'seed_hex') else mnemonic

        await self.db.execute(
            """INSERT INTO tao_deposit_addresses (user_id, ss58_address, seed_hex)
               VALUES ($1, $2, $3)""",
            user_id, kp.ss58_address, seed_hex
        )
        log.info(f"[TAO] Created deposit address {kp.ss58_address} for user {user_id}")
        return {"ss58_address": kp.ss58_address, "created_at": datetime.now(timezone.utc).isoformat()}

    @staticmethod
    def fetch_tao_price_usd() -> float:
        """Fetch current TAO/USD price from CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            price = data.get("bittensor", {}).get("usd")
            if price and price > 0:
                return float(price)
        except Exception as e:
            log.warning(f"[TAO] CoinGecko price fetch failed: {e}")
        raise HTTPException(status_code=503, detail="Unable to fetch TAO price. Try again shortly.")

    async def check_tao_deposits(self):
        """Check all deposit addresses for new TAO and credit users."""
        if not HAS_BITTENSOR:
            return

        rows = await self.db.fetchall(
            "SELECT id, user_id, ss58_address, seed_hex, last_known_balance_rao, total_credited_rao FROM tao_deposit_addresses"
        )
        if not rows:
            return

        try:
            sub = Subtensor(network="finney")
        except Exception as e:
            log.warning(f"[TAO] Cannot connect to subtensor: {e}")
            return

        tao_price = None

        for r in rows:
            try:
                balance = sub.get_balance(r["ss58_address"])
                balance_rao = int(balance.rao) if hasattr(balance, 'rao') else int(float(balance) * 1e9)
            except Exception as e:
                log.warning(f"[TAO] Balance check failed for {r['ss58_address']}: {e}")
                continue

            prev_balance = r["last_known_balance_rao"]
            if balance_rao <= prev_balance:
                if balance_rao != prev_balance:
                    await self.db.execute(
                        "UPDATE tao_deposit_addresses SET last_known_balance_rao = $1 WHERE id = $2",
                        balance_rao, r["id"]
                    )
                continue

            deposit_rao = balance_rao - prev_balance
            deposit_tao = deposit_rao / 1e9

            if tao_price is None:
                try:
                    tao_price = self.fetch_tao_price_usd()
                except Exception:
                    log.warning("[TAO] Cannot credit deposit — price unavailable")
                    continue

            credits_usd = deposit_tao * tao_price

            new_balance = await self.add_credits(
                r["user_id"], credits_usd, "tao",
                f"TAO deposit: {deposit_tao:.4f} TAO @ ${tao_price:.2f}/TAO = ${credits_usd:.2f}",
                r["ss58_address"]
            )

            await self.db.execute(
                "UPDATE tao_deposit_addresses SET last_known_balance_rao = $1, total_credited_rao = total_credited_rao + $2 WHERE id = $3",
                balance_rao, deposit_rao, r["id"]
            )

            log.info(f"[TAO] Credited user {r['user_id']}: {deposit_tao:.4f} TAO (${credits_usd:.2f}) -> balance {new_balance:.2f}")

    # ── Admin ──

    async def admin_credit_user(self, email: str, amount: float, reason: str) -> dict:
        row = await self.db.fetchone(
            "SELECT id, balance_micro FROM users WHERE email = $1", email
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"User {email} not found")

        new_balance = await self.add_credits(row["id"], amount, "admin", reason)
        return {"email": email, "credits_added": amount, "new_balance": new_balance}

    # ── Bittensor wallet auth ──

    async def create_bittensor_nonce(self, ss58_address: str) -> str:
        """Generate and store a nonce for Bittensor wallet auth."""
        nonce = secrets.token_hex(16)
        # Upsert nonce (replace if exists)
        existing = await self.db.fetchone(
            "SELECT ss58_address FROM bittensor_nonces WHERE ss58_address = $1", ss58_address
        )
        if existing:
            await self.db.execute(
                "UPDATE bittensor_nonces SET nonce = $1, created_at = NOW() WHERE ss58_address = $2",
                nonce, ss58_address
            )
        else:
            await self.db.execute(
                "INSERT INTO bittensor_nonces (ss58_address, nonce) VALUES ($1, $2)",
                ss58_address, nonce
            )
        return nonce

    async def verify_bittensor_signature(self, ss58_address: str, signature: str, message: str) -> bool:
        """Verify a Bittensor keypair signature."""
        if not HAS_BITTENSOR:
            raise HTTPException(status_code=503, detail="Bittensor SDK not installed")

        # Validate message format: <ss58_address>:<timestamp>
        parts = message.split(":", 1)
        if len(parts) != 2 or parts[0] != ss58_address:
            return False

        # Check timestamp is within 5 minutes
        try:
            msg_timestamp = int(parts[1])
            now = int(time.time())
            if abs(now - msg_timestamp) > 300:
                return False
        except (ValueError, TypeError):
            return False

        # Verify signature using bittensor Keypair
        try:
            kp = Keypair(ss58_address=ss58_address)
            sig_bytes = bytes.fromhex(signature.replace("0x", ""))
            return kp.verify(message.encode(), sig_bytes)
        except Exception as e:
            log.warning(f"[BT-AUTH] Signature verification failed for {ss58_address}: {e}")
            return False

    async def authenticate_bittensor(self, ss58_address: str, signature: str, message: str) -> dict:
        """Authenticate a Bittensor wallet, auto-create account if new, return user + API key."""
        # Verify signature
        if not await self.verify_bittensor_signature(ss58_address, signature, message):
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Check if this address already has an account
        bt_account = await self.db.fetchone(
            "SELECT * FROM bittensor_accounts WHERE ss58_address = $1", ss58_address
        )

        if bt_account:
            # Existing account — update last_login and return
            await self.db.execute(
                "UPDATE bittensor_accounts SET last_login_at = NOW() WHERE ss58_address = $1",
                ss58_address
            )
            user = await self.db.fetchone(
                "SELECT * FROM users WHERE id = $1", bt_account["user_id"]
            )
            # Clean up old session keys
            await self.db.execute(
                "DELETE FROM api_keys WHERE user_id = $1 AND name LIKE 'session-%'",
                user["id"]
            )
            session_key = await self.create_api_key(user["id"], f"session-{int(time.time())}")
            return {
                "status": "login",
                "user": {
                    "id": user["id"],
                    "email": user["email"],
                    "name": user["name"],
                    "balance": user["balance_micro"] / MICRO,
                    "tier": user["tier"],
                    "ss58_address": ss58_address,
                },
                "api_key": session_key["key"],
                "credits_granted": bt_account["credits_granted"],
            }

        # New account — query chain for registration info and balance
        credits = 0.0
        address_type = "unknown"
        subnets = []
        tao_balance = 0.0

        if HAS_BITTENSOR:
            try:
                credits, address_type, subnets, tao_balance = await asyncio.to_thread(
                    self._query_bittensor_identity, ss58_address
                )
            except Exception as e:
                log.warning(f"[BT-AUTH] Chain query failed for {ss58_address}: {e}")
                # Still allow account creation with 0 credits

        # Create user account (email = ss58 address as identifier)
        email = f"{ss58_address}@bittensor"
        pw_hash = hash_password(secrets.token_hex(32))  # random password, won't be used
        initial_balance = int(credits * MICRO)

        try:
            user_id = await self.db.execute_returning_id(
                "INSERT INTO users (email, password_hash, name, balance_micro, tier) VALUES ($1, $2, $3, $4, $5) RETURNING id",
                email, pw_hash, f"bt:{ss58_address[:8]}...{ss58_address[-4:]}", initial_balance, "bittensor"
            )
        except Exception as e:
            err_str = str(e).lower()
            if "unique" in err_str or "duplicate" in err_str:
                raise HTTPException(status_code=409, detail="Address already registered")
            raise

        # Record signup bonus
        if credits > 0:
            await self.db.execute(
                "INSERT INTO transactions (user_id, amount_micro, type, description) VALUES ($1, $2, 'bittensor_signup', $3)",
                user_id, initial_balance,
                f"Bittensor {address_type} signup: {credits:.2f} credits"
            )

        # Store bittensor account mapping
        subnets_str = ",".join(str(s) for s in subnets) if subnets else None
        await self.db.execute(
            """INSERT INTO bittensor_accounts (user_id, ss58_address, address_type, subnets_registered, tao_balance, credits_granted)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            user_id, ss58_address, address_type, subnets_str, tao_balance, credits
        )

        # Create API key
        key_info = await self.create_api_key(user_id, "default")

        log.info(f"[BT-AUTH] New {address_type} account: {ss58_address[:12]}... -> user {user_id}, ${credits:.2f} credits")

        return {
            "status": "created",
            "user": {
                "id": user_id,
                "email": email,
                "name": f"bt:{ss58_address[:8]}...{ss58_address[-4:]}",
                "balance": credits,
                "tier": "bittensor",
                "ss58_address": ss58_address,
            },
            "api_key": key_info["key"],
            "credits_granted": credits,
            "breakdown": {
                "address_type": address_type,
                "subnets_registered": subnets,
                "tao_balance": tao_balance,
                "hotkey_bonus": 100.0 if address_type == "hotkey" and subnets else 0.0,
                "tao_credits": tao_balance if address_type == "coldkey" else 0.0,
            },
            "message": f"Account created with ${credits:.2f} in free credits. Save your API key."
        }

    @staticmethod
    def _query_bittensor_identity(ss58_address: str) -> tuple:
        """Query chain for hotkey/coldkey status and balance. Runs in thread (blocking)."""
        sub = Subtensor(network="finney")
        credits = 0.0
        address_type = "unknown"
        subnets = []
        tao_balance = 0.0

        # Check if address is a registered hotkey on any subnet
        try:
            all_netuids = sub.get_all_subnet_netuids()
            for netuid in all_netuids:
                try:
                    if sub.is_hotkey_registered(netuid=netuid, hotkey_ss58=ss58_address):
                        subnets.append(netuid)
                except Exception:
                    continue

            if subnets:
                address_type = "hotkey"
                credits = 100.0  # $100 for any registered hotkey
        except Exception as e:
            log.warning(f"[BT-AUTH] Subnet registration check failed: {e}")

        # Check TAO balance (works for both hotkeys and coldkeys)
        try:
            balance = sub.get_balance(ss58_address)
            tao_balance = float(balance) if not hasattr(balance, 'tao') else float(balance.tao)
        except Exception as e:
            log.warning(f"[BT-AUTH] Balance check failed: {e}")

        # If not a hotkey but has TAO balance, it's a coldkey
        if not subnets and tao_balance > 0:
            address_type = "coldkey"
            credits = tao_balance  # $1 per TAO
        elif subnets:
            # Hotkey gets $100 + any TAO balance
            credits = 100.0 + tao_balance

        return credits, address_type, subnets, tao_balance

    # ── Inference proxy ──

    async def proxy_inference(self, request: Request, user: dict, endpoint: str) -> Response:
        """Proxy an inference request to the gateway, deducting credits."""
        if user["balance_micro"] <= 0:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Insufficient credits",
                    "balance": user["balance_micro"] / MICRO,
                    "message": "Please add credits to continue. Visit constantinople.cloud/billing"
                }
            )

        body = await request.body()
        request_id = secrets.token_hex(8)
        stream = False

        try:
            parsed = json.loads(body)
            stream = parsed.get("stream", False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        session = await self.get_session()
        start = time.monotonic()

        target_url = f"{self.gateway_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        try:
            if stream:
                return await self._proxy_stream(session, target_url, headers, body,
                                                 user, request_id, endpoint, start)
            else:
                return await self._proxy_sync(session, target_url, headers, body,
                                               user, request_id, endpoint, start)
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=502, detail=f"Gateway error: {str(e)}")

    async def _proxy_sync(self, session, url, headers, body, user, request_id, endpoint, start):
        async with session.post(url, headers=headers, data=body) as resp:
            result = await resp.json()
            latency_ms = (time.monotonic() - start) * 1000

            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            model = result.get("model", "unknown")

            cost = await self.deduct_credits(
                user["id"], user["api_key_id"], request_id,
                input_tokens, output_tokens, model, endpoint,
                latency_ms, resp.status
            )

            result["x_billing"] = {
                "request_id": request_id,
                "cost": round(cost, 6),
                "balance_remaining": await self.get_balance(user["id"]),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            return JSONResponse(content=result, status_code=resp.status)

    async def _proxy_stream(self, session, url, headers, body, user, request_id, endpoint, start):
        """Proxy a streaming response, counting tokens as they arrive."""
        input_tokens = 0
        output_tokens = 0
        model = "unknown"

        async def stream_generator():
            nonlocal input_tokens, output_tokens, model
            async with session.post(url, headers=headers, data=body) as resp:
                async for line in resp.content:
                    decoded = line.decode("utf-8", errors="replace")
                    yield decoded

                    if decoded.startswith("data: ") and decoded.strip() != "data: [DONE]":
                        try:
                            chunk = json.loads(decoded[6:])
                            if "model" in chunk:
                                model = chunk["model"]
                            if "usage" in chunk:
                                u = chunk["usage"]
                                input_tokens = u.get("prompt_tokens", input_tokens)
                                output_tokens = u.get("completion_tokens", output_tokens)
                            choices = chunk.get("choices", [])
                            for c in choices:
                                delta = c.get("delta", {})
                                if delta.get("content"):
                                    output_tokens += 1
                        except (json.JSONDecodeError, KeyError):
                            pass

            latency_ms = (time.monotonic() - start) * 1000
            await self.deduct_credits(
                user["id"], user["api_key_id"], request_id,
                input_tokens, output_tokens, model, endpoint,
                latency_ms, 200
            )

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id}
        )


# ─── FastAPI app ─────────────────────────────────────────────────────────────

backend: Optional[APIBackend] = None

async def _tao_deposit_watcher():
    """Background task: check TAO deposit addresses every 60 seconds."""
    while True:
        try:
            await backend.check_tao_deposits()
        except Exception as e:
            log.error(f"[TAO] Deposit watcher error: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect DB pool
    if backend:
        await backend.db.connect()
        log.info("[DB] Database connection ready")

    watcher_task = None
    tao_watcher_task = None

    if backend and backend.crypto_deposit_address:
        try:
            from crypto_watcher import CryptoWatcher
            rpc_url = os.environ.get("BASE_RPC_URL", "https://base-rpc.publicnode.com")
            watcher = CryptoWatcher(
                deposit_address=backend.crypto_deposit_address,
                db=backend.db,
                confirm_invoice_fn=backend.confirm_crypto_invoice,
                rpc_url=rpc_url,
            )
            watcher_task = asyncio.create_task(watcher.run())
            log.info("[CRYPTO] On-chain payment watcher started")
        except ImportError:
            log.warning("[CRYPTO] web3 not installed — on-chain watcher disabled")
        except Exception as e:
            log.error(f"[CRYPTO] Watcher failed to start: {e}")

    if backend and HAS_BITTENSOR:
        tao_watcher_task = asyncio.create_task(_tao_deposit_watcher())
        log.info("[TAO] Deposit watcher started (60s interval)")

    yield

    if tao_watcher_task:
        tao_watcher_task.cancel()
    if watcher_task:
        watcher_task.cancel()
    if backend:
        await backend.close()

app = FastAPI(
    title="Constantinople API",
    description="Decentralized inference API with credit-based billing",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth dependency ──

async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Extract and validate API key from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer <api_key>")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use: Bearer <api_key>")

    api_key = parts[1].strip()
    user = await backend.get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user

async def get_user_from_session(authorization: Optional[str] = Header(None)) -> dict:
    """Authenticate via email:password basic auth or API key."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")

    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    scheme = parts[0].lower()
    if scheme == "bearer":
        user = await backend.get_user_by_api_key(parts[1].strip())
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user
    elif scheme == "basic":
        import base64
        try:
            decoded = base64.b64decode(parts[1]).decode()
            email, password = decoded.split(":", 1)
            return await backend.authenticate_user(email, password)
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid Basic auth credentials")
    else:
        raise HTTPException(status_code=401, detail="Unsupported auth scheme")

# ── Registration & Login ──

@app.post("/v1/auth/register")
async def register(req: RegisterRequest):
    user = await backend.create_user(req.email, req.password, req.name)
    key_info = await backend.create_api_key(user["id"], "default")
    return {
        "user": user,
        "api_key": key_info,
        "message": f"Account created with {FREE_TIER_CREDITS} free credits. Save your API key — it won't be shown again."
    }

@app.post("/v1/quickstart")
async def quickstart(req: RegisterRequest):
    """Zero-to-inference in one call. Registers, creates API key, returns ready-to-use examples."""
    user = await backend.create_user(req.email, req.password, req.name)
    key_info = await backend.create_api_key(user["id"], "default")
    api_key = key_info["key"]
    base = "https://api.constantinople.cloud"

    return {
        "status": "ready",
        "api_key": api_key,
        "balance": FREE_TIER_CREDITS,
        "curl": (
            f'curl -s {base}/v1/chat/completions '
            f'-H "Authorization: Bearer {api_key}" '
            f'-H "Content-Type: application/json" '
            f'-d \'{{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{{"role":"user","content":"Hello!"}}]}}\''
        ),
        "python": (
            f"from openai import OpenAI\n"
            f"client = OpenAI(base_url='{base}/v1', api_key='{api_key}')\n"
            f"r = client.chat.completions.create(\n"
            f"    model='Qwen/Qwen2.5-7B-Instruct',\n"
            f"    messages=[{{'role': 'user', 'content': 'Hello!'}}]\n"
            f")\n"
            f"print(r.choices[0].message.content)"
        ),
        "next_steps": {
            "check_balance": f"GET {base}/v1/user/balance",
            "add_tao": f"GET {base}/v1/billing/tao-deposit",
            "view_models": f"GET {base}/v1/models",
            "view_usage": f"GET {base}/v1/user/usage",
        },
        "note": f"You have ${FREE_TIER_CREDITS:.2f} free credit. Add TAO to get more."
    }

@app.post("/v1/auth/login")
async def login(req: LoginRequest):
    user = await backend.authenticate_user(req.email, req.password)
    # Clean up old session keys
    await backend.db.execute(
        "DELETE FROM api_keys WHERE user_id = $1 AND name LIKE 'session-%'",
        user["id"]
    )
    session_key_info = await backend.create_api_key(user["id"], f"session-{int(time.time())}")
    keys = await backend.list_api_keys(user["id"])
    return {
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "balance": user["balance_micro"] / MICRO,
            "tier": user["tier"],
        },
        "api_keys": keys,
        "session_key": session_key_info["key"],
    }

# ── Bittensor Wallet Auth ──

@app.post("/v1/auth/bittensor/nonce")
async def bittensor_nonce(req: BittensorNonceRequest):
    """Get a signing nonce for Bittensor wallet authentication."""
    nonce = await backend.create_bittensor_nonce(req.ss58_address)
    timestamp = int(time.time())
    message = f"{req.ss58_address}:{timestamp}"
    return {
        "message": message,
        "nonce": nonce,
        "timestamp": timestamp,
        "instructions": "Sign the 'message' field with your Bittensor keypair and POST to /v1/auth/bittensor",
        "example": {
            "python": (
                "from bittensor import Keypair\n"
                f"kp = Keypair.create_from_seed('<your-seed>')\n"
                f"sig = kp.sign('{message}'.encode()).hex()\n"
                "# POST /v1/auth/bittensor with ss58_address, signature, message"
            )
        }
    }

@app.post("/v1/auth/bittensor")
async def bittensor_auth(req: BittensorAuthRequest):
    """Authenticate with a Bittensor wallet. Auto-creates account with free credits.

    Hotkeys registered on any subnet get $100 free.
    Coldkeys get $1 per TAO in balance.
    """
    result = await backend.authenticate_bittensor(req.ss58_address, req.signature, req.message)
    return result

# ── API Key Management ──

@app.post("/v1/keys/create")
async def create_key(req: CreateKeyRequest, user: dict = Depends(get_user_from_session)):
    key_info = await backend.create_api_key(user["id"], req.name)
    return key_info

@app.get("/v1/keys")
async def list_keys(user: dict = Depends(get_user_from_session)):
    return await backend.list_api_keys(user["id"])

@app.delete("/v1/keys/{key_id}")
async def revoke_key(key_id: int, user: dict = Depends(get_user_from_session)):
    if user.get("api_key_id") == key_id:
        raise HTTPException(status_code=409, detail="Cannot revoke the key you are currently using. Switch to another key first.")
    if await backend.revoke_api_key(user["id"], key_id):
        return {"status": "revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

# ── Balance & Usage ──

@app.get("/v1/user/balance")
async def get_balance(user: dict = Depends(get_current_user)):
    balance = await backend.get_balance(user["id"])
    return {
        "balance": balance,
        "tier": user.get("tier", "free"),
        "pricing": {
            "input_per_1m_tokens": DEFAULT_INPUT_PRICE,
            "output_per_1m_tokens": DEFAULT_OUTPUT_PRICE,
            "currency": "credits",
        }
    }

@app.get("/v1/user/usage")
async def get_usage(days: int = 30, user: dict = Depends(get_current_user)):
    return await backend.get_usage(user["id"], min(days, 365))

@app.get("/v1/user/transactions")
async def get_transactions(limit: int = 50, user: dict = Depends(get_current_user)):
    return await backend.get_transactions(user["id"], min(limit, 200))

# ── Billing ──

@app.post("/v1/billing/topup")
async def topup(req: TopUpRequest, user: dict = Depends(get_user_from_session)):
    if req.payment_method == "stripe":
        if not backend.stripe_secret:
            raise HTTPException(status_code=503, detail="Stripe not configured. Contact support.")

        try:
            import stripe
            stripe.api_key = backend.stripe_secret

            stripe_customer_id = user.get("stripe_customer_id")
            if not stripe_customer_id:
                customer = stripe.Customer.create(email=user["email"])
                stripe_customer_id = customer.id
                await backend.db.execute(
                    "UPDATE users SET stripe_customer_id = $1 WHERE id = $2",
                    stripe_customer_id, user["id"]
                )

            amount_cents = max(int(req.amount * 100), 500)

            checkout = stripe.checkout.Session.create(
                customer=stripe_customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": f"Constantinople API Credits ({req.amount:.2f})",
                            "description": f"{req.amount:.2f} API credits for inference",
                        },
                        "unit_amount": amount_cents,
                    },
                    "quantity": 1,
                }],
                mode="payment",
                success_url="https://www.constantinople.cloud/billing?status=success",
                cancel_url="https://www.constantinople.cloud/billing?status=cancelled",
                metadata={
                    "user_id": str(user["id"]),
                    "credits": str(req.amount),
                },
            )

            return {
                "checkout_url": checkout.url,
                "session_id": checkout.id,
                "amount_usd": amount_cents / 100,
                "credits": req.amount,
            }

        except ImportError:
            raise HTTPException(status_code=503, detail="Stripe library not installed")
        except Exception as e:
            log.error(f"Stripe error: {e}")
            raise HTTPException(status_code=500, detail="Payment processing error")

    elif req.payment_method == "crypto":
        currency = req.currency.upper() if req.currency else "USDC"
        if currency not in ("USDC", "ETH"):
            raise HTTPException(status_code=400, detail="Supported currencies: USDC, ETH")
        invoice = await backend.create_crypto_invoice(
            user["id"], req.amount,
            chain="base", currency=currency
        )
        return invoice
    else:
        raise HTTPException(status_code=400, detail="Unsupported payment method")

# ── Stripe Webhook ──

@app.post("/v1/billing/webhook/stripe")
async def stripe_webhook(request: Request):
    if not backend.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook not configured")

    try:
        import stripe
        stripe.api_key = backend.stripe_secret

        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")

        event = stripe.Webhook.construct_event(
            payload, sig_header, backend.stripe_webhook_secret
        )

        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = int(session["metadata"]["user_id"])
            credits = float(session["metadata"]["credits"])

            new_balance = await backend.add_credits(
                user_id, credits, "stripe",
                f"Stripe checkout ${session['amount_total']/100:.2f}",
                session["id"]
            )
            log.info(f"[STRIPE] User {user_id} topped up {credits} credits. New balance: {new_balance}")

        return {"status": "ok"}

    except ImportError:
        raise HTTPException(status_code=503, detail="Stripe library not installed")
    except Exception as e:
        log.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ── Crypto Invoices ──

@app.get("/v1/billing/invoices")
async def list_invoices(user: dict = Depends(get_current_user)):
    return await backend.get_pending_invoices(user["id"])

@app.get("/v1/billing/crypto-info")
async def crypto_info():
    """Public endpoint showing accepted crypto payment methods."""
    return {
        "accepted": [
            {
                "chain": "Base",
                "currency": "USDC",
                "contract": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "decimals": 6,
                "confirmations": 3,
                "min_amount": 1.0,
                "note": "Recommended — low fees, fast confirmation",
            },
            {
                "chain": "Base",
                "currency": "ETH",
                "contract": "native",
                "confirmations": 3,
                "min_amount": 1.0,
                "note": "Converted at market rate (CoinGecko)",
            },
        ],
        "deposit_address": backend.crypto_deposit_address or "Not configured — contact support",
        "rate": "1 USDC = 1 credit = $1 USD",
        "auto_confirm": True,
        "network": "Base (Chain ID 8453)",
    }

# ── TAO Payments ──

@app.get("/v1/billing/tao-price")
async def tao_price():
    """Get current TAO/USD price."""
    price = backend.fetch_tao_price_usd()
    return {"tao_usd": price, "source": "coingecko", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/v1/user/deposit-address")
@app.get("/v1/billing/tao-deposit")
async def tao_deposit(user: dict = Depends(get_user_from_session)):
    """Get the user's unique TAO deposit address. Creates one if it doesn't exist."""
    addr_info = await backend.get_or_create_tao_address(user["id"])
    price = backend.fetch_tao_price_usd()
    return {
        "deposit_address": addr_info["ss58_address"],
        "network": "Bittensor (finney)",
        "currency": "TAO",
        "tao_usd_price": price,
        "rate": f"1 TAO = ${price:.2f} in credits",
        "created_at": addr_info["created_at"],
        "instructions": f"Send any amount of TAO to {addr_info['ss58_address']}. "
                        f"Credits will be added automatically at the current rate (${price:.2f}/TAO). "
                        f"Deposits are checked every 60 seconds.",
        "min_deposit": 0.01,
    }

@app.get("/v1/billing/tao-balance")
async def tao_deposit_balance(user: dict = Depends(get_user_from_session)):
    """Check the balance of the user's TAO deposit address."""
    row = await backend.db.fetchone(
        "SELECT ss58_address, last_known_balance_rao, total_credited_rao FROM tao_deposit_addresses WHERE user_id = $1",
        user["id"]
    )
    if not row:
        return {"has_address": False, "message": "No deposit address yet. Call /v1/billing/tao-deposit first."}
    return {
        "has_address": True,
        "deposit_address": row["ss58_address"],
        "pending_balance_tao": row["last_known_balance_rao"] / 1e9,
        "total_credited_tao": row["total_credited_rao"] / 1e9,
    }

# ── Admin ──

async def verify_admin(authorization: Optional[str] = Header(None)):
    """Verify admin API key."""
    if not backend.admin_key:
        raise HTTPException(status_code=503, detail="Admin not configured")
    if not authorization:
        raise HTTPException(status_code=401, detail="Admin auth required")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth format")
    if not hmac.compare_digest(parts[1].strip(), backend.admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True

@app.post("/v1/admin/credit")
async def admin_credit(
    email: str, amount: float, reason: str = "Manual credit",
    _admin: bool = Depends(verify_admin)
):
    """Admin: manually credit a user account."""
    return await backend.admin_credit_user(email, amount, reason)

@app.post("/v1/admin/confirm-invoice")
async def admin_confirm_invoice(
    invoice_id: str, tx_hash: str,
    _admin: bool = Depends(verify_admin)
):
    """Admin: confirm a crypto payment invoice."""
    return await backend.confirm_crypto_invoice(invoice_id, tx_hash)

@app.get("/v1/admin/users")
async def admin_list_users(_admin: bool = Depends(verify_admin)):
    """Admin: list all users with balances."""
    rows = await backend.db.fetchall(
        "SELECT id, email, name, balance_micro, tier, created_at, is_active FROM users ORDER BY created_at DESC"
    )
    users = []
    for r in rows:
        d = dict(r)
        d["balance"] = d["balance_micro"] / MICRO
        del d["balance_micro"]
        if isinstance(d.get('created_at'), datetime):
            d['created_at'] = d['created_at'].isoformat()
        users.append(d)
    return users

@app.get("/v1/admin/pending-invoices")
async def admin_pending_invoices(_admin: bool = Depends(verify_admin)):
    """Admin: list all pending crypto invoices."""
    rows = await backend.db.fetchall(
        "SELECT * FROM crypto_invoices WHERE status = 'pending' ORDER BY created_at DESC"
    )
    for r in rows:
        for k in ('created_at', 'expires_at', 'paid_at'):
            if isinstance(r.get(k), datetime):
                r[k] = r[k].isoformat()
    return rows

# ── Inference Proxy ──

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, user: dict = Depends(get_current_user)):
    return await backend.proxy_inference(request, user, "/v1/chat/completions")

@app.post("/v1/completions")
async def completions(request: Request, user: dict = Depends(get_current_user)):
    return await backend.proxy_inference(request, user, "/v1/completions")

@app.post("/v1/embeddings")
async def embeddings(request: Request, user: dict = Depends(get_current_user)):
    return await backend.proxy_inference(request, user, "/v1/embeddings")

# ── Public endpoints (no auth) ──

@app.get("/")
async def root():
    """Landing page for agents and developers hitting the API root."""
    return {
        "service": "Constantinople",
        "description": "Decentralized LLM inference on Bittensor. OpenAI-compatible API.",
        "quickstart": {
            "1_register": {
                "method": "POST",
                "url": "/v1/auth/register",
                "body": {"email": "you@example.com", "password": "your-password"},
                "note": "Returns an API key and $1 free credit."
            },
            "2_inference": {
                "method": "POST",
                "url": "/v1/chat/completions",
                "headers": {"Authorization": "Bearer <your-api-key>"},
                "body": {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [{"role": "user", "content": "Hello!"}]
                }
            },
            "one_step": {
                "method": "POST",
                "url": "/v1/quickstart",
                "body": {"email": "you@example.com", "password": "your-password"},
                "note": "Register + get a ready-to-use curl command in one call."
            }
        },
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "auth": ["/v1/auth/register", "/v1/auth/login", "/v1/auth/bittensor/nonce", "/v1/auth/bittensor"],
            "inference": ["/v1/chat/completions", "/v1/completions"],
            "billing": ["/v1/billing/tao-deposit", "/v1/billing/topup", "/v1/user/balance"],
            "info": ["/v1/models", "/v1/pricing", "/health"],
        },
        "openai_sdk": {
            "base_url": "https://api.constantinople.cloud/v1",
            "note": "Drop-in replacement for OpenAI SDK. Set base_url and api_key."
        }
    }

@app.get("/health")
async def health():
    try:
        session = await backend.get_session()
        async with session.get(f"{backend.gateway_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            gw_health = await resp.json()
    except Exception:
        gw_health = {"status": "unreachable"}

    user_count = await backend.db.fetchval("SELECT COUNT(*) FROM users")
    key_count = await backend.db.fetchval("SELECT COUNT(*) FROM api_keys WHERE is_active = TRUE")

    db_type = "postgresql" if isinstance(backend.db, PostgresDB) else "sqlite"

    return {
        "status": "ok",
        "service": "constantinople-api",
        "version": "0.2.0",
        "database": db_type,
        "users": user_count,
        "active_keys": key_count,
        "gateway": gw_health,
        "pricing": {
            "input_per_1m_tokens": DEFAULT_INPUT_PRICE,
            "output_per_1m_tokens": DEFAULT_OUTPUT_PRICE,
            "free_tier_credits": FREE_TIER_CREDITS,
        }
    }

@app.get("/v1/models")
async def list_models():
    """Proxy to gateway models endpoint."""
    try:
        session = await backend.get_session()
        async with session.get(f"{backend.gateway_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            return await resp.json()
    except Exception:
        return {"object": "list", "data": []}

@app.get("/v1/pricing")
async def pricing():
    return {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "input_per_1m_tokens": DEFAULT_INPUT_PRICE,
        "output_per_1m_tokens": DEFAULT_OUTPUT_PRICE,
        "free_tier_credits": FREE_TIER_CREDITS,
        "currency": "USD (1 credit = $1)",
        "payment_methods": ["crypto", "tao"],
        "notes": "Pay with TAO on Bittensor, or USDC/ETH on Base.",
    }

# ── Passthrough for audit/dataset endpoints ──

def _get_r2_client():
    """Get R2 (S3-compatible) client for audit data access."""
    import boto3
    url = os.environ.get("R2_URL", "")
    key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if not (url and key and secret):
        return None
    return boto3.client("s3", endpoint_url=url, aws_access_key_id=key,
                        aws_secret_access_key=secret, region_name="auto")

@app.get("/v1/dataset/recent")
async def dataset_recent(limit: int = 50, date: str = None):
    """Get recent inference audit records from R2. Public, no auth needed.
    Use ?date=2026-03-14 to filter by date, ?limit=N for count (max 200)."""
    s3 = _get_r2_client()
    if not s3:
        raise HTTPException(503, "Dataset storage not configured")
    bucket = os.environ.get("R2_BUCKET", "affine")
    limit = min(limit, 200)
    if date:
        prefix = f"audit/{date.replace('-','/')[:10]}/"
    else:
        from datetime import datetime as dt
        now = dt.utcnow()
        prefix = f"audit/{now.strftime('%Y/%m/%d')}/"
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=limit)
        records = []
        for obj in resp.get("Contents", []):
            data = s3.get_object(Bucket=bucket, Key=obj["Key"])
            record = json.loads(data["Body"].read())
            # Strip raw hidden state data (large) — keep metadata only
            record.pop("commitments", None)
            record.pop("all_token_ids", None)
            records.append(record)
        return {"count": len(records), "date": date or "today", "records": records}
    except Exception as e:
        raise HTTPException(500, f"Dataset read error: {e}")

@app.get("/v1/dataset/epochs")
async def dataset_epochs():
    """Get all epoch weight summaries. Public, no auth needed."""
    s3 = _get_r2_client()
    if not s3:
        raise HTTPException(503, "Dataset storage not configured")
    bucket = os.environ.get("R2_BUCKET", "affine")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="epochs/", MaxKeys=100)
        epochs = []
        for obj in resp.get("Contents", []):
            data = s3.get_object(Bucket=bucket, Key=obj["Key"])
            epochs.append(json.loads(data["Body"].read()))
        return {"count": len(epochs), "epochs": epochs}
    except Exception as e:
        raise HTTPException(500, f"Dataset read error: {e}")

@app.get("/v1/dataset/stats")
async def dataset_stats():
    """Get dataset summary stats. Public, no auth needed."""
    s3 = _get_r2_client()
    if not s3:
        raise HTTPException(503, "Dataset storage not configured")
    bucket = os.environ.get("R2_BUCKET", "affine")
    try:
        paginator = s3.get_paginator("list_objects_v2")
        total = 0
        for page in paginator.paginate(Bucket=bucket, Prefix="audit/"):
            total += len(page.get("Contents", []))
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="epochs/", MaxKeys=100)
        n_epochs = len(resp.get("Contents", []))
        return {
            "total_audit_records": total,
            "total_epochs": n_epochs,
            "access": {
                "recent": "/v1/dataset/recent?limit=50&date=2026-03-14",
                "epochs": "/v1/dataset/epochs",
                "curl": "curl https://api.constantinople.cloud/v1/dataset/recent?limit=10",
            },
        }
    except Exception as e:
        raise HTTPException(500, f"Dataset read error: {e}")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Constantinople API Backend")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gateway", type=str, default="http://localhost:8081",
                        help="Gateway URL to proxy inference requests to")
    parser.add_argument("--db", type=str, default="./api.db",
                        help="SQLite database path (fallback if DATABASE_URL not set)")
    parser.add_argument("--input-price", type=float, default=DEFAULT_INPUT_PRICE,
                        help="Credits per 1M input tokens")
    parser.add_argument("--output-price", type=float, default=DEFAULT_OUTPUT_PRICE,
                        help="Credits per 1M output tokens")
    parser.add_argument("--free-credits", type=float, default=FREE_TIER_CREDITS,
                        help="Free credits on signup")
    args = parser.parse_args()

    # Choose database backend
    database_url = os.environ.get("DATABASE_URL", "")
    if database_url and HAS_ASYNCPG:
        db = PostgresDB(database_url)
        log.info(f"[DB] Using PostgreSQL (Supabase)")
    else:
        if database_url and not HAS_ASYNCPG:
            log.warning("[DB] DATABASE_URL set but asyncpg not installed — falling back to SQLite")
        db = SQLiteDB(args.db)
        log.info(f"[DB] Using SQLite: {args.db}")

    global backend
    backend = APIBackend(
        db=db,
        gateway_url=args.gateway,
        input_price=args.input_price,
        output_price=args.output_price,
        free_credits=args.free_credits,
        stripe_secret=os.environ.get("STRIPE_SECRET_KEY"),
        stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET"),
        crypto_deposit_address=os.environ.get("CRYPTO_DEPOSIT_ADDRESS", ""),
        admin_key=os.environ.get("API_ADMIN_KEY"),
    )

    log.info(f"Starting Constantinople API Backend on {args.host}:{args.port}")
    log.info(f"Gateway: {args.gateway}")
    log.info(f"Pricing: ${args.input_price}/1M input, ${args.output_price}/1M output")
    log.info(f"Free tier: {args.free_credits} credits")
    log.info(f"Crypto deposit: {'configured' if backend.crypto_deposit_address else 'not configured'}")
    log.info(f"Admin key: {'configured' if backend.admin_key else 'not configured'}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
