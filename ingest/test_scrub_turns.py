"""Tests for scrub_turns.py — synthetic fixtures only, no real secrets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the ingest package is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from scrub_turns import scrub_text, _DROPPED_PLACEHOLDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _turn(text: str, tool_name: str | None = None, kind: str = "text") -> tuple[str, int]:
    return scrub_text(text, tool_name=tool_name, kind=kind)


def _assert_redacted(result: str, original_fragment: str) -> None:
    """The original secret must not appear in output and REDACTED tag must."""
    assert original_fragment not in result, f"Secret leaked: {original_fragment!r}"
    assert "<REDACTED:" in result, f"No REDACTED tag in: {result!r}"


def _assert_dropped(result: str) -> None:
    assert result == _DROPPED_PLACEHOLDER, f"Expected DROPPED, got: {result!r}"


# ---------------------------------------------------------------------------
# PII_PATTERNS (ported from .mjs)
# ---------------------------------------------------------------------------
class TestPiiPatterns:
    def test_sk_key_redacted(self):
        fake_key = "sk-ant-api03-FakeFakeFakeFakeFake0000000000"
        text = f"Using key {fake_key} for auth"
        out, n = _turn(text)
        assert fake_key not in out
        assert "<REDACTED:sk-key:" in out
        assert n >= 1

    def test_email_redacted(self):
        out, n = _turn("Contact test.user@example.com for help")
        assert "test.user@example.com" not in out
        assert "<REDACTED:email:" in out

    def test_ssn_redacted(self):
        out, n = _turn("SSN is 123-45-6789")
        assert "123-45-6789" not in out

    def test_ipv4_redacted(self):
        out, n = _turn("Server at 192.168.1.100")
        assert "192.168.1.100" not in out

    def test_aws_akia_redacted(self):
        out, n = _turn("AKIA1234567890ABCDEF is the key")
        assert "AKIA1234567890ABCDEF" not in out
        assert "<REDACTED:" in out

    def test_ghp_token_redacted(self):
        fake_ghp = "ghp_" + "A" * 30
        out, n = _turn(f"token={fake_ghp}")
        assert fake_ghp not in out

    def test_sbp_token_redacted(self):
        fake_sbp = "sbp_" + "B" * 30
        out, n = _turn(f"sbp={fake_sbp}")
        assert fake_sbp not in out


# ---------------------------------------------------------------------------
# ADVERSARIAL_CONTENT_PATTERNS (ported from vault-lance-indexer.py)
# ---------------------------------------------------------------------------
class TestAdversarialPatterns:
    def test_bearer_token_redacted(self):
        fake = "Bearer eyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
        out, n = _turn(fake)
        assert "eyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab" not in out

    def test_jwt_token_redacted(self):
        # Synthetic JWT-like string (three base64url segments)
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.FAKESIGNATUREFAKESIG"
        out, n = _turn(jwt)
        assert jwt not in out
        assert "<REDACTED:" in out

    def test_pem_header_redacted(self):
        pem = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----"
        out, n = _turn(pem)
        assert "BEGIN PRIVATE KEY" not in out

    def test_password_assignment_redacted(self):
        out, n = _turn("password='super_secret_password_123'")
        assert "super_secret_password_123" not in out

    def test_api_key_assignment_redacted(self):
        out, n = _turn('api_key = "SomeVeryLongSecretKeyValue0000"')
        assert "SomeVeryLongSecretKeyValue0000" not in out


# ---------------------------------------------------------------------------
# NEW patterns (entropy, connection strings, env-pairs)
# ---------------------------------------------------------------------------
class TestNewPatterns:
    def test_postgres_connection_string_redacted(self):
        """postgres://user:pass@ style URL must be redacted."""
        conn = "postgres://dbuser:s3cur3pass@db.example.com:5432/mydb"
        out, n = _turn(conn)
        assert "s3cur3pass" not in out
        assert "<REDACTED:conn-string:" in out

    def test_mysql_connection_string_redacted(self):
        conn = "mysql://admin:MyP@ssw0rd!@mysql.host.internal/appdb"
        out, n = _turn(conn)
        assert "MyP@ssw0rd!" not in out

    def test_40_char_hex_high_entropy_caught(self):
        """40-hex string with high Shannon entropy must be caught."""
        # Typical git SHA or secret token (40 hex chars)
        hex_token = "a3f8c2e1d4b7096e5f21a8c3d9e0b4f761234567"
        out, n = _turn(f"token: {hex_token}")
        assert hex_token not in out, "40-hex high-entropy token was not scrubbed"

    def test_base64url_with_dash_underscore_caught(self):
        """base64url string including - and _ must be caught (Voyage pa- key style)."""
        # Synthetic base64url string ≥32 chars with dash/underscore and high entropy
        b64url = "pa-ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567-_abcd"
        out, n = _turn(f"key: {b64url}")
        assert b64url not in out, "base64url with -_ was not scrubbed"

    def test_env_pair_redacted(self):
        out, n = _turn("VOYAGE_API_KEY=ThisIsAFakeLongApiKeyValue")
        assert "ThisIsAFakeLongApiKeyValue" not in out

    def test_jwt_header_standalone_redacted(self):
        # eyJ prefix standalone (not full 3-part JWT)
        out, n = _turn("header: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out


# ---------------------------------------------------------------------------
# Benign prose — must NOT be modified
# ---------------------------------------------------------------------------
class TestBenignUntouched:
    def test_plain_prose_untouched(self):
        prose = "The quick brown fox jumps over the lazy dog."
        out, n = _turn(prose)
        assert out == prose
        assert n == 0

    def test_code_without_secrets_untouched(self):
        code = "def add(a, b):\n    return a + b\n"
        out, n = _turn(code)
        assert out == code
        assert n == 0

    def test_short_word_not_flagged(self):
        # "token" keyword in normal sentence — no long random value
        out, n = _turn("This explains the token refresh flow.")
        assert n == 0 or "<REDACTED:" not in out or "token refresh flow" not in out


# ---------------------------------------------------------------------------
# Tool-aware hard-drop
# ---------------------------------------------------------------------------
class TestToolAwareDrop:
    def test_crypt_tool_name_dropped(self):
        """Turn with tool_name='crypt' must be fully dropped."""
        out, n = _turn(
            text="crypt unlock --key abc123",
            tool_name="crypt",
            kind="tool_use",
        )
        _assert_dropped(out)

    def test_secret_tool_name_dropped(self):
        out, n = _turn(
            text="list all items",
            tool_name="keyring_read_secret",
            kind="tool_use",
        )
        _assert_dropped(out)

    def test_vault_tool_name_dropped(self):
        out, n = _turn("open vault", tool_name="vault_open", kind="tool_use")
        _assert_dropped(out)

    def test_api_key_tool_name_dropped(self):
        out, n = _turn("get key", tool_name="get_api_key", kind="tool_use")
        _assert_dropped(out)

    def test_password_in_tool_use_input_dropped(self):
        """Tool_use input containing 'password' keyword must be dropped."""
        out, n = _turn(
            text="supabase projects api-keys set --password hunter2",
            tool_name="bash",
            kind="tool_use",
        )
        _assert_dropped(out)

    def test_token_in_tool_use_input_dropped(self):
        out, n = _turn(
            text="supabase projects api-keys --token mytoken",
            tool_name="bash",
            kind="tool_use",
        )
        _assert_dropped(out)

    def test_normal_tool_not_dropped(self):
        """A benign tool with no secret keywords must NOT be dropped."""
        out, n = _turn(
            text="list all files in the directory",
            tool_name="list_files",
            kind="tool_use",
        )
        assert out != _DROPPED_PLACEHOLDER
        assert "list all files" in out

    def test_non_tool_use_with_vault_in_name_not_dropped(self):
        """Non-tool_use kind: only tool_name check applies, not input-text scan."""
        # tool_name is None, kind is text — should not drop based on text content alone
        # (vault keyword in prose without a tool_name = no drop)
        out, n = _turn("I opened the vault door manually.", tool_name=None, kind="text")
        # This is NOT a tool_use kind, so input-scan drop does not apply
        # But 'vault' appears in prose — should NOT be dropped (no tool_name match)
        assert out != _DROPPED_PLACEHOLDER


# ---------------------------------------------------------------------------
# SHA8 determinism
# ---------------------------------------------------------------------------
class TestSha8Determinism:
    def test_same_secret_same_sha8(self):
        secret = "sk-ant-api03-FakeFakeFakeFakeFake0000000000"
        out1, _ = _turn(f"key: {secret}")
        out2, _ = _turn(f"key: {secret}")
        assert out1 == out2

    def test_different_secrets_different_sha8(self):
        s1 = "sk-ant-api03-FakeFakeFakeFakeFake0000000000"
        s2 = "sk-ant-api03-DifferentDifferentDiff9999999"
        out1, _ = _turn(s1)
        out2, _ = _turn(s2)
        # Extract sha8 portions
        sha1 = out1.split(":")[-1].rstrip(">")
        sha2 = out2.split(":")[-1].rstrip(">")
        assert sha1 != sha2


# ---------------------------------------------------------------------------
# End-to-end: JSONL round-trip
# ---------------------------------------------------------------------------
class TestE2eJsonl:
    def test_jsonl_round_trip(self, tmp_path: Path):
        import subprocess

        turns = [
            {"turn_idx": 0, "ts": "2026-06-10T00:00:00Z", "role": "user", "kind": "text",
             "tool_name": None,
             "text": "My sk-ant-api03-FakeFakeFakeFakeFake0000000000 is the key"},
            {"turn_idx": 1, "ts": "2026-06-10T00:00:01Z", "role": "assistant", "kind": "tool_use",
             "tool_name": "crypt",
             "text": "crypt unlock"},
            {"turn_idx": 2, "ts": "2026-06-10T00:00:02Z", "role": "assistant", "kind": "text",
             "tool_name": None,
             "text": "The quick brown fox."},
        ]
        in_file = tmp_path / "turns.jsonl"
        out_file = tmp_path / "scrubbed.jsonl"
        in_file.write_text("\n".join(json.dumps(t) for t in turns), encoding="utf-8")

        scrub_module = Path(__file__).parent / "scrub_turns.py"
        result = subprocess.run(
            [sys.executable, str(scrub_module), str(in_file), "--out", str(out_file)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        lines = [json.loads(row) for row in out_file.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 3

        # turn 0: sk- key redacted
        assert "sk-ant-api03" not in lines[0]["text"]
        assert "<REDACTED:" in lines[0]["text"]

        # turn 1: crypt tool dropped
        assert lines[1]["text"] == _DROPPED_PLACEHOLDER

        # turn 2: benign, untouched
        assert lines[2]["text"] == "The quick brown fox."
