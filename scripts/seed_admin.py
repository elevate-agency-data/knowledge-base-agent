"""
Seed the first admin account for the ACTIVE brand's user store.

Each brand has its own SQLite user DB (shared/brand.py → users_db_path), so a
fresh brand starts with zero accounts and nobody can log in. Run this once per
brand to bootstrap an owner/admin.

Usage (the profile is chosen by BRAND_PROFILE, default = hermes):

    # Windows PowerShell
    $env:BRAND_PROFILE="hermes"; python -m scripts.seed_admin --email a@b.com --password "Secret123" --name "Admin"

    # bash
    BRAND_PROFILE=hermes python -m scripts.seed_admin --email a@b.com --password "Secret123" --name "Admin"

The account is created with is_admin=1 and the brand's owner role (roles[0]).
Idempotent-ish: if the email already exists, it reports and exits without error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.brand import ACTIVE            # noqa: E402
from ui import auth                        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the first admin for the active brand.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument(
        "--role",
        default=None,
        help=f"Business role (default = owner role '{ACTIVE.roles[0] if ACTIVE.roles else ACTIVE.default_role}').",
    )
    args = parser.parse_args()

    owner_role = args.role or (ACTIVE.roles[0] if ACTIVE.roles else ACTIVE.default_role)

    print(f"Brand         : {ACTIVE.name}")
    print(f"User store    : {auth._DB_PATH}")
    print(f"Creating admin: {args.email}  (role={owner_role}, is_admin=True)")

    user = auth.create_user(
        email=args.email,
        password=args.password,
        display_name=args.name,
        is_admin=True,
        role=owner_role,
    )
    if user is None:
        print("→ Email already exists in this brand's store. Nothing created.")
        return 0
    print(f"→ Created (id={user['id']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
