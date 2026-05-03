#!/usr/bin/env python3
"""Create and push a multi-arch OCI manifest list to an insecure HTTP registry.

Usage:
    python3 scripts/push_manifest.py \
        --registry 192.168.0.71:31500 \
        --repo llmmllab-api \
        --tag main \
        --user bcf186aef4ebc292 \
        --password b6c98846d1e66359903a2137
"""

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


def get_image_digest(image: str) -> str:
    """Get the full sha256:<hex> digest for a locally tagged image."""
    out = subprocess.check_output(
        [
            "docker", "inspect", "--format", "{{index .RepoDigests 0}}", image,
        ],
        text=True,
    ).strip()
    if "@" in out:
        return out.split("@", 1)[1]
    img_id = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Id}}", image], text=True
    ).strip()
    return f"sha256:{img_id}"


def fetch_manifest_from_registry(
    registry: str, repo: str, tag: str, auth: str,
) -> tuple[str, int]:
    """Fetch a manifest from the registry, returning (digest, size)."""
    url = f"http://{registry}/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {auth}",
            "Accept": "application/vnd.oci.image.manifest.v1+json",
        },
        method="GET",
    )
    for attempt in range(1, 7):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read()
                size = len(body)
                digest = resp.headers.get(
                    "Docker-Content-Digest"
                ) or resp.headers.get("ETag", "").strip('"')
                return digest, size
        except urllib.error.HTTPError as e:
            if e.code == 404 and attempt < 7:
                wait = 2 ** attempt
                print(f"  Manifest not found (attempt {attempt}/6), waiting {wait}s...")
                time.sleep(wait)
            else:
                detail = e.read().decode()
                raise SystemExit(f"  HTTP {e.code}: {detail}")
    raise SystemExit("  Manifest fetch exhausted retries")


def push_manifest(
    registry: str, repo: str, tag: str, auth: str, body: bytes,
) -> None:
    """PUT the manifest body to the registry under the given tag."""
    url = f"http://{registry}/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/vnd.oci.image.index.v1+json",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"  Tag '{tag}' pushed: {resp.status}")
            hdr = resp.headers.get("Docker-Content-Digest", "")
            if hdr:
                print(f"  Digest: {hdr}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        print(f"  HTTP {e.code}: {detail}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Push multi-arch manifest list")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    auth = base64.b64encode(f"{args.user}:{args.password}".encode()).decode()

    platforms = [
        ("amd64", f"{args.tag}-amd64"),
        ("arm64", f"{args.tag}-arm64"),
    ]

    manifests = []
    for arch, arch_tag in platforms:
        full_image = f"{args.registry}/{args.repo}:{arch_tag}"
        print(f"Getting digest for {full_image}...")
        local_digest = get_image_digest(full_image)
        print(f"  Local digest: {local_digest}")

        print(f"  Fetching manifest from registry...")
        digest, size = fetch_manifest_from_registry(
            args.registry, args.repo, arch_tag, auth,
        )
        print(f"  Registry digest: {digest}")
        print(f"  Size: {size}")
        manifests.append(
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": digest,
                "size": size,
                "platform": {"architecture": arch, "os": "linux"},
            }
        )

    index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": manifests,
    }
    body = json.dumps(index).encode()

    print(f"Pushing manifest list to {args.registry}/{args.repo}:{args.tag}...")
    push_manifest(args.registry, args.repo, args.tag, auth, body)
    print("Done.")


if __name__ == "__main__":
    main()
