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
import hashlib
import json
import subprocess
import sys
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
    # Returns e.g. "registry/repo:tag@sha256:abc..."
    if "@" in out:
        return out.split("@", 1)[1]
    # Fallback: get the ID
    img_id = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Id}}", image], text=True
    ).strip()
    return f"sha256:{img_id}"


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

    import base64

    auth = base64.b64encode(f"{args.user}:{args.password}".encode()).decode()

    platforms = [
        ("amd64", f"{args.registry}/{args.repo}:{args.tag}-amd64"),
        ("arm64", f"{args.registry}/{args.repo}:{args.tag}-arm64"),
    ]

    manifests = []
    for arch, image in platforms:
        print(f"Getting digest for {image}...")
        digest = get_image_digest(image)
        print(f"  {digest}")
        manifests.append(
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": digest,
                "size": 2582,  # typical OCI manifest size
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
