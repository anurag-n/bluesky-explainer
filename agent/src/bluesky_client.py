"""
Bluesky AT Protocol client for fetching posts and their embedded content.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from atproto import Client


@dataclass
class BlueskyImage:
    """Represents an image attached to a Bluesky post."""

    alt_text: str
    blob_cid: str
    mime_type: str
    image_bytes: Optional[bytes] = None


@dataclass
class BlueskyPost:
    """Structured representation of a fetched Bluesky post."""

    uri: str
    cid: str
    text: str
    author_handle: str
    created_at: str
    images: list[BlueskyImage] = field(default_factory=list)
    external_url: Optional[str] = None
    external_title: Optional[str] = None
    external_description: Optional[str] = None
    quoted_post_uri: Optional[str] = None


class BlueskyClient:
    """
    Wraps the atproto Client to fetch posts and resolve Bluesky URLs.

    Authenticates with a Bluesky handle and app password (not the main account
    password). Authentication is required to fetch image blobs.
    """

    BSKY_URL_PATTERN = re.compile(
        r"https://bsky\.app/profile/(?P<handle>[^/]+)/post/(?P<rkey>[a-zA-Z0-9]+)"
    )

    def __init__(self, handle: str, app_password: str) -> None:
        self._client = Client()
        self._client.login(handle, app_password)

    def parse_post_url(self, url: str) -> tuple[str, str]:
        """
        Parse a bsky.app URL into (handle, rkey).

        Raises:
            ValueError: If URL does not match the expected pattern.
        """
        match = self.BSKY_URL_PATTERN.match(url.strip())
        if not match:
            raise ValueError(f"Invalid Bluesky post URL: {url!r}")
        return match.group("handle"), match.group("rkey")

    def resolve_did(self, handle: str) -> str:
        """Resolve a Bluesky handle to its DID."""
        profile = self._client.app.bsky.actor.get_profile({"actor": handle})
        return profile.did

    def build_at_uri(self, did: str, rkey: str) -> str:
        """Construct an AT URI from a DID and record key."""
        return f"at://{did}/app.bsky.feed.post/{rkey}"

    def fetch_post(self, url: str) -> BlueskyPost:
        """
        Main entry point: fetch a Bluesky post from its bsky.app URL.

        Resolves the handle to a DID, builds the AT URI, fetches the post
        record, and parses all embedded content (images, URLs, quoted posts).

        Args:
            url: Full bsky.app post URL.

        Returns:
            A populated BlueskyPost dataclass.

        Raises:
            ValueError: If the URL is malformed or the post is not found.
        """
        handle, rkey = self.parse_post_url(url)
        did = self.resolve_did(handle)
        at_uri = self.build_at_uri(did, rkey)
        return self._fetch_by_at_uri(at_uri, did)

    def fetch_post_by_uri(self, at_uri: str) -> BlueskyPost:
        """
        Fetch a post directly by its AT URI.

        Used when resolving quoted post references where only the AT URI
        is available (not a bsky.app URL).
        """
        # Extract the DID from the AT URI (at://did:plc:.../...)
        did = at_uri.split("/")[2]
        return self._fetch_by_at_uri(at_uri, did)

    def _fetch_by_at_uri(self, at_uri: str, did: str) -> BlueskyPost:
        """Internal: fetch and parse a post given its AT URI and author DID."""
        response = self._client.app.bsky.feed.get_posts({"uris": [at_uri]})
        if not response.posts:
            raise ValueError(f"Post not found: {at_uri}")

        post_view = response.posts[0]
        record = post_view.record

        post = BlueskyPost(
            uri=at_uri,
            cid=post_view.cid,
            text=record.text or "",
            author_handle=post_view.author.handle,
            created_at=str(record.created_at),
        )

        if post_view.embed:
            post = self._parse_embed(post, post_view.embed, did)

        return post

    def _parse_embed(self, post: BlueskyPost, embed, did: str) -> BlueskyPost:
        """
        Populate post fields from the embed object.

        Handles three embed types:
        - Images: fetches blob bytes for vision LLM calls
        - External link card: stores URL, title, description
        - Quoted post (record): stores the quoted post's AT URI
        """
        embed_type = type(embed).__name__

        if hasattr(embed, "images"):
            for img_view in embed.images:
                blob_cid = ""
                if hasattr(img_view, "image") and hasattr(img_view.image, "ref"):
                    blob_cid = str(img_view.image.ref.link)
                image = BlueskyImage(
                    alt_text=getattr(img_view, "alt", "") or "",
                    blob_cid=blob_cid,
                    mime_type=getattr(getattr(img_view, "image", None), "mime_type", "image/jpeg") or "image/jpeg",
                )
                if blob_cid:
                    image.image_bytes = self._fetch_blob(did, blob_cid)
                post.images.append(image)

        elif hasattr(embed, "external"):
            ext = embed.external
            post.external_url = getattr(ext, "uri", None)
            post.external_title = getattr(ext, "title", None)
            post.external_description = getattr(ext, "description", None)

        elif hasattr(embed, "record"):
            quoted = embed.record
            if hasattr(quoted, "uri"):
                post.quoted_post_uri = quoted.uri

        return post

    def _fetch_blob(self, did: str, cid: str) -> Optional[bytes]:
        """
        Download raw image blob bytes from the Bluesky CDN.

        Returns None if the fetch fails (e.g. blob not found, network error).
        """
        try:
            response = self._client.com.atproto.sync.get_blob({"did": did, "cid": cid})
            return response
        except Exception:
            return None
