"""
BermudaFindMyManager for tracking Apple FindMy accessories (AirTags etc).

FindMy accessories rotate their BLE MAC address every 15 minutes, in lockstep
with the advertising key they broadcast. Unlike IRK devices (see bermuda_irk.py)
the rotation is *not* resolvable from the address itself - it is generated from
a key schedule seeded at pairing time. That means we cannot test an address for
membership; we have to derive the schedule forward and look the address up.

Fortunately the address is a pure function of the advertising key:

    MAC = public_key[0:6], with the top two bits of byte 0 set to 1

...so a lookup table of "MACs this accessory could currently be using" is enough
to identify it, and works for both `separated` adverts (which carry 22 bytes of
the key) and `nearby` adverts (which carry almost nothing but still use the
derived MAC). That matters: an AirTag sitting at home with its owner present is
in nearby state, so key-matching alone would never see it.

Key material comes from pairing and is supplied by the user as JSON (the format
emitted by the FindMy.py library's `FindMyAccessory.to_json()`).

The key schedule is implemented here directly against `cryptography` rather than
depending on the `findmy` package, which would pull in aiohttp, bleak, srp,
beautifulsoup4 and anisette for functionality we do not use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.x963kdf import X963KDF

from .const import (
    _LOGGER,
    FINDMY_ALIGNMENT_TRUST_INDICES,
    FINDMY_KEY_INTERVAL,
    FINDMY_LOOKAHEAD_INDICES,
    FINDMY_LOOKBEHIND_INDICES,
    FINDMY_MAX_UNALIGNED_INDICES,
    FINDMY_SECONDARY_INTERVAL,
    FINDMY_SK_CHECKPOINT_INTERVAL,
)

# Order of the NIST P-224 curve.
P224_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFF16A2E0B8F03E13DD29455C5C2A3D

KEY_TYPE_PRIMARY = "primary"
KEY_TYPE_SECONDARY = "secondary"


class FindMyKeyError(ValueError):
    """Raised when accessory key material is missing or malformed."""


def _x963_kdf(value: bytes, sharedinfo: bytes, length: int) -> bytes:
    """Single pass of the X9.63 KDF with SHA256, as used by the FindMy protocol."""
    return X963KDF(algorithm=hashes.SHA256(), sharedinfo=sharedinfo, length=length).derive(value)


def _derive_ps_key(master_key: bytes, sk: bytes) -> bytes:
    """
    Derive the private key for one time-period from the master key and that period's SK.

    Uses the SKN chain for primary keys, the SKS chain for secondary ones.
    """
    at = _x963_kdf(sk, b"diversify", 72)
    u = int.from_bytes(at[:36], "big") % (P224_N - 1) + 1
    v = int.from_bytes(at[36:], "big") % (P224_N - 1) + 1
    key = (u * int.from_bytes(master_key, "big") + v) % P224_N
    return key.to_bytes(28, "big")


def _public_key_bytes(private_key: bytes) -> bytes:
    """Get the 28-byte advertised public key (P-224 x coordinate) for a private key."""
    privkey = ec.derive_private_key(int.from_bytes(private_key, "big"), ec.SECP224R1())
    return privkey.public_key().public_numbers().x.to_bytes(28, "big")


def mac_from_public_key(public_key: bytes) -> str:
    """
    Derive the BLE address an accessory will advertise for a given public key.

    The top two bits of the first byte are set, making it a Random Static address.
    Returns lower-cased colon-separated form, matching Bermuda's internal convention.
    """
    first = public_key[0] | 0b11000000
    return ":".join(f"{b:02x}" for b in bytes([first]) + public_key[1:6])


@dataclass
class FindMyMacMatch:
    """The result of matching an observed MAC against a known accessory."""

    accessory_id: str
    index: int
    key_type: str


class FindMyAccessoryKeys:
    """
    The key schedule for a single FindMy accessory.

    Holds the pairing secrets and derives the set of MAC addresses the accessory
    could plausibly be advertising right now.
    """

    def __init__(
        self,
        *,
        master_key: bytes,
        skn: bytes,
        sks: bytes,
        paired_at: datetime,
        name: str | None = None,
        model: str | None = None,
        identifier: str | None = None,
        serial_number: str | None = None,
        alignment_date: datetime | None = None,
        alignment_index: int = 0,
    ) -> None:
        if len(master_key) != 28:
            msg = f"master_key must be 28 bytes, got {len(master_key)}"
            raise FindMyKeyError(msg)
        if len(skn) != 32:
            msg = f"skn must be 32 bytes, got {len(skn)}"
            raise FindMyKeyError(msg)
        if len(sks) != 32:
            msg = f"sks must be 32 bytes, got {len(sks)}"
            raise FindMyKeyError(msg)

        self._master_key = master_key
        self._skn = skn
        self._sks = sks
        self.paired_at = _ensure_aware(paired_at)
        self.name = name
        self.model = model
        self.serial_number = serial_number
        # The metadevice address. Stable for the life of the accessory.
        #
        # When the export doesn't carry an identifier we hash the master key rather
        # than slicing it: the identifier ends up in device names, logs and
        # diagnostics, so it must not be reversible into key material. Hashing keeps
        # it deterministic, so re-pasting the same keys still resolves to the same
        # metadevice instead of spawning a duplicate.
        self.identifier = identifier or hashlib.sha256(master_key).hexdigest()[:32]

        # Alignment is a single (date, index) fact and is read from the executor
        # thread while the event loop may be updating it - see update_alignment().
        # Keeping it in one tuple makes each read see a consistent pair.
        self._alignment: tuple[datetime, int] = (
            _ensure_aware(alignment_date) if alignment_date else self.paired_at,
            alignment_index,
        )

        # SK chain state. The chain is strictly sequential - sk[n] derives from
        # sk[n-1] - and an accessory paired years ago can be 180,000+ steps along
        # it. A KDF step is only ~3us, but we must never scan or store the whole
        # chain: we keep a moving head plus sparse checkpoints to rewind to.
        self._sk_head: dict[str, tuple[int, bytes]] = {KEY_TYPE_PRIMARY: (0, skn), KEY_TYPE_SECONDARY: (0, sks)}
        self._sk_checkpoints: dict[str, dict[int, bytes]] = {
            KEY_TYPE_PRIMARY: {0: skn},
            KEY_TYPE_SECONDARY: {0: sks},
        }
        self._mac_cache: dict[tuple[int, str], str] = {}

    @property
    def alignment_date(self) -> datetime:
        """When we last confirmed this accessory's position in its key schedule."""
        return self._alignment[0]

    @property
    def alignment_index(self) -> int:
        """The key index we last confirmed this accessory was using."""
        return self._alignment[1]

    @property
    def address(self) -> str:
        """The metadevice address Bermuda will track this accessory under."""
        return f"findmy_{self.identifier.replace('-', '').lower()}"

    @property
    def friendly_name(self) -> str:
        """Best available human-readable name."""
        return self.name or self.model or self.serial_number or self.identifier

    def max_index(self, now: datetime | None = None) -> int:
        """
        The highest key index the accessory could have reached by `now`.

        The key rolls at most once per interval, so this is an upper bound - the
        accessory may have rolled more slowly, or not at all if powered off.

        A recent sighting is trusted on its own, which keeps the window to a
        handful of indices. Once the alignment goes stale we can no longer be sure
        it is right - it may be a stale value reloaded from the Store - so the
        pairing-derived bound is taken as well, whichever is higher.

        Both halves matter. Anchoring solely on the alignment is a trap: a too-low
        alignment gives a too-low ceiling, the accessory's real index climbs past
        it, and since update_alignment() never moves backwards nothing can raise
        the ceiling again - a permanent lockout. But always taking the pairing
        bound is the opposite trap: an accessory that was powered off has a real
        index behind wall-clock, so the pairing bound stays permanently above it
        and the window never collapses after a sighting. Trusting a fresh
        alignment and widening only a stale one avoids both.
        """
        now = now or datetime.now(UTC)
        align_date, align_index = self._alignment
        elapsed = 0
        if now > align_date:
            elapsed = int((now - align_date) // FINDMY_KEY_INTERVAL)
        from_alignment = align_index + elapsed
        if elapsed <= FINDMY_ALIGNMENT_TRUST_INDICES:
            return from_alignment
        from_pairing = 0
        if now > self.paired_at:
            from_pairing = int((now - self.paired_at) // FINDMY_KEY_INTERVAL)
        return max(from_alignment, from_pairing)

    def index_window(self, now: datetime | None = None) -> tuple[int, int]:
        """
        The range of indices worth generating MACs for.

        Without alignment the lower bound is whatever we last confirmed (initially
        the pairing index, 0), which for a long-paired accessory is a very wide
        window. We cap it: an accessory that is actually present and advertising
        will be near the top of the range, and a single sighting collapses the
        window via update_alignment().
        """
        top = self.max_index(now) + FINDMY_LOOKAHEAD_INDICES
        floor = max(0, self.alignment_index - FINDMY_LOOKBEHIND_INDICES)
        bottom = max(floor, top - FINDMY_MAX_UNALIGNED_INDICES)
        return bottom, top

    def _sk_at(self, ind: int, key_type: str) -> bytes:
        """
        Walk the SK chain to the given index.

        Callers iterate ascending, so the common case is a one-step advance of the
        head. Going backwards rewinds to the nearest checkpoint at or below the
        target and walks forward from there.
        """
        head_ind, head_sk = self._sk_head[key_type]
        if ind == head_ind:
            return head_sk

        checkpoints = self._sk_checkpoints[key_type]
        if ind > head_ind:
            start, sk = head_ind, head_sk
        else:
            start = max((i for i in checkpoints if i <= ind), default=0)
            sk = checkpoints[start]

        for cur in range(start + 1, ind + 1):
            sk = _x963_kdf(sk, b"update", 32)
            if cur % FINDMY_SK_CHECKPOINT_INTERVAL == 0:
                checkpoints[cur] = sk

        if ind > head_ind:
            self._sk_head[key_type] = (ind, sk)
        return sk

    def _mac_at(self, ind: int, key_type: str) -> str:
        """Derive (and cache) the MAC address for one index and key type."""
        if (mac := self._mac_cache.get((ind, key_type))) is not None:
            return mac
        sk = self._sk_at(ind, key_type)
        privkey = _derive_ps_key(self._master_key, sk)
        mac = mac_from_public_key(_public_key_bytes(privkey))
        self._mac_cache[(ind, key_type)] = mac
        return mac

    def macs_for_window(self, now: datetime | None = None) -> dict[str, FindMyMacMatch]:
        """
        Build the lookup table of MACs this accessory might currently advertise.

        Includes the primary key for each index in the window, plus both candidate
        secondary keys - the secondary chain advances once per FINDMY_SECONDARY_INTERVAL
        primary steps, but the exact rollover point depends on the pairing time of day,
        so we cover both possibilities.
        """
        bottom, top = self.index_window(now)
        macs: dict[str, FindMyMacMatch] = {}
        for ind in range(bottom, top + 1):
            macs[self._mac_at(ind, KEY_TYPE_PRIMARY)] = FindMyMacMatch(self.address, ind, KEY_TYPE_PRIMARY)

        # Secondary keys are indexed by primary_index // interval, so the window
        # collapses to a handful of distinct values.
        for sec_ind in {i // FINDMY_SECONDARY_INTERVAL for i in range(bottom, top + 1)}:
            for offset in (1, 2):
                mac = self._mac_at(sec_ind + offset, KEY_TYPE_SECONDARY)
                macs.setdefault(mac, FindMyMacMatch(self.address, sec_ind + offset, KEY_TYPE_SECONDARY))

        self._prune_caches(bottom)
        return macs

    def _prune_caches(self, bottom: int) -> None:
        """
        Drop cached MACs well below the current window so we don't grow forever.

        Only worth doing once the cache has actually grown: the scan is O(cache),
        and running it on every build spent more time looking for stale entries
        than deriving the addresses. Only primary indices track the window -
        secondary ones are few and bounded, so they are never evicted.
        """
        if len(self._mac_cache) <= FINDMY_MAX_UNALIGNED_INDICES:
            return
        cutoff = bottom - FINDMY_LOOKBEHIND_INDICES
        stale = [key for key in self._mac_cache if key[1] == KEY_TYPE_PRIMARY and key[0] < cutoff]
        for key in stale:
            del self._mac_cache[key]

    def update_alignment(self, seen_at: datetime, index: int) -> bool:
        """
        Record a confirmed sighting, narrowing the search window.

        Ignores anything that moves backwards, since we may be handed conflicting
        observations and a stable, most-recent value is what we want.
        Returns True if the alignment changed (ie, worth persisting).

        Called from the event loop while the executor thread may be reading the
        alignment, so the update is published as a single tuple assignment.
        """
        seen_at = _ensure_aware(seen_at)
        current_date, current_index = self._alignment
        if seen_at < current_date or index < current_index:
            return False
        if index == current_index and seen_at == current_date:
            return False
        _LOGGER.debug(
            "FindMy %s: alignment updated to index %d (was %d)",
            self.friendly_name,
            index,
            current_index,
        )
        self._alignment = (seen_at, index)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage in the config entry."""
        return {
            "master_key": self._master_key.hex(),
            "skn": self._skn.hex(),
            "sks": self._sks.hex(),
            "paired_at": self.paired_at.isoformat(),
            "name": self.name,
            "model": self.model,
            "identifier": self.identifier,
            "serial_number": self.serial_number,
            "alignment_date": self.alignment_date.isoformat(),
            "alignment_index": self.alignment_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FindMyAccessoryKeys:
        """
        Build from a stored dict or user-supplied FindMy.py accessory JSON.

        Accepts both, since the field names are identical - which is the point:
        users can paste the exported file unmodified.
        """
        try:
            master_key = _unhex(data["master_key"], "master_key")
            skn = _unhex(data["skn"], "skn")
            sks = _unhex(data["sks"], "sks")
            paired_at = parse_findmy_datetime(data["paired_at"])
        except KeyError as err:
            msg = f"Missing required field: {err.args[0]}"
            raise FindMyKeyError(msg) from err

        alignment_date = data.get("alignment_date")
        return cls(
            master_key=master_key,
            skn=skn,
            sks=sks,
            paired_at=paired_at,
            name=data.get("name"),
            model=data.get("model"),
            identifier=data.get("identifier"),
            serial_number=data.get("serial_number"),
            alignment_date=parse_findmy_datetime(alignment_date) if alignment_date else None,
            alignment_index=data.get("alignment_index") or 0,
        )

    @classmethod
    def from_json(cls, raw: str) -> FindMyAccessoryKeys:
        """Build from pasted JSON text, with errors suitable for showing a user."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            msg = f"Not valid JSON: {err}"
            raise FindMyKeyError(msg) from err
        if not isinstance(data, dict):
            msg = "Expected a JSON object describing a single accessory"
            raise FindMyKeyError(msg)
        if data.get("type") not in (None, "accessory"):
            msg = f"Unsupported accessory type '{data.get('type')}' - expected 'accessory'"
            raise FindMyKeyError(msg)
        return cls.from_dict(data)


def _unhex(value: str, field_name: str) -> bytes:
    """Decode a hex field, with a message naming the offending field."""
    try:
        return bytes.fromhex(value)
    except (ValueError, TypeError) as err:
        msg = f"Field '{field_name}' is not valid hex"
        raise FindMyKeyError(msg) from err


def parse_findmy_datetime(value: str) -> datetime:
    """Parse an ISO timestamp, assuming UTC if no timezone is given."""
    try:
        return _ensure_aware(datetime.fromisoformat(value))
    except (ValueError, TypeError) as err:
        msg = f"Could not parse timestamp '{value}'"
        raise FindMyKeyError(msg) from err


def _ensure_aware(value: datetime) -> datetime:
    """Treat naive timestamps as UTC - FindMy exports are UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


@dataclass
class _TableState:
    """The current MAC lookup table and when it was built."""

    macs: dict[str, FindMyMacMatch] = field(default_factory=dict)
    built_at: datetime | None = None


class BermudaFindMyManager:
    """
    Manager for FindMy accessory tracking.

    - add_accessory() as each accessory is configured
    - check_mac() for every observed address (cheap dict lookup)
    - async_refresh_table() periodically, to roll the window forward
    """

    def __init__(self) -> None:
        self._accessories: dict[str, FindMyAccessoryKeys] = {}
        self._table = _TableState()
        self._dirty = True

    @property
    def accessories(self) -> dict[str, FindMyAccessoryKeys]:
        """All configured accessories, keyed by metadevice address."""
        return self._accessories

    def add_accessory(self, accessory: FindMyAccessoryKeys) -> FindMyAccessoryKeys:
        """Add or replace an accessory. Returns the stored instance."""
        existing = self._accessories.get(accessory.address)
        if existing is not None:
            # Preserve hard-won alignment across a re-paste of the same keys.
            accessory.update_alignment(existing.alignment_date, existing.alignment_index)
        self._accessories[accessory.address] = accessory
        self._dirty = True
        _LOGGER.debug("FindMy accessory registered: %s (%s)", accessory.friendly_name, accessory.address)
        return accessory

    def remove_accessory(self, address: str) -> bool:
        """Remove an accessory by metadevice address."""
        if self._accessories.pop(address, None) is None:
            return False
        # Drop the table too, so its addresses stop matching immediately rather
        # than until the next rebuild.
        self._table = _TableState()
        self._dirty = True
        return True

    def load(self, stored: list[dict[str, Any]]) -> None:
        """Restore accessories from config entry data."""
        for item in stored:
            try:
                self.add_accessory(FindMyAccessoryKeys.from_dict(item))
            except FindMyKeyError:
                _LOGGER.exception("Discarding unreadable FindMy accessory entry")

    def dump(self) -> list[dict[str, Any]]:
        """Serialise all accessories for saving to the config entry."""
        return [acc.to_dict() for acc in self._accessories.values()]

    def check_mac(self, address: str) -> FindMyMacMatch | None:
        """Look up an observed address. Cheap - a dict hit on a precomputed table."""
        return self._table.macs.get(address)

    def needs_refresh(self, now: datetime | None = None) -> bool:
        """Whether the lookup table has aged out of its interval."""
        if self._dirty or self._table.built_at is None:
            return True
        now = now or datetime.now(UTC)
        return now - self._table.built_at >= FINDMY_KEY_INTERVAL

    def build_table(self, now: datetime | None = None) -> dict[str, FindMyMacMatch]:
        """
        Rebuild the MAC lookup table.

        This is the expensive call - one elliptic curve operation per candidate
        index - so it belongs in an executor, and is only run once per key interval.
        """
        now = now or datetime.now(UTC)
        # Clear the flag first, and iterate a snapshot. This runs in an executor
        # while the event loop may be adding or removing accessories or recording
        # sightings: iterating the live dict can raise "changed size during
        # iteration", and clearing the flag afterwards would discard a change that
        # arrived mid-build, hiding it until the next interval.
        self._dirty = False
        accessories = list(self._accessories.values())
        macs: dict[str, FindMyMacMatch] = {}
        for accessory in accessories:
            macs.update(accessory.macs_for_window(now))
        self._table = _TableState(macs=macs, built_at=now)
        _LOGGER.debug(
            "FindMy MAC table rebuilt: %d addresses across %d accessories",
            len(macs),
            len(accessories),
        )
        return macs

    def note_sighting(self, match: FindMyMacMatch, seen_at: datetime | None = None) -> bool:
        """
        Record that a matched MAC was actually observed.

        Narrows that accessory's search window. Returns True if the alignment
        changed and should be persisted.
        """
        accessory = self._accessories.get(match.accessory_id)
        if accessory is None or match.key_type != KEY_TYPE_PRIMARY:
            # Secondary indices are on a different scale; only primary tells us
            # where we are in the schedule.
            return False
        previous_index = accessory.alignment_index
        changed = accessory.update_alignment(seen_at or datetime.now(UTC), match.index)
        # Only a *moved index* changes which addresses we should be looking for.
        # Every advert refreshes the timestamp, so dirtying the table on that would
        # rebuild it on every cycle instead of once per key interval - the caller
        # still gets True, because the newer timestamp is worth persisting.
        if changed and accessory.alignment_index != previous_index:
            self._dirty = True
        return changed

    def async_diagnostics_no_redactions(self) -> dict[str, Any]:
        """Diagnostic info. Secrets are deliberately excluded, not merely redacted."""
        now = datetime.now(UTC)
        return {
            "accessories": {
                acc.address: {
                    "name": acc.friendly_name,
                    "model": acc.model,
                    "identifier": acc.identifier,
                    "alignment_index": acc.alignment_index,
                    "alignment_date": acc.alignment_date.isoformat(),
                    "index_window": list(acc.index_window(now)),
                }
                for acc in self._accessories.values()
            },
            "table_size": len(self._table.macs),
            "table_built_at": self._table.built_at.isoformat() if self._table.built_at else None,
        }
