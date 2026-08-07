"""Tests for FindMy accessory key derivation and MAC matching."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from custom_components.bermuda.bermuda_findmy import (
    KEY_TYPE_PRIMARY,
    BermudaFindMyManager,
    FindMyAccessoryKeys,
    FindMyKeyError,
    mac_from_public_key,
)

# A synthetic accessory. The key material is arbitrary but well-formed; the
# derived values below were produced by this implementation and cross-checked
# against the FindMy.py reference implementation (findmy.accessory.FindMyAccessory),
# which is the upstream authority for the key schedule.
PAIRED_AT = "2024-01-01T00:00:00+00:00"
ACCESSORY_JSON = json.dumps(
    {
        "type": "accessory",
        "master_key": "00" * 28,
        "skn": "11" * 32,
        "sks": "22" * 32,
        "paired_at": PAIRED_AT,
        "name": "Test Tag",
        "model": "Test Model",
        "identifier": "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
        "serial_number": "TESTSERIAL01",
        "alignment_date": PAIRED_AT,
        "alignment_index": 0,
    }
)


def _accessory() -> FindMyAccessoryKeys:
    return FindMyAccessoryKeys.from_json(ACCESSORY_JSON)


def test_mac_from_public_key_sets_random_static_bits():
    """The advertised address is the pubkey prefix with the top two bits set."""
    assert mac_from_public_key(bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]) + b"\x00" * 22) == "c0:11:22:33:44:55"
    assert mac_from_public_key(bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]) + b"\x00" * 22) == "d2:34:56:78:9a:bc"
    # Already-set bits must be preserved, not toggled.
    assert mac_from_public_key(bytes([0x3F, 0x00, 0x00, 0x00, 0x00, 0x00]) + b"\x00" * 22) == "ff:00:00:00:00:00"


def test_parse_rejects_bad_input():
    """Malformed key material is reported, not silently accepted."""
    with pytest.raises(FindMyKeyError, match="Not valid JSON"):
        FindMyAccessoryKeys.from_json("{nope")
    with pytest.raises(FindMyKeyError, match="Missing required field"):
        FindMyAccessoryKeys.from_json('{"type": "accessory", "skn": "00"}')
    with pytest.raises(FindMyKeyError, match="master_key must be 28 bytes"):
        FindMyAccessoryKeys.from_json(json.dumps({**json.loads(ACCESSORY_JSON), "master_key": "0011"}))
    with pytest.raises(FindMyKeyError, match="not valid hex"):
        FindMyAccessoryKeys.from_json(json.dumps({**json.loads(ACCESSORY_JSON), "skn": "zz" * 32}))
    with pytest.raises(FindMyKeyError, match="Unsupported accessory type"):
        FindMyAccessoryKeys.from_json(json.dumps({**json.loads(ACCESSORY_JSON), "type": "custom_rolling_key"}))


def test_key_schedule_is_deterministic_and_index_dependent():
    """Each index yields a distinct, reproducible address."""
    acc1, acc2 = _accessory(), _accessory()
    macs = [acc1._mac_at(i, KEY_TYPE_PRIMARY) for i in range(8)]  # noqa: SLF001
    assert macs == [acc2._mac_at(i, KEY_TYPE_PRIMARY) for i in range(8)]  # noqa: SLF001
    assert len(set(macs)) == len(macs), "indices must not collide"
    for mac in macs:
        assert int(mac[:2], 16) & 0b11000000 == 0b11000000


def test_sk_chain_rewind_matches_forward_walk():
    """
    Rewinding to a checkpoint must give the same result as walking forward.

    The chain is sequential and we keep only sparse checkpoints, so a backwards
    lookup rewinds - this guards that optimisation.
    """
    acc = _accessory()
    forward = [acc._mac_at(i, KEY_TYPE_PRIMARY) for i in range(0, 40)]  # noqa: SLF001
    # Jump far ahead to move the head, then walk back over the same range.
    acc._mac_at(3000, KEY_TYPE_PRIMARY)  # noqa: SLF001
    acc._mac_cache.clear()  # noqa: SLF001
    backward = [acc._mac_at(i, KEY_TYPE_PRIMARY) for i in range(0, 40)]  # noqa: SLF001
    assert forward == backward


def test_index_window_tracks_elapsed_time():
    """The key rolls every 15 minutes, so the window's top tracks the clock."""
    acc = _accessory()
    paired = datetime.fromisoformat(PAIRED_AT)
    assert acc.max_index(paired) == 0
    assert acc.max_index(paired + timedelta(minutes=15)) == 1
    assert acc.max_index(paired + timedelta(hours=1)) == 4
    assert acc.max_index(paired + timedelta(days=1)) == 96


def test_index_window_is_capped_for_unaligned_accessory():
    """
    An accessory never sighted must not produce an unbounded search window.

    A tag paired years ago would otherwise mean hundreds of thousands of
    elliptic curve operations at startup.
    """
    acc = _accessory()
    now = datetime.fromisoformat(PAIRED_AT) + timedelta(days=365 * 3)
    bottom, top = acc.index_window(now)
    assert top - bottom <= 2880
    assert top == acc.max_index(now) + 2


def test_alignment_collapses_window_and_ignores_regressions():
    """A confirmed sighting narrows the search; stale data must not widen it."""
    acc = _accessory()
    # Well inside the cap, so the window really does start at the pairing index.
    now = datetime.fromisoformat(PAIRED_AT) + timedelta(days=10)
    assert acc.index_window(now)[0] == 0

    assert acc.update_alignment(now, 960) is True
    assert acc.index_window(now)[0] == 960

    # Older observation, and a backwards index, are both ignored.
    assert acc.update_alignment(now - timedelta(days=1), 10) is False
    assert acc.update_alignment(now, 5) is False
    assert acc.alignment_index == 960


def test_manager_matches_generated_macs():
    """The manager resolves an address the accessory would actually advertise."""
    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())
    now = datetime.fromisoformat(PAIRED_AT) + timedelta(hours=1)

    manager.build_table(now)
    expected_index = acc.max_index(now)
    mac = acc._mac_at(expected_index, KEY_TYPE_PRIMARY)  # noqa: SLF001

    match = manager.check_mac(mac)
    assert match is not None
    assert match.accessory_id == acc.address
    assert match.index == expected_index
    assert match.key_type == KEY_TYPE_PRIMARY

    assert manager.check_mac("00:11:22:33:44:55") is None


def test_manager_sighting_updates_alignment():
    """Recording a sighting narrows that accessory's window."""
    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())
    now = datetime.fromisoformat(PAIRED_AT) + timedelta(days=10)
    manager.build_table(now)

    mac = acc._mac_at(acc.max_index(now), KEY_TYPE_PRIMARY)  # noqa: SLF001
    match = manager.check_mac(mac)
    assert match is not None

    assert manager.note_sighting(match, now) is True
    assert acc.alignment_index == match.index
    # Repeating the same sighting is not a change worth persisting.
    assert manager.note_sighting(match, now) is False


def test_roundtrip_through_config_entry_storage():
    """Accessories survive a save/load cycle with their alignment intact."""
    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())
    now = datetime.fromisoformat(PAIRED_AT) + timedelta(days=5)
    acc.update_alignment(now, 480)

    restored = BermudaFindMyManager()
    restored.load(manager.dump())

    assert list(restored.accessories) == [acc.address]
    new_acc = restored.accessories[acc.address]
    assert new_acc.alignment_index == 480
    assert new_acc.alignment_date == acc.alignment_date
    assert new_acc.friendly_name == "Test Tag"
    # Same keys must still produce the same schedule after a round trip.
    assert new_acc._mac_at(500, KEY_TYPE_PRIMARY) == acc._mac_at(500, KEY_TYPE_PRIMARY)  # noqa: SLF001


def test_readding_accessory_preserves_alignment():
    """Re-pasting the same JSON must not throw away a narrowed window."""
    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())
    now = datetime.fromisoformat(PAIRED_AT) + timedelta(days=5)
    acc.update_alignment(now, 480)

    reAdded = manager.add_accessory(_accessory())
    assert reAdded.alignment_index == 480


def test_load_discards_unreadable_entries():
    """One corrupt stored accessory must not take out the others."""
    manager = BermudaFindMyManager()
    manager.load([{"master_key": "oops"}, json.loads(ACCESSORY_JSON)])
    assert len(manager.accessories) == 1


def test_dump_excludes_nothing_needed_for_restore():
    """Stored form must carry everything required to rebuild the schedule."""
    acc = _accessory()
    data = acc.to_dict()
    for required in ("master_key", "skn", "sks", "paired_at", "alignment_index", "alignment_date"):
        assert required in data


def test_diagnostics_omit_secrets():
    """Diagnostics are shared in bug reports - they must not carry key material."""
    manager = BermudaFindMyManager()
    manager.add_accessory(_accessory())
    blob = json.dumps(manager.async_diagnostics_no_redactions())
    assert "00" * 28 not in blob
    assert "11" * 32 not in blob
    assert "22" * 32 not in blob


def test_needs_refresh_follows_key_interval():
    """The table is rebuilt once per key interval, not on every cycle."""
    manager = BermudaFindMyManager()
    manager.add_accessory(_accessory())
    now = datetime.now(UTC)

    assert manager.needs_refresh(now) is True
    manager.build_table(now)
    assert manager.needs_refresh(now) is False
    assert manager.needs_refresh(now + timedelta(minutes=14)) is False
    assert manager.needs_refresh(now + timedelta(minutes=15)) is True


def test_remove_accessory():
    """Removing an accessory drops it from matching."""
    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())
    now = datetime.fromisoformat(PAIRED_AT)
    manager.build_table(now)
    mac = acc._mac_at(0, KEY_TYPE_PRIMARY)  # noqa: SLF001
    assert manager.check_mac(mac) is not None

    assert manager.remove_accessory(acc.address) is True
    assert manager.remove_accessory(acc.address) is False
    manager.build_table(now)
    assert manager.check_mac(mac) is None
