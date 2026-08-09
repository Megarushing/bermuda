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


def test_options_flow_handler_initialises_errors():
    """
    The options flow must define _errors before any step renders a form.

    Regression test: async_step_findmy passed errors=self._errors, but _errors
    was only initialised on the config flow handler, not the options flow one,
    so opening the FindMy menu raised AttributeError.
    """
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.bermuda.config_flow import BermudaOptionsFlowHandler
    from custom_components.bermuda.const import DOMAIN

    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    handler = BermudaOptionsFlowHandler(entry)

    assert handler._errors == {}  # noqa: SLF001


def test_identifier_fallback_does_not_expose_key_material():
    """
    An accessory with no identifier must not derive its address from the key.

    Regression test: the fallback was hexlify(master_key[:16]), putting 16 of the
    28 master key bytes into the metadevice address - which ends up in device
    names, logs and diagnostics. Hashing keeps the address deterministic (so
    re-pasting the same keys resolves to the same metadevice) without being
    reversible into key material.
    """
    raw = json.loads(ACCESSORY_JSON)
    del raw["identifier"]
    master_key_hex = raw["master_key"]

    acc = FindMyAccessoryKeys.from_json(json.dumps(raw))

    assert master_key_hex[:32] not in acc.address
    assert acc.address.startswith("findmy_")
    # Deterministic: the same keys must not spawn a second metadevice.
    assert FindMyAccessoryKeys.from_json(json.dumps(raw)).address == acc.address


def test_alignment_persists_without_touching_the_config_entry():
    """
    Alignment must be saved to its own Store, never to the config entry.

    Regression test for the reload loop: __init__ registers
    entry.add_update_listener(async_reload_entry), so *any* config entry write
    reloads the integration. Alignment updates on every sighting, so persisting
    it via async_update_entry reloaded Bermuda constantly, tearing down the
    FindMy metadevices it had just built and leaving their entities
    permanently unavailable.
    """
    from types import SimpleNamespace

    from custom_components.bermuda.coordinator import BermudaDataUpdateCoordinator

    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())

    saved: dict = {}

    def _delay_save(func, delay):
        saved["payload"] = func()
        saved["delay"] = delay

    def _explode(*_args, **_kwargs):
        msg = "alignment must not be written to the config entry - it triggers a reload"
        raise AssertionError(msg)

    coordinator = SimpleNamespace(
        findmy_manager=manager,
        _findmy_store=SimpleNamespace(async_delay_save=_delay_save),
        hass=SimpleNamespace(config_entries=SimpleNamespace(async_update_entry=_explode)),
    )
    coordinator._findmy_alignment_data = lambda: BermudaDataUpdateCoordinator._findmy_alignment_data(coordinator)  # noqa: SLF001

    BermudaDataUpdateCoordinator.async_save_findmy_alignment(coordinator)

    assert saved["payload"] == {
        acc.address: {
            "alignment_index": acc.alignment_index,
            "alignment_date": acc.alignment_date.isoformat(),
        }
    }
    # And the payload is runtime state only - no key material rides along.
    blob = json.dumps(saved["payload"])
    assert "00" * 28 not in blob
    assert "11" * 32 not in blob
    assert "22" * 32 not in blob


def test_removed_accessory_drops_out_of_the_alignment_payload():
    """Removing an accessory must clear it from the Store, not orphan its index."""
    from types import SimpleNamespace

    from custom_components.bermuda.coordinator import BermudaDataUpdateCoordinator

    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())
    coordinator = SimpleNamespace(findmy_manager=manager)

    payload = BermudaDataUpdateCoordinator._findmy_alignment_data(coordinator)  # noqa: SLF001
    assert acc.address in payload

    manager.remove_accessory(acc.address)
    assert BermudaDataUpdateCoordinator._findmy_alignment_data(coordinator) == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_removing_an_accessory_triggers_an_alignment_save():
    """
    The removal step must queue a Store save.

    Regression test: the payload is rebuilt from the live accessory list, but
    nothing else triggers a save once an accessory stops being sighted - so
    without this call a removed accessory's alignment index lingered in
    .storage indefinitely, forever if it was the last one.
    """
    from types import SimpleNamespace

    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.bermuda.config_flow import BermudaOptionsFlowHandler
    from custom_components.bermuda.const import DOMAIN

    manager = BermudaFindMyManager()
    acc = manager.add_accessory(_accessory())

    saves: list[bool] = []
    coordinator = SimpleNamespace(
        findmy_manager=manager,
        async_save_findmy_alignment=lambda: saves.append(True),
    )

    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.runtime_data = SimpleNamespace(coordinator=coordinator)
    handler = BermudaOptionsFlowHandler(entry)
    # OptionsFlow.config_entry resolves the entry by id off hass; the id comes
    # from the flow's `handler` attribute.
    handler.handler = entry.entry_id
    handler.hass = SimpleNamespace(
        config_entries=SimpleNamespace(
            async_update_entry=lambda *a, **k: None,
            # OptionsFlow.config_entry resolves through the manager.
            async_get_known_entry=lambda _entry_id: entry,
        ),
    )

    # Stop after the removal branch; the follow-on menu step needs a real flow.
    async def _skip_menu(user_input=None):
        return None

    handler.async_step_findmy = _skip_menu

    await handler.async_step_findmy_remove({"remove": [acc.address]})

    assert saves == [True], "removal must queue an alignment save"
    assert manager.accessories == {}

    # A no-op removal must not churn the Store.
    saves.clear()
    await handler.async_step_findmy_remove({"remove": ["findmy_nosuchthing"]})
    assert saves == []


def test_alignment_writes_cannot_be_starved_by_continuous_sightings():
    """
    Queueing an alignment save must be throttled, not merely debounced.

    Regression test: Store.async_delay_save is a resetting debounce with no max
    wait - each call pushes its timer forward, and a timer firing early
    reschedules itself rather than writing (see
    homeassistant/helpers/storage.py::_async_schedule_callback_delayed_write).
    Alignment changes on essentially every sighting, so queueing a save every
    coordinator cycle meant a tag that stayed in view starved the write for as
    long as it was visible. Observed live: alignment on disk was 19 hours stale
    while the tag had been tracking continuously, and a restart then reloaded
    the ancient index.

    This models the real Store timer semantics and asserts a write lands.
    """
    from types import SimpleNamespace

    from custom_components.bermuda.const import FINDMY_STORAGE_MIN_INTERVAL, FINDMY_STORAGE_SAVE_DELAY

    class FakeStore:
        """Store.async_delay_save's resetting-debounce behaviour."""

        def __init__(self):
            self.fire_at: float | None = None
            self.writes = 0

        def async_delay_save(self, _func, delay):
            # Every call pushes the deadline out - this is the trap.
            self.fire_at = NOW[0] + delay

        def tick(self):
            if self.fire_at is not None and NOW[0] >= self.fire_at:
                self.writes += 1
                self.fire_at = None

    NOW = [0.0]
    store = FakeStore()

    # A tag in continuous view: alignment goes dirty on every cycle.
    last_queued = 0.0
    cycle = 10.0  # coordinator cycles far more often than the debounce delay
    for _ in range(360):  # one hour of continuous sightings
        NOW[0] += cycle
        dirty = True
        if dirty and NOW[0] - last_queued >= FINDMY_STORAGE_MIN_INTERVAL:
            last_queued = NOW[0]
            store.async_delay_save(lambda: {}, FINDMY_STORAGE_SAVE_DELAY)
        store.tick()

    assert store.writes > 0, "throttled queueing must let the debounce actually fire"
    # Sanity: the throttle must be comfortably longer than the debounce, or the
    # same starvation returns.
    assert FINDMY_STORAGE_MIN_INTERVAL > FINDMY_STORAGE_SAVE_DELAY

    # And prove the old behaviour (queue every cycle) really did starve.
    NOW[0] = 0.0
    naive = FakeStore()
    for _ in range(360):
        NOW[0] += cycle
        naive.async_delay_save(lambda: {}, FINDMY_STORAGE_SAVE_DELAY)
        naive.tick()
    assert naive.writes == 0, "the un-throttled version should never have written"


def test_stale_alignment_cannot_lock_an_accessory_out():
    """
    A too-low alignment anchor must not put the real key index out of reach.

    Regression test, from a live lockout. Write starvation left an accessory's
    stored alignment at index 204 / 2026-08-08T19:25 while its real index had
    reached 299. A restart reloaded that stale anchor, and index_window()
    derived its ceiling purely from it: 204 + 78 elapsed + 2 = 284. The real
    index sat 15 above the ceiling, so no advert could ever match, and because
    update_alignment() refuses to move backwards nothing could raise the ceiling
    again - the accessory was locked out permanently.

    The pairing time is an independent, always-valid upper bound.
    """
    raw = json.loads(ACCESSORY_JSON)
    # Paired ~3 days before "now", which is where the real index comes from.
    paired = datetime(2026, 8, 6, 11, 45, tzinfo=UTC)
    raw["paired_at"] = paired.isoformat()
    raw["alignment_date"] = paired.isoformat()
    raw["alignment_index"] = 0
    acc = FindMyAccessoryKeys.from_json(json.dumps(raw))

    now = datetime(2026, 8, 9, 15, 0, tzinfo=UTC)
    real_index = int((now - paired) // timedelta(minutes=15))  # ~301

    # The stale anchor that caused the lockout.
    acc._alignment = (datetime(2026, 8, 8, 19, 25, tzinfo=UTC), 204)  # noqa: SLF001

    bottom, top = acc.index_window(now)
    assert top >= real_index, f"ceiling {top} must still reach the real index {real_index}"
    assert bottom <= real_index

    # The accessory must actually be findable at its real index.
    manager = BermudaFindMyManager()
    manager.add_accessory(acc)
    manager.build_table(now)
    assert manager.check_mac(acc._mac_at(real_index, KEY_TYPE_PRIMARY)) is not None  # noqa: SLF001

    # And a fresh, correct alignment must still win when it is ahead of pairing.
    acc._alignment = (now, real_index + 40)  # noqa: SLF001
    assert acc.max_index(now) == real_index + 40


def test_purge_removes_only_malformed_bluetooth_connections():
    """
    Malformed CONNECTION_BLUETOOTH tuples must be removed, real ones kept.

    Regression test: metadevices have no bluetooth address, but an earlier
    version let them fall through to the generic branch, which registered the
    metadevice id as a bluetooth connection. Device registry connections merge
    on update, so emitting the correct tuple never displaced the wrong one -
    it has to be removed explicitly. Connections are how HA matches devices
    across integrations, so a malformed one is not merely cosmetic.
    """
    from types import SimpleNamespace

    from homeassistant.helpers import device_registry as dr

    from custom_components.bermuda.coordinator import BermudaDataUpdateCoordinator

    bogus = (dr.CONNECTION_BLUETOOTH, "FINDMY_DC447A349C7243948C966A381492382D")
    good = (dr.CONNECTION_BLUETOOTH, "AA:BB:CC:DD:EE:FF")
    findmy_conn = ("findmy", "findmy_dc447a349c7243948c966a381492382d")

    devices = [
        SimpleNamespace(id="d1", name="Snowden", connections={bogus, findmy_conn}),
        SimpleNamespace(id="d2", name="A real scanner", connections={good}),
        SimpleNamespace(id="d3", name="Morticia", connections={bogus}),
    ]
    updates: dict[str, set] = {}

    coordinator = SimpleNamespace(
        dr=SimpleNamespace(
            async_update_device=lambda did, new_connections: updates.__setitem__(did, new_connections),
        ),
        config_entry=SimpleNamespace(entry_id="e1"),
    )

    import custom_components.bermuda.coordinator as coord_mod

    original = coord_mod.dr.async_entries_for_config_entry
    coord_mod.dr.async_entries_for_config_entry = lambda _reg, _eid: devices
    try:
        cleaned = BermudaDataUpdateCoordinator.async_purge_invalid_bluetooth_connections(coordinator)
    finally:
        coord_mod.dr.async_entries_for_config_entry = original

    assert cleaned == 2
    assert updates["d1"] == {findmy_conn}, "the namespaced findmy connection must survive"
    assert updates["d3"] == set()
    assert "d2" not in updates, "a real MAC must not be touched"
