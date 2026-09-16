# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""``defineSgpr`` versus stale ``freeSgprVarPool`` entries (CPU-only, no assembler).

``addSgprVarToPool`` parks a permanent SGPR: the name goes into
``states.freeSgprVarPool`` and ``addFromCheckOut`` flips the slot back to
``Available`` so the temp allocator may reuse it. ``defineSgpr`` then re-pins every
parked name to ``InUse`` for the duration of its own ``checkOutAligned`` (so the new
variable cannot land on a slot a parked name still owns) and frees them again
afterwards.

That re-pin goes through ``removeFromCheckOut``, which requires the slot to be
``Available``. So a parked name whose slot has since been handed out to somebody else
is not merely stale bookkeeping -- it arms the *next* ``defineSgpr`` anywhere in the
kernel to throw. This is the failure that surfaced once ``CompactLoopStore`` became
the default and started calling ``defineSgpr`` during the store phase, while
wave-separated TDM had left ``GlobalReadIncsA`` parked:

    RegisterPool::removeFromCheckOut(26, 1, GlobalReadIncsA) is not in Available state

The invariant these tests pin down: a name may stay in ``freeSgprVarPool`` only while
its slot is still ``Available``. Whoever parks an SGPR owns unparking it before the
slot can be reused -- which is why ``endSummation`` unparks ``GlobalReadIncs*`` at the
summation/post-loop boundary, ahead of every store path.

No GPU and no assembler: this drives the real park/unpark helpers over a real rocisa
``RegisterPool``.
"""

from types import SimpleNamespace

import pytest

from Tensile.Common.RegisterPool import RegisterPool
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from rocisa.enum import RegisterType

pytestmark = pytest.mark.unit

# The names endSummation unparks, in the order it lists them.
_GR_INCS = ("GlobalReadIncsA", "GlobalReadIncsB", "GlobalReadIncsMXSA", "GlobalReadIncsMXSB")


class _SgprHarness:
    """Minimal stand-in for the writer, bound to the real park/unpark helpers.

    ``defineSgpr`` and the ``setSgprTo*State`` helpers reach for ``self.sgprPool``,
    ``self.sgprs`` and ``self.states.freeSgprVarPool`` and nothing else, so binding the
    production implementations here exercises them without constructing a kernel.
    """

    defineSgpr = KernelWriterAssembly.defineSgpr
    defineSgprIdx = KernelWriterAssembly.defineSgprIdx
    setSgprToInUseState = KernelWriterAssembly.setSgprToInUseState
    setSgprToFreeState = KernelWriterAssembly.setSgprToFreeState
    addSgprVarToPool = KernelWriterAssembly.addSgprVarToPool
    removeSgprVarFromPool = KernelWriterAssembly.removeSgprVarFromPool

    def __init__(self, size=64):
        self.sgprPool = RegisterPool(8, RegisterType.Sgpr, False)
        self.sgprPool.add(0, size, "init")
        self.sgprs = {}
        self.states = SimpleNamespace(freeSgprVarPool=set())

    def statusOf(self, name):
        return self.sgprPool.getPool()[self.sgprs[name]].status

    @property
    def parked(self):
        return self.states.freeSgprVarPool


@pytest.fixture
def hw():
    """A harness with one permanent SGPR ("GlobalReadIncsA") already defined."""
    h = _SgprHarness()
    h.defineSgpr("GlobalReadIncsA", 1)
    assert h.statusOf("GlobalReadIncsA") == RegisterPool.Status.InUse
    return h


def test_parking_frees_the_slot_but_keeps_the_name(hw):
    """The park step is what makes the slot reusable -- and the name outlive it."""
    hw.addSgprVarToPool("GlobalReadIncsA")
    assert hw.parked == {"GlobalReadIncsA"}
    assert hw.statusOf("GlobalReadIncsA") == RegisterPool.Status.Available


def test_define_sgpr_tolerates_a_parked_entry(hw):
    """With the slot untouched, defineSgpr re-pins and re-frees the parked name."""
    hw.addSgprVarToPool("GlobalReadIncsA")
    grIncsIdx = hw.sgprs["GlobalReadIncsA"]

    hw.defineSgpr("CLSm0Base", 1)

    # The re-pin is what stops the new variable from landing on the parked slot.
    assert hw.sgprs["CLSm0Base"] != grIncsIdx
    # ...and it is undone, so the slot stays available to the temp allocator.
    assert hw.parked == {"GlobalReadIncsA"}
    assert hw.statusOf("GlobalReadIncsA") == RegisterPool.Status.Available


def test_define_sgpr_raises_once_a_parked_slot_is_handed_out(hw):
    """The regressed failure: a parked name whose slot a temp already took.

    Nothing complains at the point of reuse -- the slot was legitimately
    ``Available``. The throw lands on the next ``defineSgpr``, arbitrarily far away,
    which is why this presented as a store-phase error about a global-read SGPR.
    """
    hw.addSgprVarToPool("GlobalReadIncsA")
    grIncsIdx = hw.sgprs["GlobalReadIncsA"]

    storeTmp = hw.sgprPool.checkOut(1, "storeTmp")
    assert storeTmp == grIncsIdx, "expected the temp to reuse the parked slot"

    with pytest.raises(RuntimeError, match="is not in Available state"):
        hw.defineSgpr("CLSm0Base", 1)


def test_unparking_before_the_slot_is_reused_keeps_define_sgpr_safe(hw):
    """The fix: drop the name while its slot is still Available, then release it.

    ``removeSgprVarFromPool`` re-pins the slot to ``InUse``, so endSummation's
    release walk (which only undefines ``InUse`` slots) frees it for real instead of
    skipping it. After that the slot is reusable with no name left pointing at it.
    """
    hw.addSgprVarToPool("GlobalReadIncsA")
    grIncsIdx = hw.sgprs["GlobalReadIncsA"]

    hw.removeSgprVarFromPool("GlobalReadIncsA")
    assert hw.parked == set()
    assert hw.statusOf("GlobalReadIncsA") == RegisterPool.Status.InUse

    # Stands in for the release walk's undefineSgpr.
    hw.sgprPool.checkIn(grIncsIdx)
    assert hw.statusOf("GlobalReadIncsA") == RegisterPool.Status.Available

    storeTmp = hw.sgprPool.checkOut(1, "storeTmp")
    assert storeTmp == grIncsIdx

    hw.defineSgpr("CLSm0Base", 1)
    assert hw.sgprs["CLSm0Base"] != storeTmp


def test_parked_names_left_at_the_store_boundary_all_own_available_slots(hw):
    """The general invariant, stated over the whole pool rather than one name.

    Any parked name with a non-Available slot is a latent throw in the next
    ``defineSgpr``. Asserting it this way keeps the check meaningful if other
    prologue SGPRs start being parked.
    """
    hw.addSgprVarToPool("GlobalReadIncsA")
    hw.sgprPool.checkOut(1, "storeTmp")

    stolen = [
        name for name in hw.parked
        if hw.statusOf(name) != RegisterPool.Status.Available
    ]
    assert stolen == ["GlobalReadIncsA"], "expected the harness to have staged the bug"

    hw.removeSgprVarFromPool("GlobalReadIncsA")
    assert [
        name for name in hw.parked
        if hw.statusOf(name) != RegisterPool.Status.Available
    ] == []


def test_remove_sgpr_var_from_pool_ignores_names_it_never_parked():
    """endSummation unparks all four GlobalReadIncs* unconditionally.

    Most kernels define neither MXSA/MXSB nor park any of them, so the helper has to
    no-op on both an undefined name and a defined-but-unparked one.
    """
    h = _SgprHarness()
    h.defineSgpr("GlobalReadIncsA", 1)

    for name in _GR_INCS:
        h.removeSgprVarFromPool(name)

    assert h.parked == set()
    assert h.statusOf("GlobalReadIncsA") == RegisterPool.Status.InUse
    assert "GlobalReadIncsMXSA" not in h.sgprs
