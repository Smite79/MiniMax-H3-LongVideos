"""Tests for the prompt engine.

Every case here is a bug that was REPORTED against the old engine. The engine was
replaced; the reports are the specification and they carry over.

Run: python test_engine.py
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import engine as E

_fails = []


def check(label, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else f"   {extra}"))
    if not ok:
        _fails.append(label)


def scene(beats, cast=("Ana", "Guard")):
    """Run beats through the state and return the continuity text per shot."""
    st = E.SceneState()
    out = []
    for i, b in enumerate(beats, 1):
        ch = st.read(b, cast=cast, shot=i)
        out.append(st.continuity(described=[n for n in cast], changed=ch))
    return st, out


def test_two_things_in_one_beat():
    """REPORTED: "she is still breaking handcuffs".

    A beat that cuffs the wrists and locks on a collar recorded whichever phrase
    was longer and dropped the other. From the next shot the handcuffs were not
    in the prompt at all -- not held, not named -- and hardware nobody mentions
    is hardware the model stops drawing."""
    print("\n=== two things in one beat ===")
    hw = E.hardware_in("The guard handcuffs Ana's wrists behind her back and "
                       "locks a steel collar around her neck.")
    names = [c for c, _p, _w in hw]
    check(f"both are read: {names}", "handcuffs" in names and "collar" in names)
    check("the material is kept", any(w == "steel collar" for _c, _p, w in hw))
    _st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back and locks a steel "
        "collar around her neck, chained to the wall.",
        "Ana sits against the wall.",
        "Ana pulls at the chain.",
        "Ana looks at the door.",
    ])
    for i, s in enumerate(shots[1:], 2):
        check(f"shot {i} names the handcuffs", "handcuffs" in s, s)
        check(f"shot {i} names the collar", "collar" in s, s)


def test_a_neck_is_not_behind_a_back():
    """REPORTED: nonsense in the hold clause.

    A limb position describes where the ARMS are. The old engine read the part
    from the item list and the position from the beat and joined them, giving
    "holding the neck behind the back"."""
    print("\n=== a neck is not behind a back ===")
    _st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back and locks a steel "
        "collar around her neck, chained to the wall.",
        "Ana sits against the wall.",
    ])
    s = shots[1]
    check("no neck behind a back", "neck behind the back" not in s, s)
    check("the wrists take the position", "wrists behind the back" in s, s)
    check("the collar takes the neck", "the neck" in s, s)
    check("...and the anchor", "fast to the wall" in s, s)
    # The constructor refuses it even if a caller tries.
    r = E.Restraint("collar", "neck", position="behind the back")
    check("a collar cannot hold a position", r.position == "")


def test_a_door_is_not_a_room():
    """REPORTED: "camera views are all screwed up".

    "door" was a place, so "Ana looks at the door" relocated the shot into a
    door. A place has to be somewhere a person can be, and behind a preposition
    that puts them there."""
    print("\n=== a door is not a room ===")
    for t in ("Ana looks at the door.", "Ana opens the door.",
              "Ana stares at the door.", "Ana knocks on the door."):
        check(f"no room in {t!r}", not E.place_in(t), repr(E.place_in(t)))
    check("a doorway is a place", E.place_in("Ana waits in the doorway.") == "doorway")
    for room in ("kitchen", "basement", "hallway", "bathroom", "cell"):
        check(f"{room} reads", E.place_in(f"Ana walks into the {room}.") == room)
    check("an adjective does not hide it",
          E.place_in("Ana walks into the tiled bathroom.") == "tiled bathroom")
    check("shallow depth of field is not a hall",
          not E.place_in("shallow depth of field, 35mm"))


def test_movement_is_not_fastening():
    """REPORTED indirectly: hardware latched over furniture somebody walked to.

    "to the <thing>" on its own is movement. The fastening verb is required, and
    a noun that is also a verb ("the chain", "the clip") must not supply it."""
    print("\n=== movement is not fastening ===")
    for t in ("chains it to the wall", "chained to the wall",
              "chains her collar to the wall", "secures the chain to the floor",
              "bolted to the ceiling", "clips the chain to a ring",
              "a chain runs from her collar to the wall",
              "a short chain holds her collar to the wall"):
        check(f"anchors: {t!r}", E.anchor_in(t), repr(E.anchor_in(t)))
    for t in ("she sinks to the floor", "he walks to the table",
              "she is dragged to the bed", "she falls to the floor",
              "she runs to the wall", "they move to the bed",
              "She drops the rope to the floor.", "The clip fell to the floor.",
              "He throws the chain to the floor.", "The chains hang by the door."):
        check(f"not an anchor: {t!r}", not E.anchor_in(t), repr(E.anchor_in(t)))


def test_the_agent_is_not_the_wearer():
    """REPORTED: a second figure invented to wear the hardware.

    "The guard cuffs Ana" puts them on Ana. Describing them on the guard gives
    the model wrists that belong to nobody the text put them on."""
    print("\n=== the agent is not the wearer ===")
    st, _ = scene(["The guard handcuffs Ana."])
    check("Ana is restrained", st.person("Ana").restrained())
    check("the guard is not", not st.person("Guard").restrained())


def test_it_comes_off_when_the_text_takes_it_off():
    """The latch has to release, or a script that unlocks the cuffs in its own
    prose goes on insisting they are fastened over hardware on the floor."""
    print("\n=== released when released ===")
    st, shots = scene([
        "The guard handcuffs Ana.",
        "Ana sits down.",
        "The guard unlocks the handcuffs and takes them off.",
        "Ana stands up.",
    ])
    check("held while on", "handcuffs" in shots[1], shots[1])
    check("gone after the release", "handcuffs" not in shots[3], shots[3])
    check("state agrees", not st.person("Ana").restrained())


def test_the_shot_that_applies_says_both_ends():
    """A standing hold on the shot that STAGES the fastening is a lie about the
    first frame: it says closed before it was closed, and the struggle then
    happens in whatever order is left over."""
    print("\n=== both ends on the applying shot ===")
    _st, shots = scene(["The guard handcuffs Ana.", "Ana sits down."])
    check("shot 1 says both ends", "off the body at the first frame" in shots[0],
          shots[0])
    check("shot 2 does not", "off the body at the first frame" not in shots[1],
          shots[1])
    check("shot 2 holds instead", "closed and fastened" in shots[1], shots[1])


def test_a_chain_is_a_tether_not_a_second_restraint():
    """"a collar chained to the wall" is ONE restraint with an anchor. Recording
    the chain separately gives the shot a chain holding the wrists beside a
    collar holding the neck, which is two things to draw where there is one."""
    print("\n=== a chain is a tether ===")
    st, shots = scene([
        "The guard locks a steel collar around Ana's neck and chains it to the wall.",
        "Ana sits down."])
    hw = list(st.person("Ana").hardware)
    check(f"one item, not two: {hw}", hw == ["collar"], str(hw))
    # Guarded: with the item reader broken there is no collar to index, and a
    # KeyError reports as a crashed suite rather than as the failure it is.
    check("it carries the anchor",
          "collar" in hw and st.person("Ana").hardware["collar"].anchor == "wall")
    check("the shot says so", "fast to the wall" in shots[1], shots[1])
    check("no loose chain in the text", "chain" not in shots[1], shots[1])


def test_a_modifier_belongs_to_its_own_item():
    """Two items and two modifiers in one beat. Giving both modifiers to both
    items produced handcuffs chained to a wall they were never near, and a
    collar held behind a back."""
    print("\n=== a modifier belongs to its own item ===")
    st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back and locks a steel "
        "collar around her neck, chained to the wall.",
        "Ana sits down."])
    hw = st.person("Ana").hardware
    # Guarded: with the item reader broken one of these is missing entirely, and
    # a KeyError reports as a crashed suite rather than as the failure it is.
    check(f"both items recorded: {list(hw)}", "handcuffs" in hw and "collar" in hw)
    c, k = hw.get("handcuffs"), hw.get("collar")
    check("cuffs take the position", bool(c) and c.position == "behind the back")
    check("...and not the wall", bool(c) and c.anchor == "")
    check("the collar takes the wall", bool(k) and k.anchor == "wall")
    check("...and not the position", bool(k) and k.position == "")
    s = shots[1]
    check("the shot reads correctly",
          "the wrists behind the back and the neck fast to the wall" in s, s)


def test_nothing_is_said_twice():
    """The old engine restated the same fact from several readers at once and
    the guards reached 65% of a shot against a 12% beat."""
    print("\n=== nothing said twice ===")
    _st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back.",
        "Ana sits against the wall.",
        "Ana twists her wrists in the cuffs.",
    ])
    for i, s in enumerate(shots[1:], 2):
        check(f"shot {i}: cuffs named once", s.lower().count("cuffs") <= 1, s)
        check(f"shot {i}: under 45 words", len(s.split()) < 45,
              f"{len(s.split())}w: {s}")


def test_a_garment_change_says_both_ends():
    """REPORTED: "a diaper instantly changed into shorts before the beat where
    she puts her shorts back on".

    A shot told only the RESULT is free to open with the result already true.
    Both ends, or the change happens whenever the model feels like it."""
    print("\n=== a garment change says both ends ===")
    st, shots = scene([
        "Ana stands in the room.",
        "Ana takes off her blue shorts.",
        "Ana sits down.",
        "Ana puts her blue shorts back on.",
        "Ana walks out.",
    ], cast=("Ana",))
    check("the removal shot says both ends",
          "on the body as the shot opens and fully off it by the last frame"
          in shots[1], shots[1])
    check("the next shot says it stays off",
          "off the body and stays where it was put" in shots[2], shots[2])
    check("the wearing shot says both ends",
          "off the body as the shot opens and fully on by the last frame"
          in shots[3], shots[3])
    check("...and not still-off as well",
          "off the body and stays where it was put" not in shots[3], shots[3])
    check("state agrees at the end",
          [g for g in st.person("Ana").worn if "shorts" in g]
          and not st.person("Ana").removed)
    check("one pair of shorts, not two",
          shots[3].lower().count("shorts") == 1, shots[3])


def test_pulled_aside_is_not_taken_off():
    """A garment moved aside is still ON. Recording it as removed loses it."""
    print("\n=== pulled aside is not off ===")
    st, shots = scene(["Ana pulls her skirt aside."], cast=("Ana",))
    check("it stays on the body", "stays on the body" in shots[0], shots[0])
    check("it is not removed", not st.person("Ana").removed)
    check("it is displaced", st.person("Ana").displaced)


def main():
    test_two_things_in_one_beat()
    test_a_neck_is_not_behind_a_back()
    test_a_door_is_not_a_room()
    test_movement_is_not_fastening()
    test_the_agent_is_not_the_wearer()
    test_it_comes_off_when_the_text_takes_it_off()
    test_the_shot_that_applies_says_both_ends()
    test_a_chain_is_a_tether_not_a_second_restraint()
    test_a_modifier_belongs_to_its_own_item()
    test_nothing_is_said_twice()
    test_a_garment_change_says_both_ends()
    test_pulled_aside_is_not_taken_off()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
    else:
        print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
