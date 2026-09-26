"""Check spinner geometry, logo and kitty art, and label alignment."""

from agentperf_local.tui.branding import (
    AA_NEUTRAL_50,
    AA_NEUTRAL_500,
    KITTY_FRAMES,
    KITTY_HAPPY,
    PIXEL_LOGO_SMALL,
    SPINNER_FRAMES,
    key_value_block,
    spinner_frame,
)

HALF_BLOCK_CHARACTERS = {"█", "▀", "▄", " ", "\n"}


def _shape(art: str) -> tuple[int, int]:
    """Return (rows, columns), requiring every line to share one width."""
    lines = art.splitlines()
    widths = {len(line) for line in lines}
    assert len(widths) == 1
    return len(lines), widths.pop()


def test_spinner_ticks_cycle_distinct_one_cell_frames() -> None:
    cycle = [spinner_frame(tick) for tick in range(len(SPINNER_FRAMES))]
    assert len(set(cycle)) == len(SPINNER_FRAMES)
    assert all(len(frame) == 1 for frame in cycle)


def test_key_value_block_aligns_values_on_a_computed_muted_key_column() -> None:
    block = key_value_block((("AA upload", "off"), ("replay destination", "configured model server")))
    assert block.splitlines() == [
        f"[{AA_NEUTRAL_500}]AA upload         [/]  [{AA_NEUTRAL_50}]off[/]",
        f"[{AA_NEUTRAL_500}]replay destination[/]  [{AA_NEUTRAL_50}]configured model server[/]",
    ]


def test_logo_stays_terminal_safe_with_uniform_rows() -> None:
    assert set(PIXEL_LOGO_SMALL) <= HALF_BLOCK_CHARACTERS
    _shape(PIXEL_LOGO_SMALL)


def test_kitty_poses_share_one_ascii_shape() -> None:
    # Every pose is the same size, so a change of expression never moves nearby text.
    poses = (*(frame.art for frame in KITTY_FRAMES), KITTY_HAPPY)
    assert all(pose.isascii() for pose in poses)
    assert len({_shape(pose) for pose in poses}) == 1
