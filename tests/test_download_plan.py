from scripts.download_models import MODELS


def test_download_plan_is_pinned_and_excludes_humanart():
    assert set(MODELS) == {"gemx", "sonic"}
    for spec in MODELS.values():
        assert len(spec["revision"]) == 40
        assert all("humanart" not in pattern.lower() for pattern in spec["files"])
        assert all("smpl" not in pattern.lower() for pattern in spec["files"])
