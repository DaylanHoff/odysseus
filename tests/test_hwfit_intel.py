"""Intel Arc/Xe hardware detection for Cookbook hwfit."""

import pytest

from services.hwfit import hardware


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    hardware._cache_by_host.clear()
    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_remote_port", None)
    monkeypatch.setattr(hardware, "_remote_platform", None)
    yield
    hardware._cache_by_host.clear()


def test_detect_intel_discrete_gpu_uses_vulkan_backend(monkeypatch):
    def fake_run(cmd):
        if cmd == ["ls", "/sys/class/drm"]:
            return "card0 card0-DP-1 renderD128"
        if cmd[0:2] == ["cat", "/sys/class/drm/card0/device/vendor"]:
            return "0x8086"
        if cmd[0:2] == ["cat", "/sys/class/drm/card0/device/mem_info_vram_total"]:
            return str(8 * 1024**3)
        if cmd[0:2] == ["cat", "/sys/class/drm/card0/device/product_name"]:
            return "Intel Arc A770"
        if cmd == ["which", "vulkaninfo"]:
            return "/usr/bin/vulkaninfo"
        return None

    monkeypatch.setattr(hardware, "_remote_host", "intel-box")
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(hardware, "_get_ram_gb", lambda: 32.0)

    info = hardware._detect_intel()
    assert info is not None
    assert info["gpu_name"] == "Intel Arc A770"
    assert info["gpu_count"] == 1
    assert info["gpu_vram_gb"] == 8.0
    assert info["backend"] == "vulkan"
    assert info["unified_memory"] is False


def test_detect_intel_unified_memory_caps_gtt_and_reports_cpu_backend_without_vulkan(monkeypatch):
    def fake_run(cmd):
        if cmd == ["ls", "/sys/class/drm"]:
            return "card0 card0-eDP-1 renderD128"
        if cmd[0:2] == ["cat", "/sys/class/drm/card0/device/vendor"]:
            return "0x8086"
        if cmd[0:2] == ["cat", "/sys/class/drm/card0/device/mem_info_gtt_total"]:
            return str(32 * 1024**3)
        if cmd[0:2] == ["cat", "/sys/class/drm/card0/device/product_name"]:
            return "Intel Arc 140V"
        # No vulkaninfo binary and no visible libvulkan.so in this fixture.
        if cmd == ["which", "vulkaninfo"] or cmd[0:2] == ["test", "-e"]:
            return None
        return None

    monkeypatch.setattr(hardware, "_remote_host", "intel-ultrabook")
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(hardware, "_get_ram_gb", lambda: 16.0)

    info = hardware._detect_intel()
    assert info is not None
    # 32 GiB GTT capped to 80% of 16 GiB system RAM => 12.8 GiB budget.
    assert info["gpu_vram_gb"] == 12.8
    assert info["unified_memory"] is True
    assert info["backend"] == "cpu_x86"


def test_detect_system_includes_intel_probe(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_apple_silicon", lambda: None)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "_detect_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_intel", lambda: {
        "gpu_name": "Intel Arc A770",
        "gpu_vram_gb": 8.0,
        "gpu_count": 1,
        "gpus": [{"index": 0, "name": "Intel Arc A770", "vram_gb": 8.0}],
        "gpu_groups": [{"name": "Intel Arc A770", "vram_each": 8.0, "count": 1, "indices": [0], "vram_total": 8.0}],
        "homogeneous": True,
        "backend": "vulkan",
        "unified_memory": False,
    })
    monkeypatch.setattr(hardware, "_get_ram_gb", lambda: 32.0)
    monkeypatch.setattr(hardware, "_get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr(hardware, "_get_cpu_count", lambda: 12)
    monkeypatch.setattr(hardware, "_get_cpu_name", lambda: "Intel Core Ultra")

    system = hardware.detect_system(fresh=True)

    assert system["has_gpu"] is True
    assert system["backend"] == "vulkan"
    assert system["gpu_name"] == "Intel Arc A770"
